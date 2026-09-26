"""The explorer: choose a slice of the corpus, cluster it, look at it, take it away.

Everything expensive that does not depend on the parameters — the embeddings —
is read from the store. What a request actually costs is one KMeans over the
selected chunks, which is why the parameters here are the cheap ones.
"""
import hashlib
import json
import pathlib
import threading
import time
import uuid

import numpy as np
import pandas as pd
from django.conf import settings
from django.http import FileResponse, Http404, JsonResponse
from django.shortcuts import render

import pipeline_core
import cluster_visualization_report as cvr

from . import jobs, store

K_RANGE = list(range(2, 31))
DEFAULT_K = 10
GROUPING_LEVELS = [(1, "קורפוס"), (2, "תת-קורפוס"), (3, "חיבור")]

# The sweep fits KMeans once per k, so it runs on a sample of the selection rather
# than all of it; silhouette is quadratic in the sample and takes a smaller one.
SWEEP_MAX_POINTS = 20000
SILHOUETTE_SAMPLE = 2000
WEBGL_ABOVE = 20000
HOVER_CHARS_LARGE = 60
DEFAULT_PENALTY = 1.0


def _cache_key(filenames, chunk_size, k, smoothing, grouping_level, want_cpd,
               block_size, penalty, model):
    """Identical parameters produce an identical run, so they need computing once."""
    payload = json.dumps({
        "files": sorted(filenames), "chunk_size": chunk_size, "k": k, "model": model,
        "smoothing": smoothing, "grouping_level": grouping_level,
        "cpd": [want_cpd, block_size, penalty],
    }, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _cache_index():
    path = pathlib.Path(settings.RUNS_DIR) / "index.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            return {}
    return {}


def _cache_put(key, token):
    path = pathlib.Path(settings.RUNS_DIR) / "index.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    index = _cache_index()
    index[key] = token
    path.write_text(json.dumps(index, indent=1), encoding="utf-8")


def _cached_token(key):
    """The token of a finished run with these parameters, if its files are still there."""
    token = _cache_index().get(key)
    if token and (pathlib.Path(settings.RUNS_DIR) / token / "summary.json").exists():
        return token
    return None


def _catalogue(model, chunk_size, selected=()):
    df = store.catalogue(model, chunk_size)
    if df.empty:
        return []
    out = []
    for corpus, rows in df.groupby("corpus", sort=True):
        files = rows.to_dict("records")
        picked = sum(1 for f in files if f["filename"] in selected)
        out.append({
            "corpus": corpus,
            "corpus_he": rows["corpus_he"].iloc[0],
            "words": int(rows["words"].sum()),
            "files": files,
            "picked": picked,
            # A group holding part of the current selection opens by itself: a check
            # mark inside a collapsed section is the same as no check mark at all.
            "open": picked > 0,
        })
    return out


def _context(chunk_size=None, model=None, **extra):
    all_models = store.models()
    model = model or (all_models[0]["name"] if all_models else None)
    sizes = store.available_chunk_sizes(model) if model else []
    chunk_size = chunk_size if chunk_size in sizes else (sizes[0] if sizes else None)
    selected = extra.get("selected") or set()
    ctx = {
        "models": all_models,
        "model": model,
        "model_label": next((m["label"] for m in all_models if m["name"] == model), model),
        "sizes": sizes,
        "chunk_size": chunk_size,
        "groups": _catalogue(model, chunk_size, selected) if chunk_size else [],
        "k_range": K_RANGE,
        "grouping_levels": GROUPING_LEVELS,
        "store_dir": str(store.STORE_DIR),
        # Defaults, so a page rendered after a run still shows the form the way the
        # run was made rather than falling back to the first option in every select.
        "k": DEFAULT_K,
        "grouping_level": 3,
        "block_size": 50,
        "penalty": DEFAULT_PENALTY,
        "selected": set(),
    }
    ctx.update(extra)
    return ctx


def index(request):
    # Switching the chunk size re-posts the form here: keep the chosen size and the
    # selection, since the file list itself depends on which store is being read.
    data = request.POST if request.method == "POST" else request.GET
    size = data.get("chunk_size")
    return render(request, "explorer/index.html",
                  _context(int(size) if size else None, model=data.get("model") or None,
                           selected=set(data.getlist("files")),
                           k=int(data.get("k", DEFAULT_K)),
                           grouping_level=int(data.get("grouping_level", 3))))


def run(request):
    if request.method != "POST":
        return render(request, "explorer/index.html", _context())

    model = request.POST.get("model") or store.default_model()
    chunk_size = int(request.POST.get("chunk_size"))
    filenames = request.POST.getlist("files")
    k = int(request.POST.get("k", DEFAULT_K))
    grouping_level = int(request.POST.get("grouping_level", 3))
    smoothing = request.POST.get("smoothing") == "on"
    want_cpd = request.POST.get("change_points") == "on"
    block_size = int(request.POST.get("block_size", 50))
    # The signal PELT runs on is a vector of cluster proportions, so its values sit
    # between 0 and 1 and the cost of a boundary is small; penalties above about 2
    # return nothing at all on a corpus of this size.
    penalty = float(request.POST.get("penalty", DEFAULT_PENALTY))

    if not filenames:
        return render(request, "explorer/index.html",
                      _context(chunk_size, model=model, error="יש לבחור טקסט אחד לפחות."))

    df = store.load_selection(filenames, chunk_size, model)

    if request.POST.get("action") == "sweep":
        sweep_html, sweep = _sweep(df, k)
        return render(request, "explorer/index.html",
                      _context(chunk_size, model=model, figure_html=sweep_html, sweep=sweep,
                               selected=set(filenames), k=k, grouping_level=grouping_level,
                               smoothing=smoothing, want_cpd=want_cpd,
                               block_size=block_size, penalty=penalty))

    if len(df) < k:
        return render(request, "explorer/index.html",
                      _context(chunk_size, model=model,
                               error="בבחירה הזו יש %d צ'אנקים בלבד. "
                                     "בחרי יותר טקסט או k קטן יותר." % len(df)))

    # Hand the work to a thread and give the page a job to watch. Anything large
    # takes minutes, and a request that returns only at the end cannot say so.
    form = {
        "chunk_size": chunk_size, "model": model, "selected": sorted(filenames), "k": k,
        "grouping_level": grouping_level, "smoothing": smoothing,
        "want_cpd": want_cpd, "block_size": block_size, "penalty": penalty,
    }
    # The model belongs in the key: the same selection read in the two spaces is two
    # different runs, and serving one for the other would be silently wrong.
    key = _cache_key(filenames, chunk_size, k, smoothing, grouping_level,
                     want_cpd, block_size, penalty, model)
    cached = _cached_token(key)
    if cached:
        return _render_result(request, cached, from_cache=True)

    job_id = jobs.create(len(df), k, want_cpd)
    jobs.JOBS[job_id]["form"] = form
    jobs.JOBS[job_id]["key"] = key
    threading.Thread(target=_compute, daemon=True, args=(
        job_id, df, k, smoothing, grouping_level, want_cpd, block_size, penalty)).start()
    return render(request, "explorer/index.html",
                  _context(chunk_size, job_id=job_id, selected=set(filenames), k=k,
                           grouping_level=grouping_level, smoothing=smoothing,
                           want_cpd=want_cpd, block_size=block_size, penalty=penalty))


def _compute(job_id, df, k, smoothing, grouping_level, want_cpd, block_size, penalty):
    try:
        t0 = time.time()
        jobs.set_stage(job_id, "קלאסטרינג")
        df = pipeline_core.cluster_chunks(df, k=k, apply_smoothing=smoothing)
        df = pipeline_core.stringify_vectors(df)
        df = cvr.prepare_dataframe(df, grouping_level=grouping_level)
        df = cvr.merge_small_groups(df)
        df = cvr.compute_density_original(df)

        # The figure's hover layer reads these two: the per-chunk affinities and the
        # cluster's characteristic n-grams. Skipping them costs the hover, not the plot.
        jobs.set_stage(job_id, "שיוך לקלאסטרים")
        df = cvr.compute_chunk_cluster_affinities(df)
        jobs.set_stage(job_id, "צירופי מילים אופייניים")
        df = cvr.add_cluster_ctfidf_ngrams(df, ngram_n=3, top_k=20)

        change_points = None
        if want_cpd:
            jobs.set_stage(job_id, "נקודות שינוי")
            df, change_points = cvr.compute_change_points(df, block_size=block_size, penalty=penalty)

        jobs.set_stage(job_id, "ציור")
        # Past this many points the SVG layer makes hovering and panning unusable.
        render_mode = "webgl" if len(df) > WEBGL_ABOVE else "svg"
        # Every point carries its own hover text, so on a large selection the snippet
        # length, not the number of points, is what decides the weight of the page.
        if len(df) > WEBGL_ABOVE and "text_start" in df.columns:
            df["text_start"] = df["text_start"].str.slice(0, HOVER_CHARS_LARGE)
        fig = cvr.build_figure(df, change_points=change_points, render_mode=render_mode)
        # Wheel zoom, and a figure that keeps its view when the page re-renders.
        fig.update_layout(hovermode="closest", hoverdistance=12,
                          uirevision="explorer")

        token = uuid.uuid4().hex[:12]
        out_dir = pathlib.Path(settings.RUNS_DIR) / token
        out_dir.mkdir(parents=True, exist_ok=True)
        columns = [c for c in ("filename", "hierarchy", "chunk_number", "chronological_index",
                               "scluster", "segment", "group_label", "chunk") if c in df.columns]
        df[columns].to_csv(out_dir / "chunks.csv", index=False, encoding="utf-8-sig")
        plot_config = {"displaylogo": False, "responsive": True, "scrollZoom": True,
                       "doubleClick": "reset", "modeBarButtonsToAdd": ["drawopenpath", "eraseshape"]}
        fig.write_html(out_dir / "figure.html", include_plotlyjs="cdn", config=plot_config)
        # Plotly's own HTML keeps the hover layer, which is what the Streamlit
        # embedding lost; the figure is inserted into the page as-is.
        (out_dir / "_embed.html").write_text(
            fig.to_html(full_html=False, include_plotlyjs="cdn", config=plot_config),
            encoding="utf-8")
        if change_points:
            rows = [{"chronological_index": ci, "score": round(float(score), 4),
                     "top_clusters": "; ".join("%s %+.3f" % (c, d) for c, d in contrib[:5])}
                    for ci, score, contrib in change_points]
            pd.DataFrame(rows).to_csv(out_dir / "change_points.csv", index=False,
                                      encoding="utf-8-sig")

        summary = {
            "chunks": len(df),
            "files": df["filename"].nunique(),
            "words": int(df["chunk"].str.split().str.len().sum()),
            "clusters": int(df["scluster"].nunique()),
            "change_points": len(change_points) if change_points else 0,
            "seconds": round(time.time() - t0, 1),
            "webgl": render_mode == "webgl",
            "cpd_requested": want_cpd,
            "penalty": penalty,
        }
        # Written beside the outputs so a result can be re-served without the job,
        # which is what makes the cache survive a restart.
        (out_dir / "summary.json").write_text(
            json.dumps({"summary": summary, "form": jobs.JOBS[job_id].get("form", {})},
                       ensure_ascii=False), encoding="utf-8")
        _cache_put(jobs.JOBS[job_id].get("key", token), token)
        jobs.finish(job_id, token, summary)
    except Exception as exc:                                  # surfaced on the page
        jobs.fail(job_id, "%s: %s" % (type(exc).__name__, exc))


def progress(request, job_id):
    state = jobs.status(job_id)
    if state is None:
        raise Http404
    return JsonResponse(state)


def result(request, job_id):
    """A finished run, addressed either by its job or by its run token."""
    state = jobs.status(job_id)
    if state is not None:
        if not state["done"]:
            raise Http404
        if state["error"]:
            return render(request, "explorer/index.html", _context(error=state["error"]))
        return _render_result(request, state["token"])
    return _render_result(request, job_id)


def _render_result(request, token, from_cache=False):
    out_dir = pathlib.Path(settings.RUNS_DIR) / token
    if not (out_dir / "summary.json").exists():
        raise Http404
    stored = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    form = dict(stored.get("form") or {})
    form["selected"] = set(form.get("selected") or [])
    chunk_size = form.pop("chunk_size", None)
    form.setdefault("model", store.default_model())
    downloads = [(p.name, p.stat().st_size) for p in sorted(out_dir.iterdir())
                 if not p.name.startswith("_") and p.suffix != ".json"]
    return render(request, "explorer/index.html",
                  _context(chunk_size, figure_html=(out_dir / "_embed.html").read_text(encoding="utf-8"),
                           summary=stored["summary"], token=token, downloads=downloads,
                           from_cache=from_cache, **form))


def _sweep(df, current_k, k_min=2, k_max=20):
    """Inertia and silhouette across k, the two readings the choice of k rests on.

    Inertia always falls as k rises, so what it shows is where the fall stops paying
    — the elbow. Silhouette has a maximum, and it is the one that says which k the
    data actually support. They are plotted separately: the two have no common
    scale, and drawing them on one pair of axes would invent a comparison.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    X = np.stack(df["vec"].to_numpy())
    rng = np.random.default_rng(42)
    if len(X) > SWEEP_MAX_POINTS:
        X = X[rng.choice(len(X), SWEEP_MAX_POINTS, replace=False)]

    ks, inertia, silhouette = [], [], []
    t0 = time.time()
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
        ks.append(k)
        inertia.append(float(km.inertia_))
        sample = min(SILHOUETTE_SAMPLE, len(X))
        silhouette.append(float(silhouette_score(X, km.labels_, sample_size=sample, random_state=42)))

    best = ks[int(np.argmax(silhouette))]
    # The elbow, stated rather than eyeballed: the k whose point lies furthest from the
    # straight line joining the ends of the inertia curve.
    x = np.array(ks, dtype=float)
    y = np.array(inertia, dtype=float)
    xn = (x - x[0]) / (x[-1] - x[0])
    yn = (y - y[-1]) / (y[0] - y[-1])
    elbow = ks[int(np.argmax(np.abs(yn - (1 - xn)) / np.sqrt(2)))]
    # A range worth trying, rather than one number: everything within 5% of the best
    # silhouette is, on this evidence, as defensible as the maximum.
    good = [k for k, s in zip(ks, silhouette) if s >= max(silhouette) * 0.95]
    band = (min(good), max(good))
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Inertia — where the fall flattens", "Silhouette — higher is better separated"))
    fig.add_trace(go.Scatter(x=ks, y=inertia, mode="lines+markers", name="inertia",
                             line=dict(color="#2a78d6", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ks, y=silhouette, mode="lines+markers", name="silhouette",
                             line=dict(color="#eb6834", width=2)), row=1, col=2)
    for col in (1, 2):
        fig.add_vline(x=current_k, line=dict(color="#767f79", width=1, dash="dot"), row=1, col=col)
        fig.update_xaxes(title_text="k", row=1, col=col)
    fig.add_vline(x=best, line=dict(color="#1d6360", width=2), row=1, col=2)
    fig.update_layout(height=340, showlegend=False, margin=dict(t=50, b=40, l=50, r=20))

    fig.add_vline(x=elbow, line=dict(color="#1d6360", width=2), row=1, col=1)
    sweep = {"points": len(X), "k_min": k_min, "k_max": k_max, "best": best,
             "best_score": round(max(silhouette), 3), "at_current": round(
                 silhouette[ks.index(current_k)], 3) if current_k in ks else None,
             "current_k": current_k, "seconds": round(time.time() - t0, 1),
             "sampled": len(df) > SWEEP_MAX_POINTS, "elbow": elbow,
             "band_low": band[0], "band_high": band[1],
             "agree": abs(elbow - best) <= 2,
             "weak": max(silhouette) < 0.15}
    return fig.to_html(full_html=False, include_plotlyjs="cdn",
                       config={"displaylogo": False, "responsive": True}), sweep


def download(request, token, name):
    path = pathlib.Path(settings.RUNS_DIR) / token / name
    if name not in {"chunks.csv", "figure.html", "change_points.csv"} or not path.exists():
        raise Http404
    return FileResponse(open(path, "rb"), as_attachment=True, filename=name)
