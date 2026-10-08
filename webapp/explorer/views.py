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
# than all of it.
SWEEP_MAX_POINTS = 20000
WEBGL_ABOVE = 20000
HOVER_CHARS_LARGE = 60
DEFAULT_PENALTY = 1.0

# On the shared server only one computation runs at a time: two at once would
# double the memory the cap below is sized for. A run holds this from the request
# until its thread finishes; a sweep for as long as it computes.
_BUSY = threading.Lock()
BUSY_MESSAGE = "חישוב אחר רץ כרגע על השרת. נסי שוב בעוד דקה."


def _selection_size(filenames, chunk_size, model):
    """(chunks, words) of a selection, read from the store's index - nothing loaded."""
    index = store.index_for(model, chunk_size)
    rows = [index[f] for f in filenames if f in index]
    return (sum(int(r.get("n_chunks", 0)) for r in rows),
            sum(int(r.get("words", 0)) for r in rows))


def _estimate_mb(chunks, words):
    """Peak memory of a run, in MB. Measured 8.10.2026 on runs of 6k-50k chunks at
    every chunk size and k up to 30 (the process itself is ~200 MB of libraries):
    the plot and the clustering grow with the number of chunks, the n-gram
    vocabulary with the number of words. Rounded up; it over-estimates slightly."""
    return 200 + 7.6 * chunks / 1000 + 248 * words / 1e6


def _too_large(filenames, chunk_size, model):
    """The refusal to show, or None when the selection fits the server."""
    limit = getattr(settings, "MAX_RUN_MB", None)
    if not limit:
        return None
    chunks, words = _selection_size(filenames, chunk_size, model)
    if _estimate_mb(chunks, words) <= limit:
        return None
    # How much of this selection would fit, at this chunk size.
    share = (limit - 200) / max(_estimate_mb(chunks, words) - 200, 1)
    return ("הבחירה גדולה מדי לשרת: %s צ'אנקים, %s מילים. "
            "השרת מחזיק בערך %d%% ממנה. אפשר לבחור פחות טקסטים, "
            "או צ'אנקים גדולים יותר (פחות צ'אנקים לאותו טקסט)."
            % (format(chunks, ","), format(words, ","), int(share * 100)))


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

    form_back = dict(selected=set(filenames), k=k, grouping_level=grouping_level,
                     smoothing=smoothing, want_cpd=want_cpd,
                     block_size=block_size, penalty=penalty)
    too_large = _too_large(filenames, chunk_size, model)
    if too_large:
        return render(request, "explorer/index.html",
                      _context(chunk_size, model=model, error=too_large, **form_back))

    if request.POST.get("action") == "sweep":
        if not _BUSY.acquire(blocking=False):
            return render(request, "explorer/index.html",
                          _context(chunk_size, model=model, error=BUSY_MESSAGE, **form_back))
        try:
            df = store.load_selection(filenames, chunk_size, model)
            sweep_html, sweep = _sweep(df, k)
        finally:
            _BUSY.release()
        return render(request, "explorer/index.html",
                      _context(chunk_size, model=model, figure_html=sweep_html, sweep=sweep,
                               selected=set(filenames), k=k, grouping_level=grouping_level,
                               smoothing=smoothing, want_cpd=want_cpd,
                               block_size=block_size, penalty=penalty))

    # Hand the work to a thread and give the page a job to watch. Anything large
    # takes minutes, and a request that returns only at the end cannot say so.
    form = {
        "chunk_size": chunk_size, "model": model, "selected": sorted(filenames), "k": k,
        "grouping_level": grouping_level, "smoothing": smoothing,
        "want_cpd": want_cpd, "block_size": block_size, "penalty": penalty,
    }
    # The model belongs in the key: the same selection read in the two spaces is two
    # different runs, and serving one for the other would be silently wrong. Whether
    # the change points are *shown* does not: they are always computed, and the page
    # toggles them, so a run differing only in that is the same run.
    key = _cache_key(filenames, chunk_size, k, smoothing, grouping_level,
                     True, block_size, penalty, model)
    cached = _cached_token(key)
    if cached:
        return _render_result(request, cached, from_cache=True)

    if not _BUSY.acquire(blocking=False):
        return render(request, "explorer/index.html",
                      _context(chunk_size, model=model, error=BUSY_MESSAGE, **form_back))
    try:
        df = store.load_selection(filenames, chunk_size, model)
    except Exception:
        _BUSY.release()
        raise
    if len(df) < k:
        _BUSY.release()
        return render(request, "explorer/index.html",
                      _context(chunk_size, model=model,
                               error="בבחירה הזו יש %d צ'אנקים בלבד. "
                                     "בחרי יותר טקסט או k קטן יותר." % len(df), **form_back))

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
        # Vectors stay numpy arrays: the one reader of `vec` downstream
        # (compute_chunk_cluster_affinities) takes them as they are, and
        # stringify_vectors exists for CSV round trips this app never makes.
        df = cvr.prepare_dataframe(df, grouping_level=grouping_level)
        df = cvr.merge_small_groups(df)
        df = cvr.compute_density_original(df)

        # The figure's hover layer reads these two: the per-chunk affinities and the
        # cluster's characteristic n-grams. Skipping them costs the hover, not the plot.
        jobs.set_stage(job_id, "שיוך לקלאסטרים")
        df = cvr.compute_chunk_cluster_affinities(df)
        jobs.set_stage(job_id, "צירופי מילים אופייניים")
        df = cvr.add_cluster_ctfidf_ngrams(df, ngram_n=3, top_k=20)

        # Always computed, never conditional: PELT over the block composition costs a
        # few seconds against minutes for the rest of the run, and computing it here
        # is what lets the checkbox show and hide the result without running again.
        jobs.set_stage(job_id, "נקודות שינוי")
        try:
            df, change_points = cvr.compute_change_points(df, block_size=block_size,
                                                          penalty=penalty)
        except Exception:
            change_points = None

        # The table goes out before the figure is drawn, so that the columns only
        # it needs - the full text and the vectors - can be let go first: plotly
        # takes a copy of the frame it is given, and on all of Midrash those two
        # columns are a few hundred MB the figure never reads.
        token = uuid.uuid4().hex[:12]
        out_dir = pathlib.Path(settings.RUNS_DIR) / token
        out_dir.mkdir(parents=True, exist_ok=True)
        columns = [c for c in ("filename", "hierarchy", "chunk_number", "chronological_index",
                               "scluster", "segment", "group_label", "chunk") if c in df.columns]
        df[columns].to_csv(out_dir / "chunks.csv", index=False, encoding="utf-8-sig")
        # One chunk at a time: .str.split() on the whole column held every word of
        # the selection as a string at once. Same count, NaN skipped as before.
        n_words = sum(len(c.split()) for c in df["chunk"] if isinstance(c, str))
        df = df.drop(columns=[c for c in ("vec", "chunk", "hierarchy_list") if c in df.columns])

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

        # build_figure draws a change point as a red dashed line with a red score
        # beside it. Collecting those indices lets the page hide and show them with
        # one relayout, instead of asking for the whole run again.
        cp_shapes = [i for i, s in enumerate(fig.layout.shapes or ())
                     if str(getattr(s.line, "color", "")) == "red"]
        cp_notes = [i for i, a in enumerate(fig.layout.annotations or ())
                    if str(getattr(a, "arrowcolor", "")) == "red"]
        if not want_cpd:
            fig.update_layout(
                shapes=[s.update(visible=False) if i in cp_shapes else s
                        for i, s in enumerate(fig.layout.shapes or ())],
                annotations=[a.update(visible=False) if i in cp_notes else a
                             for i, a in enumerate(fig.layout.annotations or ())])

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
            "words": n_words,
            "clusters": int(df["scluster"].nunique()),
            "change_points": len(change_points) if change_points else 0,
            "seconds": round(time.time() - t0, 1),
            "webgl": render_mode == "webgl",
            "cpd_requested": want_cpd,
            "penalty": penalty,
            "block_size": block_size,
            "cp_shapes": cp_shapes,
            "cp_notes": cp_notes,
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
    finally:
        _BUSY.release()


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
    """Inertia across k, read twice: where the fall flattens, and what each k still buys.

    Silhouette used to be shown beside this and has been dropped. On embeddings of this
    kind it almost always peaks at k = 2, because it rewards splitting the cloud into two
    well-separated lobes — a fact about the geometry of the vector space, not about the
    number of registers in the material. At the resolution this tool is used at it pointed
    at an answer nobody would act on, so it was reading as noise in the interface.

    Both panels here come from the same inertia curve. The first shows the curve with the
    chord between its ends, because the elbow is defined against that chord. The second
    shows what the curve implies but does not display: the share of the remaining spread
    that each additional cluster removes. That is what "the fall stops paying" means, and
    it is a number a reader can act on.
    """
    from sklearn.cluster import KMeans
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    X = np.stack(df["vec"].to_numpy())
    rng = np.random.default_rng(42)
    if len(X) > SWEEP_MAX_POINTS:
        X = X[rng.choice(len(X), SWEEP_MAX_POINTS, replace=False)]

    ks, inertia = [], []
    t0 = time.time()
    for k in range(k_min, k_max + 1):
        ks.append(k)
        inertia.append(float(KMeans(n_clusters=k, random_state=42, n_init="auto")
                             .fit(X).inertia_))

    # The elbow, stated rather than eyeballed: the k whose point lies furthest from the
    # straight line joining the ends of the inertia curve.
    x = np.array(ks, dtype=float)
    y = np.array(inertia, dtype=float)
    xn = (x - x[0]) / (x[-1] - x[0])
    yn = (y - y[-1]) / (y[0] - y[-1])
    bend = np.abs(yn - (1 - xn)) / np.sqrt(2)
    elbow = ks[int(np.argmax(bend))]
    # A range rather than one number: every k that bends the curve nearly as hard as the
    # elbow does is as defensible a choice on this evidence.
    good = [k for k, b in zip(ks, bend) if b >= bend.max() * 0.85]
    band = (min(good), max(good))
    # What each extra cluster buys, as a percentage of the spread still unexplained.
    gain = [0.0] + [100 * (inertia[i - 1] - inertia[i]) / inertia[i - 1]
                    for i in range(1, len(inertia))]

    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "המרפק — היכן הירידה מתיישרת", "מה כל קלאסטר נוסף עוד מסביר"))
    fig.add_trace(go.Scatter(x=[ks[0], ks[-1]], y=[inertia[0], inertia[-1]], mode="lines",
                             line=dict(color="#c9c2b6", width=1, dash="dash"),
                             hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ks, y=inertia, mode="lines+markers",
                             line=dict(color="#2d5a6b", width=2),
                             hovertemplate="k = %{x}<br>אינרציה %{y:.0f}<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=ks[1:], y=gain[1:], marker_color="#9c6b1f", opacity=.8,
                         hovertemplate="k = %{x}<br>%{y:.1f}%% מהפיזור שנותר<extra></extra>"),
                  row=1, col=2)
    for col in (1, 2):
        fig.add_vline(x=current_k, line=dict(color="#8a8378", width=1, dash="dot"),
                      row=1, col=col)
        fig.add_vline(x=elbow, line=dict(color="#8c2f28", width=2), row=1, col=col)
        fig.update_xaxes(title_text="k", row=1, col=col)
    fig.update_yaxes(title_text="אינרציה", row=1, col=1)
    fig.update_yaxes(title_text="% מהפיזור שנותר", row=1, col=2)
    fig.update_layout(height=340, showlegend=False, margin=dict(t=50, b=40, l=60, r=20),
                      paper_bgcolor="#f7f4ee", plot_bgcolor="#f7f4ee",
                      font=dict(family="Heebo, Segoe UI, sans-serif", size=12))

    at = ks.index(current_k) if current_k in ks else None
    sweep = {"points": len(X), "k_min": k_min, "k_max": k_max,
             "current_k": current_k, "seconds": round(time.time() - t0, 1),
             "sampled": len(df) > SWEEP_MAX_POINTS, "elbow": elbow,
             "band_low": band[0], "band_high": band[1],
             "gain_at_elbow": round(gain[ks.index(elbow)], 1),
             "gain_at_current": round(gain[at], 1) if at else None,
             "gain_last": round(gain[-1], 1),
             # A curve with no real bend: the elbow is then a formality, not a finding.
             "flat": float(bend.max()) < 0.08}
    return fig.to_html(full_html=False, include_plotlyjs="cdn",
                       config={"displaylogo": False, "responsive": True}), sweep


def download(request, token, name):
    path = pathlib.Path(settings.RUNS_DIR) / token / name
    if name not in {"chunks.csv", "figure.html", "change_points.csv"} or not path.exists():
        raise Http404
    return FileResponse(open(path, "rb"), as_attachment=True, filename=name)
