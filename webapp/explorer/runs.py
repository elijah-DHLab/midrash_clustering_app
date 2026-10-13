"""The experiment log: every run kept, with everything known about it.

A run already leaves a directory under RUNS_DIR - its figure, its CSVs and summary.json,
which the result page is drawn from. This module adds the record that makes the run an
experiment one can come back to: when it ran and for how long, every parameter, the books
it was run on, which code, store and model produced it, and what came out - the clusters,
the change points and where in the text they fall. The record is run.json, written once
when the run finishes; the reader's own name, notes and tags for it are meta.json beside
it, so editing them never touches the record.

Nothing is deleted. Runs made before the log existed are read from their summary.json,
with the time taken from the file and the versions left unknown.
"""
import datetime
import json
import pathlib
import re
import subprocess
from zoneinfo import ZoneInfo

import pandas as pd
from django.conf import settings

from . import store

RUN_FILE = "run.json"
META_FILE = "meta.json"
LOCAL_TZ = ZoneInfo("Asia/Jerusalem")
GROUPING_LABELS = {1: "קורפוס", 2: "תת-קורפוס", 3: "חיבור"}


def _code_version():
    """The commit the server runs, and whether it has uncommitted changes."""
    root = pathlib.Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(["git", "-C", str(root), "log", "-1", "--format=%h %cs"],
                                capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""
    return commit + (" + שינויים מקומיים" if dirty else "")


CODE_VERSION = _code_version()


def _dir(token):
    return pathlib.Path(settings.RUNS_DIR) / token


def _model_meta(model):
    path = store.STORE_DIR / str(model) / "model.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _iso(ts):
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).isoformat(timespec="seconds")


def write(token, form, summary, started, finished, change_points=None):
    """The record of a finished run. `change_points` are the rows of change_points.csv."""
    model = form.get("model")
    names = store.titles_for(model)
    files = sorted(form.get("selected") or [])
    record = {
        "token": token,
        "started": _iso(started),
        "finished": _iso(finished),
        "seconds": round(finished - started, 1),
        "params": {
            "model": model,
            "chunk_size": form.get("chunk_size"),
            "k": form.get("k"),
            "smoothing": bool(form.get("smoothing")),
            "grouping_level": form.get("grouping_level"),
            "show_change_points": bool(form.get("want_cpd")),
            "block_size": form.get("block_size"),
            "penalty": form.get("penalty"),
        },
        "selection": {
            "files": files,
            "titles": [names.get(f, {}).get("he") or f for f in files],
            "chunks": summary.get("chunks"),
            "words": summary.get("words"),
        },
        "versions": {
            "code": CODE_VERSION,
            "store": model,
            "store_label": _model_meta(model).get("label", model),
            "store_note": _model_meta(model).get("note", ""),
            "fasttext": _model_meta(model).get("model_file", ""),
        },
        "results": _results(summary, change_points),
    }
    (_dir(token) / RUN_FILE).write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return record


def _results(summary, change_points):
    clusters = summary.get("clusters_table") or []
    cohesion = [c["cohesion"] for c in clusters if c.get("cohesion") is not None]
    return {
        "clusters": summary.get("clusters"),
        "change_points": summary.get("change_points", 0),
        "change_point_list": change_points or [],
        "cohesion_mean": round(sum(cohesion) / len(cohesion), 3) if cohesion else None,
        "largest_cluster_share": max((c["share"] for c in clusters), default=None),
    }


def _from_summary(token):
    """A run made before the log: rebuilt from summary.json and the files beside it."""
    d = _dir(token)
    stored = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    summary, form = stored.get("summary") or {}, stored.get("form") or {}
    when = (d / "summary.json").stat().st_mtime
    cps = []
    if (d / "change_points.csv").exists():
        try:
            cps = pd.read_csv(d / "change_points.csv").fillna("").to_dict("records")
        except (OSError, ValueError, pd.errors.ParserError):
            cps = []
    # Runs from before the store could be chosen did not record which one they read.
    model = form.get("model")
    names = store.titles_for(model) if model else {}
    files = sorted(form.get("selected") or [])
    return {
        "token": token,
        "started": _iso(when - (summary.get("seconds") or 0)),
        "finished": _iso(when),
        "seconds": summary.get("seconds"),
        "imported": True,
        "params": {
            "model": model, "chunk_size": form.get("chunk_size"), "k": form.get("k"),
            "smoothing": bool(form.get("smoothing")), "grouping_level": form.get("grouping_level"),
            "show_change_points": bool(form.get("want_cpd")),
            "block_size": form.get("block_size"), "penalty": form.get("penalty"),
        },
        "selection": {
            "files": files, "titles": [names.get(f, {}).get("he") or f for f in files],
            "chunks": summary.get("chunks"), "words": summary.get("words"),
        },
        "versions": {"code": "", "store": model or "",
                     "store_label": _model_meta(model).get("label", model) if model else "לא נרשם",
                     "store_note": "", "fasttext": ""},
        "results": _results(summary, cps),
    }


def load(token):
    """The record of one run, with its meta and the files it produced; None if there is no run."""
    d = _dir(token)
    if not (d / "summary.json").exists():
        return None
    try:
        record = (json.loads((d / RUN_FILE).read_text(encoding="utf-8")) if (d / RUN_FILE).exists()
                  else _from_summary(token))
    except (OSError, ValueError):
        return None
    for cp in record["results"].get("change_point_list") or []:
        for side in ("before", "after"):
            cp[side] = _isolate_ref(cp.get(side))
    record["meta"] = load_meta(token)
    record["outputs"] = [(p.name, p.stat().st_size) for p in sorted(d.iterdir())
                         if not p.name.startswith("_") and p.suffix != ".json"]
    when = datetime.datetime.fromisoformat(record["started"]).astimezone(LOCAL_TZ)
    record["time_label"] = when.strftime("%H:%M")
    record["date_label"] = "%d.%d.%d" % (when.day, when.month, when.year)
    record["day_label"] = _day_label(when)
    record["duration_label"] = _duration(record.get("seconds"))
    record["params"]["grouping_label"] = GROUPING_LABELS.get(record["params"].get("grouping_level"), "")
    return record


WEEKDAYS = ["יום שני", "יום שלישי", "יום רביעי", "יום חמישי", "יום שישי", "שבת", "יום ראשון"]


def _day_label(when):
    """'היום', 'אתמול', or 'יום שלישי · 13.10.2026', in Israel time."""
    today = datetime.datetime.now(LOCAL_TZ).date()
    date = "%d.%d.%d" % (when.day, when.month, when.year)
    if when.date() == today:
        return "היום · " + date
    if (today - when.date()).days == 1:
        return "אתמול · " + date
    return "%s · %s" % (WEEKDAYS[when.weekday()], date)


def _duration(seconds):
    if seconds is None:
        return ""
    seconds = float(seconds)
    if seconds < 60:
        return "%.0f שניות" % seconds if seconds >= 10 else "%.1f שניות" % seconds
    return "%d:%02d דקות" % (seconds // 60, seconds % 60)


_REF = re.compile(r"(\S*\d\S*)$")


def _isolate_ref(place):
    """'אסתר רבה 5_4:67' with the reference kept left to right (see views.LRI); places
    written before that was done are fixed when read."""
    if not isinstance(place, str) or "⁦" in place:
        return place
    return _REF.sub(lambda m: "⁦" + m.group(1) + "⁩", place)


def load_meta(token):
    try:
        return json.loads((_dir(token) / META_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"name": "", "notes": "", "tags": []}


def save_meta(token, name, notes, tags):
    meta = {"name": name.strip()[:200], "notes": notes.strip()[:5000],
            "tags": [t.strip() for t in tags.replace("،", ",").split(",") if t.strip()][:20],
            "edited": _iso(datetime.datetime.now().timestamp())}
    (_dir(token) / META_FILE).write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


def all_runs():
    """Every run on disk, newest first."""
    root = pathlib.Path(settings.RUNS_DIR)
    if not root.exists():
        return []
    out = [r for r in (load(d.name) for d in root.iterdir() if d.is_dir()) if r]
    return sorted(out, key=lambda r: r["started"], reverse=True)


# ---------------------------------------------------------------------------
# Comparing two runs
# ---------------------------------------------------------------------------

PARAM_LABELS = [
    ("store_label", "מרחב האמבדינג"),
    ("chunk_size", "גודל צ'אנק (מילים)"),
    ("k", "מספר קלאסטרים (k)"),
    ("smoothing", "החלקה מול השכנים"),
    ("grouping_label", "תוויות הטקסטים לפי"),
    ("block_size", "גודל בלוק (נקודות שינוי)"),
    ("penalty", "פנלטי (נקודות שינוי)"),
]


def _param(run, key):
    if key == "store_label":
        return run["versions"].get("store_label") or run["params"].get("model")
    value = run["params"].get(key)
    if isinstance(value, bool):
        return "כן" if value else "לא"
    return value


def compare(a, b):
    """What two runs share and where they part: parameters, books, results, and - when
    they clustered the very same chunks - how far their clusterings agree."""
    params = [{"label": label, "a": _param(a, key), "b": _param(b, key),
               "differs": _param(a, key) != _param(b, key)} for key, label in PARAM_LABELS]
    fa, fb = set(a["selection"]["files"]), set(b["selection"]["files"])
    title_a = dict(zip(a["selection"]["files"], a["selection"]["titles"]))
    title_b = dict(zip(b["selection"]["files"], b["selection"]["titles"]))
    results = [
        ("צ'אנקים", a["selection"]["chunks"], b["selection"]["chunks"]),
        ("מילים", a["selection"]["words"], b["selection"]["words"]),
        ("קלאסטרים", a["results"]["clusters"], b["results"]["clusters"]),
        ("נקודות שינוי", a["results"]["change_points"], b["results"]["change_points"]),
        ("לכידות ממוצעת", a["results"]["cohesion_mean"], b["results"]["cohesion_mean"]),
        ("הקלאסטר הגדול (% מהבחירה)", a["results"]["largest_cluster_share"],
         b["results"]["largest_cluster_share"]),
        ("משך (שניות)", a["seconds"], b["seconds"]),
    ]
    return {
        "params": params,
        "n_param_diffs": sum(p["differs"] for p in params),
        "common": sorted(title_a[f] for f in fa & fb),
        "only_a": sorted(title_a[f] for f in fa - fb),
        "only_b": sorted(title_b[f] for f in fb - fa),
        "results": [{"label": l, "a": x, "b": y, "differs": x != y} for l, x, y in results],
        "agreement": _agreement(a, b),
    }


def _agreement(a, b):
    """Adjusted Rand index and the cluster-to-cluster correspondence, when the two runs
    clustered the same chunks (same books, same chunk size, same store)."""
    same = (set(a["selection"]["files"]) == set(b["selection"]["files"])
            and a["params"].get("chunk_size") == b["params"].get("chunk_size")
            and a["params"].get("model") == b["params"].get("model"))
    if not same:
        why = []
        if set(a["selection"]["files"]) != set(b["selection"]["files"]):
            why.append("הספרים שונים")
        if a["params"].get("chunk_size") != b["params"].get("chunk_size"):
            why.append("גודל הצ'אנק שונה")
        if a["params"].get("model") != b["params"].get("model"):
            why.append("מרחב האמבדינג שונה")
        return {"comparable": False, "why": " · ".join(why)}
    try:
        ca = pd.read_csv(_dir(a["token"]) / "chunks.csv", usecols=["chronological_index", "scluster"])
        cb = pd.read_csv(_dir(b["token"]) / "chunks.csv", usecols=["chronological_index", "scluster"])
    except (OSError, ValueError):
        return {"comparable": False, "why": "קובץ הצ'אנקים של אחת הריצות חסר"}
    both = ca.merge(cb, on="chronological_index", suffixes=("_a", "_b"))
    if both.empty:
        return {"comparable": False, "why": "אין צ'אנקים משותפים"}
    from sklearn.metrics import adjusted_rand_score
    ari = float(adjusted_rand_score(both["scluster_a"], both["scluster_b"]))
    table = pd.crosstab(both["scluster_a"], both["scluster_b"])
    mapping = []
    for ca_id, row in table.iterrows():
        best = int(row.idxmax())
        mapping.append({
            "a": int(ca_id), "b": best, "chunks": int(row.sum()),
            "share": round(100 * row.max() / row.sum()),
            # How much of B's cluster this A cluster accounts for: high both ways means
            # the two runs found the same group.
            "back": round(100 * row.max() / table[best].sum()),
        })
    return {"comparable": True, "ari": round(ari, 3), "ari_pct": max(0, min(100, round(ari * 100))),
            "mapping": mapping, "chunks": len(both)}
