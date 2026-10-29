"""Devarim Rabbah along its length: how much each stretch leans towards each reference genre.

Clusters are fitted on the reference corpus at equal size per genre (Devarim Rabbah left out),
and every chunk of Devarim Rabbah is assigned to its nearest cluster. A cluster's lean towards
a genre is that genre's share among the cluster's reference chunks, divided by its share in
the fitted sample (1/6 when balanced): 1 is neutral, 2 is twice as typical of that genre as
of the corpus. Each chunk takes its cluster's leans; they are averaged over five seeds and
then over a moving window, and drawn against the position in the work, with the parashot
marked and 10:1 and 11:5 drawn as lines.

    python experiments/devarim_rabbah/profile.py
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grid   # noqa: E402

SIZE, K, SEEDS, WINDOW = 25, 20, [1, 2, 3, 4, 5], 20   # window: 20 chunks = 500 words
GENRE_HE = {"classic": "מדרש אגדה קלאסי", "late": "מדרש אגדה מאוחר", "tannaitic": "מדרשי תנאים",
            "mishnah_tosefta": "משנה ותוספתא", "bavli": "תלמוד בבלי", "yerushalmi": "תלמוד ירושלמי"}
COLOURS = {"classic": "#3b6ea8", "late": "#8c2f28", "tannaitic": "#5a8f3c",
           "mishnah_tosefta": "#b08a2e", "bavli": "#6b4c9a", "yerushalmi": "#2f8f8a"}


def main():
    groups, target = grid.corpus_files()
    df = grid.store.load_selection(list(groups) + [target], SIZE, grid.STORE)
    files = df["filename"].to_numpy()
    vecs = np.vstack(df["vec"].to_numpy()).astype(np.float32)
    genre = np.array([groups.get(f, "target") for f in files])
    is_target = files == target
    names = list(grid.GROUPS)

    lean = np.zeros((is_target.sum(), len(names)))
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        m = min(int((genre == g).sum()) for g in names)
        fit_idx = np.concatenate([rng.choice(np.flatnonzero(genre == g), m, replace=False) for g in names])
        km = KMeans(n_clusters=K, random_state=seed, n_init=3).fit(vecs[fit_idx])
        fit_labels = km.labels_
        table = np.zeros((K, len(names)))
        for gi, g in enumerate(names):
            table[:, gi] = np.bincount(fit_labels[genre[fit_idx] == g], minlength=K)
        share = table / np.maximum(table.sum(axis=1, keepdims=True), 1)
        lift = share / (1 / len(names))
        lean += lift[km.predict(vecs[is_target])]
        print("seed", seed, "done", flush=True)
    lean /= len(SEEDS)

    words = np.arange(len(lean)) * SIZE
    smooth = pd.DataFrame(lean, columns=names).rolling(WINDOW, center=True, min_periods=WINDOW // 2).mean()

    # Parasha starts, for the axis.
    prefix = target[: -len(".jsonl")].replace("-", "_") + "_"
    starts, w, seen = [], 0, set()
    marks = {}
    for line in open(grid.ROOT / "corpus_v7" / target, encoding="utf-8"):
        r = json.loads(line)
        unit = r["location"][len(prefix):]
        parasha = unit.split("_")[0]
        if parasha not in seen:
            seen.add(parasha)
            starts.append((w, parasha))
        if unit in ("10_1", "11_5") and unit not in marks:
            marks[unit] = w
        w += len(r.get("sentence", "").split())

    fig = go.Figure()
    for g in names:
        fig.add_trace(go.Scatter(
            x=words, y=smooth[g], name=GENRE_HE[g], mode="lines",
            line=dict(color=COLOURS[g], width=3.5 if g == "late" else 1.8),
            hovertemplate="מילה %{x}<br>" + GENRE_HE[g] + ": %{y:.2f}<extra></extra>"))
    for w0, p in starts:
        fig.add_vline(x=w0, line=dict(color="rgba(0,0,0,0.12)", width=1))
    # The two lines are ~1,000 words apart: one label to the left of its line, one to the right.
    for unit, label, anchor in (("10_1", "תחילת פרשה י׳ ", "right"), ("11_5", " י״א, ה׳", "left")):
        fig.add_vline(x=marks[unit], line=dict(color="#8c2f28", width=1.5, dash="dash"))
        fig.add_annotation(x=marks[unit], y=1.03, yref="paper", text=label, showarrow=False,
                           font=dict(color="#8c2f28", size=12), xanchor=anchor)
    fig.add_hline(y=1, line=dict(color="rgba(0,0,0,0.35)", width=1, dash="dot"))
    fig.update_layout(
        title=dict(text="דברים רבה לאורך החיבור: הנטייה לכל סוגה", x=0.5),
        xaxis=dict(title="פרשה", tickvals=[s for s, _ in starts], ticktext=[p for _, p in starts]),
        yaxis=dict(title="נטייה (1 = ניטרלי)"),
        template="simple_white", height=560, width=1200, legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
        margin=dict(t=80, b=120), font=dict(family="Heebo, Arial", size=13),
    )
    out = HERE / "results"
    fig.write_html(out / "devarim_profile.html", include_plotlyjs="cdn")
    pd.DataFrame({"word": words, **{g: lean[:, i] for i, g in enumerate(names)}}).to_csv(
        out / "devarim_profile.csv", index=False, encoding="utf-8-sig")
    before, after = lean[: marks["10_1"] // SIZE], lean[marks["11_5"] // SIZE:]
    middle = lean[marks["10_1"] // SIZE: marks["11_5"] // SIZE]
    print("mean lean   before 10_1 | 10_1–11_5 | after 11_5")
    for i, g in enumerate(names):
        print(f"{GENRE_HE[g]:18} {before[:, i].mean():5.2f}   {middle[:, i].mean():5.2f}   {after[:, i].mean():5.2f}")


if __name__ == "__main__":
    main()
