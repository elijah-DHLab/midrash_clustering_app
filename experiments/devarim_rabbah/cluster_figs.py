"""The app's cluster figure for the task corpus, with change points found on the chunk vectors
instead of on the block composition of the clusters.

The clustering is the app's own (pipeline_core.cluster_chunks: KMeans on the whole selection,
no smoothing), drawn by cluster_visualization_report.build_figure. The red lines are the
change points of segment_stream.py for the same chunk size and the same selection (with or
without Devarim Rabbah): found on the vectors over the whole chronological stream, each kept
only if it beat every block-shuffled copy (p<0.01). The number above a line is its strength
(times the strongest shuffle); hovering the line shows how the cluster mix changes across it
and where in the text it falls. Devarim Rabbah 11:5, when it is in the selection, is a green
dotted line.

    python experiments/devarim_rabbah/cluster_figs.py --size 50 --k 15 [--without-devarim]
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import plotly.graph_objects as go

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import grid                                   # noqa: E402
import pipeline_core                          # noqa: E402
import cluster_visualization_report as cvr    # noqa: E402

OUT = HERE / "results" / "figs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=50)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--without-devarim", action="store_true")
    args = ap.parse_args()
    tag = "_without_devarim" if args.without_devarim else ""
    stream = json.loads((HERE / "results" / f"stream_changepoints_s{args.size}{tag}.json").read_text(encoding="utf-8"))

    groups, target = grid.corpus_files()
    names = grid.store.titles_for(grid.STORE)
    files = list(groups) + ([] if args.without_devarim else [target])
    df = grid.store.load_selection(files, args.size, grid.STORE)
    assert len(df) == stream["chunks"], "the stream was computed on a different selection"

    df = pipeline_core.cluster_chunks(df, k=args.k, apply_smoothing=False)
    df = cvr.prepare_dataframe(df, grouping_level=3)
    df = cvr.merge_small_groups(df)
    df = cvr.compute_density_original(df)
    df = cvr.compute_chunk_cluster_affinities(df)
    df = cvr.add_cluster_ctfidf_ngrams(df, ngram_n=3, top_k=20)

    # Change points in the figure's format: (chronological index, strength, cluster shifts),
    # the shifts being the change in cluster shares from the segment before to the one after.
    labels = df.sort_values("chronological_index")["scluster"].to_numpy()
    cps = sorted(stream["cps"], key=lambda c: c["i"])
    edges = [0] + [c["i"] for c in cps] + [len(labels)]
    share = lambda a, b: np.bincount(labels[a:b], minlength=args.k) / max(1, b - a)   # noqa: E731
    change_points = []
    for j, c in enumerate(cps, start=1):
        diff = share(edges[j], edges[j + 1]) - share(edges[j - 1], edges[j])
        order = np.argsort(-np.abs(diff))
        change_points.append((c["i"], c["strength"], [(int(i), float(diff[i])) for i in order if abs(diff[i]) > 1e-6]))

    fig = cvr.build_figure(df, change_points=change_points, render_mode="webgl")
    y_top = float(df["scluster"].max()) + 0.4
    fig.add_trace(go.Scatter(
        x=[c["i"] for c in cps], y=[y_top] * len(cps), mode="markers",
        marker=dict(size=9, color="rgba(200,0,0,0.0)"), showlegend=False,
        hovertemplate=[f"<b>{names.get(c['file'], {}).get('he', c['file'])}</b> · {c['unit']}<br>"
                       f"strength {c['strength']}{' · work boundary' if c['work_boundary'] else ''}<br>"
                       f"before: …{c['before']}<br>after: {c['after']}…<extra></extra>" for c in cps]))
    if not args.without_devarim:
        start = next(w["start"] for w in stream["works"] if w["file"] == target)
        b115 = start + grid.boundary_word(target) // args.size
        fig.add_vline(x=b115, line_width=2, line_dash="dot", line_color="green", opacity=0.9)
        fig.add_annotation(x=b115, y=y_top + 0.6, text="דברים רבה י״א, ה׳", showarrow=False,
                           font=dict(size=11, color="green"))

    found = sum(w["found_start"] for w in stream["works"][1:])
    on = sum(c["work_boundary"] for c in cps)
    fig.update_layout(title=f"קורפוס המשימה · {args.size} מילים · k={args.k} · "
                            f"{'בלי' if args.without_devarim else 'עם'} דברים רבה · {len(cps)} נקודות שינוי מהווקטורים "
                            f"({on} על גבול חיבור; נמצאו {found} מתוך {len(stream['works']) - 1} גבולות בין חיבורים)",
                      dragmode="zoom", height=760)
    fig.update_yaxes(fixedrange=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"clusters_s{args.size}_k{args.k}{tag}.html"
    fig.write_html(path, include_plotlyjs="cdn", config={"scrollZoom": True, "displaylogo": False})
    print(path, f"| {len(cps)} change points, {on} on work boundaries, {found}/{len(stream['works']) - 1} boundaries found")


if __name__ == "__main__":
    main()
