"""Redraws the change points on the cluster figures made by cluster_figs.py, without recomputing
the clustering: over the whole corpus a thousand full-height dashed red lines with a score above
each hid the clusters. They become thin lines whose weight follows the strength, in three
groups that buttons show or hide (all / strength >= 2 / >= 4), with a hover mark at the top of
each line giving the work, the unit and the text on either side.

    python experiments/devarim_rabbah/restyle_figs.py
"""
import json
import pathlib
import re
import sys

import plotly.graph_objects as go

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grid   # noqa: E402

FIGS = HERE / "results" / "figs"
BINS = [(1.0, 2.0, 0.8, "rgba(200,30,30,0.35)"), (2.0, 4.0, 1.6, "rgba(200,30,30,0.7)"), (4.0, 1e9, 2.8, "rgba(170,0,0,0.95)")]


def read_figure(html):
    """The data and layout of the Plotly.newPlot call that write_html emits."""
    i = html.index("Plotly.newPlot(")
    j = html.index("[", i)
    dec = json.JSONDecoder()
    data, end = dec.raw_decode(html, j)
    k = html.index("{", end)
    layout, _ = dec.raw_decode(html, k)
    return data, layout


def main():
    names = grid.store.titles_for(grid.STORE)
    for path in sorted(FIGS.glob("clusters_s*_k*.html")):
        if path.stem.endswith("_v2"):
            continue
        m = re.match(r"clusters_s(\d+)_k(\d+)(_without_devarim)?", path.stem)
        size, tag = int(m.group(1)), m.group(3) or ""
        stream = json.loads((HERE / "results" / f"stream_changepoints_s{size}{tag}.json").read_text(encoding="utf-8"))
        data, layout = read_figure(path.read_text(encoding="utf-8"))

        # Drop the old change-point layer: red shapes, red annotations, the hover traces on the lines.
        layout["shapes"] = [s for s in layout.get("shapes", []) if s.get("line", {}).get("color") != "red"]
        layout["annotations"] = [a for a in layout.get("annotations", []) if a.get("arrowcolor") != "red"]
        data = [t for t in data if not (str(t.get("hovertemplate", "")).startswith("<b>Change point")
                                        or (t.get("marker", {}).get("color") in ("rgba(0,0,0,0)", "rgba(200,0,0,0.0)")
                                            and t.get("mode") == "markers" and len(t.get("x", [])) <= 2000
                                            and "Cluster" not in str(t.get("name", ""))
                                            and t.get("xaxis", "x") == "x" and not t.get("legendgroup")))]
        fig = go.Figure(data=data, layout=layout)
        # Cluster ids run 0..k-1 on the y axis (the point arrays are stored binary-encoded).
        k = int(m.group(2))
        y_lo, y_hi = -0.6, k - 0.4

        groups = []
        for lo, hi, width, colour in BINS:
            cps = [c for c in stream["cps"] if lo <= c["strength"] < hi]
            xs, ys = [], []
            for c in cps:
                xs += [c["i"], c["i"], None]
                ys += [y_lo, y_hi, None]
            first = len(fig.data)
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=width, color=colour),
                                     hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(
                x=[c["i"] for c in cps], y=[y_hi] * len(cps), mode="markers",
                marker=dict(size=7, color=colour, symbol="triangle-down"), showlegend=False,
                hovertemplate=[f"<b>{names.get(c['file'], {}).get('he', c['file'])}</b> · {c['unit']}<br>"
                               f"עוצמה {c['strength']}{' · גבול בין חיבורים' if c['work_boundary'] else ''}<br>"
                               f"לפני: …{c['before']}<br>אחרי: {c['after']}…<extra></extra>" for c in cps]))
            groups.append((first, first + 1, len(cps)))

        # Change-point traces first, so the clusters are drawn over them.
        cp_idx = [i for g in groups for i in g[:2]]
        fig.data = tuple(fig.data[i] for i in cp_idx) + tuple(t for i, t in enumerate(fig.data) if i not in cp_idx)
        groups = [(2 * b, 2 * b + 1, g[2]) for b, g in enumerate(groups)]
        n = len(fig.data)
        def visible(min_bin):
            v = [True] * n
            for b, (a, h, _) in enumerate(groups):
                v[a] = v[h] = b >= min_bin
            return v
        counts = [sum(g[2] for g in groups[b:]) for b in range(3)]
        fig.update_traces(selector=dict(), patch=None)
        for i, v in enumerate(visible(2)):   # open on the strongest points only
            fig.data[i].visible = v
        fig.update_layout(title=dict(y=0.985), updatemenus=[dict(
            type="buttons", direction="left", x=0.0, y=1.0, xanchor="left", yanchor="bottom", showactive=True, active=2,
            buttons=[dict(label=f"כל הנקודות ({counts[0]})", method="restyle", args=[{"visible": visible(0)}]),
                     dict(label=f"עוצמה ≥ 2 ({counts[1]})", method="restyle", args=[{"visible": visible(1)}]),
                     dict(label=f"עוצמה ≥ 4 ({counts[2]})", method="restyle", args=[{"visible": visible(2)}])])],
            margin=dict(t=150))
        out = path.with_name(path.stem + "_v2.html")
        fig.write_html(out, include_plotlyjs="cdn", config={"scrollZoom": True, "displaylogo": False})
        print(out.name, "| points by strength band:", [g[2] for g in groups])


if __name__ == "__main__":
    main()
