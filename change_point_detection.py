"""Change Point Detection on cluster distributions using PELT.

Reads the clustered CSV, converts the sequence of cluster assignments into
a sliding-window distribution signal, and runs PELT to find optimal change
points. Outputs an HTML with the cluster scatter on top and detected change
points marked as vertical lines.
"""

import argparse
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.distance import jensenshannon
import ruptures as rpt

# ── reuse helpers from local_density_visualization ──────────────────────
from local_density_visualization import (
    _parse_hierarchy_list,
    _grouping_label,
    merge_small_groups,
    resolve_input_csv,
)


# ── data preparation ────────────────────────────────────────────────────


def prepare_dataframe(df):
    if "chronological_index" not in df.columns:
        if "hierarchy_list" in df.columns:
            df = df.copy()
            df["_ht"] = df["hierarchy_list"].apply(_parse_hierarchy_list)
            df = df.sort_values(
                by=["_ht", "filename", "chunk_number"], kind="mergesort"
            ).reset_index(drop=True)
            df["chronological_index"] = np.arange(len(df), dtype=int)
            df.drop(columns=["_ht"], inplace=True)
        else:
            df = df.copy().reset_index(drop=True)
            df["chronological_index"] = np.arange(len(df), dtype=int)

    df["scluster"] = pd.to_numeric(df["scluster"], errors="coerce")
    df = df.dropna(subset=["scluster"]).copy()
    df["scluster"] = df["scluster"].astype(int)
    return df


# ── block-level signals ──────────────────────────────────────────────


def build_block_signals(cluster_seq, n_clusters, block_size):
    """Build block-level distribution matrix AND JSD between adjacent blocks.

    Returns (distributions, jsd_values, block_edges) where:
    - distributions[i] is the cluster-proportion vector for block i
    - jsd_values[i] is JSD between block i and block i+1 (length = n_blocks-1)
    - block_edges[i] is the starting chunk index of block i
    """
    n = len(cluster_seq)
    n_blocks = n // block_size
    distributions = np.zeros((n_blocks, n_clusters), dtype=float)
    block_edges = np.zeros(n_blocks, dtype=int)

    for b in range(n_blocks):
        lo = b * block_size
        hi = lo + block_size
        counts = np.bincount(cluster_seq[lo:hi], minlength=n_clusters).astype(float)
        distributions[b] = counts / counts.sum()
        block_edges[b] = lo

    # JSD between consecutive blocks
    jsd_values = np.zeros(n_blocks - 1)
    for b in range(n_blocks - 1):
        jsd_values[b] = jensenshannon(distributions[b], distributions[b + 1])

    return distributions, jsd_values, block_edges


# ── PELT on multi-dimensional distribution signal ──────────────────────


def detect_change_points(distributions, pen=None):
    """Run PELT on the block-level distribution matrix.

    Finds points where the cluster composition changes significantly.
    pen (penalty): controls sensitivity. Lower = more change points.
    """
    n = len(distributions)
    if pen is None:
        pen = 0.1

    model = rpt.Pelt(model="l2", min_size=2, jump=1).fit(distributions)
    change_points = model.predict(pen=pen)

    # ruptures returns last index as "endpoint", remove it.
    if change_points and change_points[-1] == n:
        change_points = change_points[:-1]

    return change_points


# ── visualization ───────────────────────────────────────────────────────


def build_figure(df, jsd_x, jsd_y, cp_positions, grouping_level):
    n_clusters = int(df["scluster"].max()) + 1
    pad = len(str(n_clusters - 1))
    df = df.copy()
    df["scluster_str"] = "Cluster " + df["scluster"].astype(str).str.zfill(pad)

    # grouping
    df["grouping_label"] = df["filename"].apply(
        lambda f: _grouping_label(f, grouping_level)
    )
    df = merge_small_groups(df)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.06,
        subplot_titles=[
            "Cluster Scatter Over Time",
            "JSD Between Adjacent Blocks (PELT change points in red)",
        ],
    )

    # ─── top: scatter ───
    colors = df["scluster_str"].unique()
    color_map = {
        c: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
        for i, c in enumerate(sorted(colors))
    }

    for cluster_name in sorted(colors):
        mask = df["scluster_str"] == cluster_name
        sub = df[mask]
        fig.add_trace(
            go.Scattergl(
                x=sub["chronological_index"],
                y=sub["scluster"],
                mode="markers",
                marker=dict(size=4, opacity=0.7, color=color_map[cluster_name]),
                name=cluster_name,
                legendgroup=cluster_name,
                hovertemplate=(
                    "<b>Index:</b> %{x}<br><b>Cluster:</b> %{y}<br><extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    # ─── bottom: JSD trace ───
    fig.add_trace(
        go.Scattergl(
            x=jsd_x,
            y=jsd_y,
            mode="lines",
            line=dict(color="steelblue", width=1.5),
            name="JSD",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # ─── change-point lines on both panels ───
    for cp in cp_positions:
        for row_idx in [1, 2]:
            fig.add_vline(
                x=int(cp),
                line_width=2,
                line_dash="dash",
                line_color="red",
                opacity=0.7,
                row=row_idx,
                col=1,
            )

    # ─── group boundary lines + labels on scatter ───
    group_bounds = (
        df.groupby("grouping_label")["chronological_index"]
        .agg(["min", "max"])
        .sort_values("min")
        .reset_index()
    )
    total_span = float(group_bounds["max"].max() - group_bounds["min"].min())
    for i, row in group_bounds.iterrows():
        for row_idx in [1, 2]:
            fig.add_vline(
                x=float(row["min"]),
                line_width=1,
                line_color="rgba(60,60,60,0.3)",
                row=row_idx,
                col=1,
            )
        if i % 2 == 0:
            fig.add_vrect(
                x0=float(row["min"]),
                x1=float(row["max"]),
                fillcolor="rgba(120,120,120,0.05)",
                line_width=0,
                layer="below",
                row=1,
                col=1,
            )

    # X-axis labels
    tick_vals = []
    tick_text = []
    for _, row in group_bounds.iterrows():
        span = float(row["max"]) - float(row["min"])
        if span < total_span * 0.025:
            continue
        label = str(row["grouping_label"])
        if len(label) > 30:
            label = label[:27] + "..."
        tick_vals.append(float(row["min"]))
        tick_text.append(label)

    fig.update_xaxes(
        tickvals=tick_vals,
        ticktext=tick_text,
        tickangle=-40,
        tickfont=dict(size=11),
        row=2,
        col=1,
    )

    fig.update_yaxes(title_text="Cluster ID", row=1, col=1)
    fig.update_yaxes(title_text="JSD", row=2, col=1)

    fig.update_layout(
        width=1400,
        height=850,
        margin=dict(l=70, r=160, t=60, b=160),
        legend_title_text="Cluster",
    )

    return fig


# ── main ────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Change Point Detection on cluster distributions (PELT)."
    )
    parser.add_argument(
        "--input-csv",
        default="Results/rabani_unsmoothed_table.csv",
    )
    parser.add_argument(
        "--output-html",
        default="change_point_detection.html",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=500,
        help="Non-overlapping block size for aggregating distribution signal.",
    )
    parser.add_argument(
        "--penalty",
        type=float,
        default=None,
        help="PELT penalty. Lower = more change points. Default: auto.",
    )
    parser.add_argument(
        "--grouping-level",
        type=int,
        choices=[1, 2, 3],
        default=2,
    )
    args = parser.parse_args()

    csv_path = resolve_input_csv(args.input_csv)
    print(f"Using input CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    df = prepare_dataframe(df)
    df = df.sort_values("chronological_index").reset_index(drop=True)

    cluster_seq = df["scluster"].to_numpy()
    n_clusters = int(cluster_seq.max()) + 1

    print(f"Building block signals (block_size={args.window})...")
    distributions, jsd_values, block_edges = build_block_signals(
        cluster_seq, n_clusters, args.window
    )
    print(f"Aggregated into {len(distributions)} blocks.")

    print("Running PELT on distribution signal...")
    cp_block_indices = detect_change_points(distributions, pen=args.penalty)
    print(f"Found {len(cp_block_indices)} change points.")

    # Map block indices back to chunk positions.
    cp_positions = [int(block_edges[i]) for i in cp_block_indices]

    # JSD x-positions: boundary between block b and b+1
    jsd_x = block_edges[1:]  # position of each boundary

    print("Building visualization...")
    fig = build_figure(df, jsd_x, jsd_values, cp_positions, args.grouping_level)

    os.makedirs(os.path.dirname(args.output_html) or ".", exist_ok=True)
    fig.write_html(args.output_html, auto_open=False)
    print(f"Saved to: {args.output_html}")
    print(f"\nChange points at chronological indices: {cp_positions}")


if __name__ == "__main__":
    main()
