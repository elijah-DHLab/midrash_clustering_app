"""Cluster bubble map: every chunk as a point inside its cluster's bubble.

Clusters are laid out as circles (MDS on cosine similarity between cluster
centroids — closer circles = more similar clusters, circle size = number of
chunks). Inside each circle, individual chunks are scattered and colored by
the corpus/book they come from, with a shared legend.
"""

import argparse
import os

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from clearml import Task
from sklearn.metrics.pairwise import cosine_similarity

from cluster_visualization_report import (
    merge_small_groups,
    prepare_dataframe,
    resolve_input_csv,
)
from cluster_map_visualization import (
    compute_cluster_centroids,
    compute_cluster_map_positions,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "input_csv": "Results/rabani_unsmoothed_table.csv",
    "output_html": "Results/cluster_bubble_map.html",
    "task_name": "cluster_bubble_map",
    "color_grouping_level": 2,
    "spread": 8.0,
    "radius_scale": 0.7,
    "sample_frac": 1.0,
    "seed": 42,
}

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def resolve_overlaps(centers, radii, padding=0.15, iterations=500):
    """Push circle centers apart until no two bubbles overlap, while
    preserving their relative arrangement as much as possible."""
    centers = centers.copy()
    n = len(centers)
    for _ in range(iterations):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                min_dist = radii[i] + radii[j] + padding
                if dist < min_dist:
                    moved = True
                    if dist < 1e-9:
                        rng = np.random.default_rng(i * 1000 + j)
                        direction = rng.normal(size=2)
                        direction /= np.linalg.norm(direction)
                    else:
                        direction = diff / dist
                    overlap = min_dist - dist
                    centers[i] += direction * overlap / 2
                    centers[j] -= direction * overlap / 2
        if not moved:
            break
    return centers


def compute_cluster_layout(positions, sizes, spread, radius_scale):
    """Center & scale MDS positions, derive per-cluster bubble radii, and
    separate bubbles so they don't overlap."""
    positions = positions - positions.mean(axis=0)
    extent = np.abs(positions).max()
    if extent > 0:
        positions = positions / extent * spread

    radii = radius_scale * np.sqrt(sizes / sizes.max()) * (spread / 2.0)
    positions = resolve_overlaps(positions, radii)
    return positions, radii


def jitter_points_in_circles(df, cluster_ids, centers, radii, seed):
    """Assign each chunk a random point inside its cluster's circle
    (uniform over the disk area)."""
    rng = np.random.default_rng(seed)
    cluster_index = {cid: i for i, cid in enumerate(cluster_ids)}

    cluster_idx_arr = df["scluster"].map(cluster_index).to_numpy()
    n = len(df)

    theta = rng.uniform(0, 2 * np.pi, n)
    rho = np.sqrt(rng.uniform(0, 1, n)) * radii[cluster_idx_arr]

    cx = centers[cluster_idx_arr, 0]
    cy = centers[cluster_idx_arr, 1]

    x = cx + rho * np.cos(theta)
    y = cy + rho * np.sin(theta)
    return x, y


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def build_figure(df, cluster_ids, centers, radii, sizes, color_col):
    fig = go.Figure()

    color_groups = sorted(df[color_col].unique())
    palette = px.colors.qualitative.Dark24
    color_map = {grp: palette[i % len(palette)] for i, grp in enumerate(color_groups)}

    for grp in color_groups:
        sub = df[df[color_col] == grp]
        fig.add_trace(
            go.Scattergl(
                x=sub["_x"].round(3),
                y=sub["_y"].round(3),
                mode="markers",
                name=str(grp),
                marker=dict(size=3, color=color_map[grp], opacity=0.55),
                customdata=sub[["collection", "scluster"]].to_numpy(),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Cluster: %{customdata[1]}<extra></extra>"
                ),
            )
        )

    # --- Cluster bubble outlines & labels ---
    for i, cid in enumerate(cluster_ids):
        cx, cy = centers[i]
        r = radii[i]
        fig.add_shape(
            type="circle",
            x0=cx - r,
            y0=cy - r,
            x1=cx + r,
            y1=cy + r,
            line=dict(color="rgba(40,40,40,0.55)", width=2, dash="dot"),
            fillcolor="rgba(0,0,0,0)",
            layer="below",
        )
        fig.add_annotation(
            x=cx,
            y=cy + r,
            text=f"<b>Cluster {cid}</b> ({sizes[i]} chunks)",
            showarrow=False,
            yanchor="bottom",
            font=dict(size=13, color="rgba(20,20,20,0.85)"),
        )

    fig.update_layout(
        width=1500,
        height=1100,
        title="Cluster Bubble Map — proximity, size & textual composition",
        legend_title_text=color_col,
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=""),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            title="",
            scaleanchor="x",
            scaleratio=1,
        ),
        plot_bgcolor="white",
    )
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Cluster bubble map visualization.")
    parser.add_argument("--input-csv", default=DEFAULT_CONFIG["input_csv"])
    parser.add_argument("--output-html", default=DEFAULT_CONFIG["output_html"])
    parser.add_argument("--task-name", default=DEFAULT_CONFIG["task_name"])
    parser.add_argument(
        "--color-grouping-level",
        type=int,
        choices=[1, 2, 3],
        default=DEFAULT_CONFIG["color_grouping_level"],
        help="Hierarchy level used to color points by source text (1=Corpus, 2=Sub-corpus, 3=Book).",
    )
    parser.add_argument("--spread", type=float, default=DEFAULT_CONFIG["spread"])
    parser.add_argument("--radius-scale", type=float, default=DEFAULT_CONFIG["radius_scale"])
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=DEFAULT_CONFIG["sample_frac"],
        help="Fraction of chunks to plot (for faster previews).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    args = parser.parse_args()

    task = Task.init(project_name="MIDRASH", task_name=args.task_name)
    task.connect(vars(args))
    print("ClearML task initialized")

    script_path = os.path.abspath(__file__)
    task.upload_artifact(name="Source Code", artifact_object=script_path)

    resolved_csv = resolve_input_csv(args.input_csv)
    print(f"Using input CSV: {resolved_csv}")

    df = pd.read_csv(resolved_csv)
    df = prepare_dataframe(df, grouping_level=args.color_grouping_level)
    df = merge_small_groups(df)
    print(f"Loaded {len(df)} chunks, {df['scluster'].nunique()} clusters")

    if args.sample_frac < 1.0:
        df = df.sample(frac=args.sample_frac, random_state=args.seed).reset_index(drop=True)
        print(f"Sampled down to {len(df)} chunks")

    # --- Cluster centroids, similarity, layout ---
    cluster_ids, centroids, sizes = compute_cluster_centroids(df)
    similarity_matrix = cosine_similarity(centroids)
    raw_positions = compute_cluster_map_positions(similarity_matrix)
    centers, radii = compute_cluster_layout(raw_positions, sizes, args.spread, args.radius_scale)
    print("Computed cluster layout (centers & radii)")

    # --- Jitter individual chunks within their cluster's circle ---
    df["_x"], df["_y"] = jitter_points_in_circles(df, cluster_ids, centers, radii, args.seed)

    # --- Build figure ---
    fig = build_figure(df, cluster_ids, centers, radii, sizes, color_col="grouping_label")

    os.makedirs(os.path.dirname(args.output_html) or ".", exist_ok=True)
    fig.write_html(args.output_html, auto_open=False)
    print(f"Saved visualization to: {args.output_html}")

    task.upload_artifact(name="Visualization HTML", artifact_object=args.output_html)
    task.get_logger().flush()
    task.flush(wait_for_uploads=True)
    task.close()
    print("ClearML task closed")


if __name__ == "__main__":
    main()
