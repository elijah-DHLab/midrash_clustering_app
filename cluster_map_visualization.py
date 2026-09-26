"""Cluster map visualization: similarity between clusters and their textual
composition.

Produces an HTML report with:
- A 2D map of clusters (MDS on cosine distance between cluster centroids),
  bubble size = cluster size, hover shows top c-TF-IDF n-grams and nearest
  neighbor clusters.
- A heatmap of pairwise cosine similarity between cluster centroids.
- A stacked bar chart showing each cluster's composition by corpus
  (grouping_label).

Also writes CSVs with the similarity matrix and a per-cluster breakdown of
the top contributing texts.
"""

import argparse
import os

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from clearml import Task
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity

from cluster_visualization_report import (
    add_cluster_ctfidf_ngrams,
    merge_small_groups,
    prepare_dataframe,
    resolve_input_csv,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "input_csv": "Results/rabani_unsmoothed_table.csv",
    "output_html": "Results/cluster_map_visualization.html",
    "task_name": "cluster_map_visualization",
    "grouping_level": 2,
    "top_texts": 10,
}

# ---------------------------------------------------------------------------
# Vector parsing & cluster summaries
# ---------------------------------------------------------------------------


def _parse_vec_fast(vec_str):
    """Parse a '[0.1, 0.2, ...]' string to a float array without ast.literal_eval."""
    cleaned = str(vec_str).strip().strip("[]")
    if not cleaned:
        return None
    return np.fromstring(cleaned, sep=",")


def compute_cluster_centroids(df):
    """Return (cluster_ids, centroid_matrix, cluster_sizes)."""
    if "vec" not in df.columns:
        raise ValueError("Input CSV is missing the 'vec' column required for centroids.")

    vecs = df["vec"].apply(_parse_vec_fast)
    valid = vecs.apply(lambda v: v is not None and v.size > 0)
    df_valid = df[valid].copy()
    df_valid["_vec"] = vecs[valid]

    cluster_ids = sorted(df_valid["scluster"].unique())
    centroids = []
    sizes = []
    for cid in cluster_ids:
        cluster_vecs = np.stack(df_valid.loc[df_valid["scluster"] == cid, "_vec"].values)
        centroids.append(cluster_vecs.mean(axis=0))
        sizes.append(int(len(cluster_vecs)))

    return cluster_ids, np.stack(centroids), np.array(sizes)


def compute_cluster_map_positions(similarity_matrix, random_state=42):
    """2D MDS embedding from a cosine similarity matrix (1 - sim = distance)."""
    distance_matrix = 1.0 - similarity_matrix
    np.fill_diagonal(distance_matrix, 0.0)
    distance_matrix = np.clip(distance_matrix, 0.0, None)
    # Symmetrize to guard against floating-point asymmetry
    distance_matrix = (distance_matrix + distance_matrix.T) / 2.0

    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=random_state,
        normalized_stress="auto",
        n_init=4,
    )
    return mds.fit_transform(distance_matrix)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def compute_cluster_composition(df, group_col="grouping_label"):
    """For each cluster, the % of its chunks coming from each group."""
    counts = df.groupby(["scluster", group_col]).size().rename("count").reset_index()
    cluster_totals = counts.groupby("scluster")["count"].transform("sum")
    counts["pct_of_cluster"] = 100.0 * counts["count"] / cluster_totals
    return counts


def compute_top_texts_per_cluster(df, top_n=10):
    """For each cluster, the top-N collections by share of the cluster, plus
    the share of each collection's own chunks that fall in that cluster."""
    counts = df.groupby(["scluster", "collection"]).size().rename("count").reset_index()
    cluster_totals = counts.groupby("scluster")["count"].transform("sum")
    collection_totals = df.groupby("collection").size()

    counts["pct_of_cluster"] = 100.0 * counts["count"] / cluster_totals
    counts["pct_of_collection"] = 100.0 * counts["count"] / counts["collection"].map(collection_totals)

    rows = []
    for cid, group in counts.groupby("scluster"):
        top = group.sort_values("pct_of_cluster", ascending=False).head(top_n)
        for _, row in top.iterrows():
            rows.append(
                {
                    "Cluster": int(cid),
                    "Collection": row["collection"],
                    "Chunks": int(row["count"]),
                    "% of Cluster": round(row["pct_of_cluster"], 2),
                    "% of Text": round(row["pct_of_collection"], 2),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def _cluster_color_map(cluster_ids, pad):
    if len(cluster_ids) == 1:
        positions = [0.5]
    else:
        positions = [cid / max(cluster_ids) for cid in cluster_ids]
    sampled_colors = px.colors.sample_colorscale(px.colors.sequential.Plasma, positions)
    return {
        f"Cluster {str(cid).zfill(pad)}": color
        for cid, color in zip(cluster_ids, sampled_colors)
    }


def build_figure(
    cluster_ids,
    similarity_matrix,
    positions,
    sizes,
    ngrams_by_cluster,
    composition,
    group_col,
):
    pad = len(str(max(cluster_ids)))
    labels = [f"Cluster {str(cid).zfill(pad)}" for cid in cluster_ids]
    color_map = _cluster_color_map(cluster_ids, pad)

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "scatter"}, {"type": "heatmap"}], [{"type": "bar", "colspan": 2}, None]],
        column_widths=[0.55, 0.45],
        row_heights=[0.55, 0.45],
        subplot_titles=(
            "Cluster Map (closer = more similar)",
            "Cluster Similarity (cosine)",
            f"Cluster Composition by {group_col}",
        ),
        vertical_spacing=0.14,
        horizontal_spacing=0.12,
    )

    # --- Cluster map ---
    max_size = sizes.max()
    marker_sizes = 18.0 + 40.0 * np.sqrt(sizes / max_size)

    for i, cid in enumerate(cluster_ids):
        sims = similarity_matrix[i].copy()
        sims[i] = -1.0
        nn_order = np.argsort(sims)[::-1][:3]
        nn_text = "<br>".join(
            f"  {labels[j]}: sim={similarity_matrix[i, j]:.3f}" for j in nn_order
        )
        ngram_text = ngrams_by_cluster.get(cid, "No n-grams available")
        top_ngrams = "<br>".join(str(ngram_text).split("<br>")[:8])

        fig.add_trace(
            go.Scatter(
                x=[positions[i, 0]],
                y=[positions[i, 1]],
                mode="markers+text",
                text=[str(cid)],
                textposition="middle center",
                textfont=dict(color="white", size=12, family="Arial Black"),
                marker=dict(
                    size=marker_sizes[i],
                    color=color_map[labels[i]],
                    line=dict(width=1, color="rgba(0,0,0,0.4)"),
                ),
                hovertemplate=(
                    f"<b>{labels[i]}</b><br>"
                    f"Size: {sizes[i]} chunks<br>"
                    f"<b>Nearest clusters:</b><br>{nn_text}<br>"
                    f"<b>Top 3-grams:</b><br>{top_ngrams}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # --- Similarity heatmap ---
    off_diagonal = similarity_matrix[~np.eye(len(cluster_ids), dtype=bool)]
    fig.add_trace(
        go.Heatmap(
            z=similarity_matrix,
            x=labels,
            y=labels,
            colorscale="Viridis",
            zmin=float(off_diagonal.min()),
            zmax=1,
            text=np.round(similarity_matrix, 2),
            texttemplate="%{text}",
            textfont=dict(size=9),
            colorbar=dict(title="cosine sim", x=1.0, len=0.45, y=0.78),
        ),
        row=1,
        col=2,
    )

    # --- Composition stacked bar ---
    groups = sorted(composition[group_col].unique())
    for grp in groups:
        sub = composition[composition[group_col] == grp].set_index("scluster")
        y_vals = [sub["pct_of_cluster"].get(cid, 0.0) for cid in cluster_ids]
        fig.add_trace(
            go.Bar(
                x=labels,
                y=y_vals,
                name=str(grp),
                hovertemplate=f"<b>{grp}</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>",
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        barmode="stack",
        width=1400,
        height=1100,
        title="Cluster Map: Similarity & Composition",
        legend_title_text=group_col,
        legend=dict(orientation="h", y=-0.08),
    )
    fig.update_xaxes(title_text="MDS dim 1", row=1, col=1, showticklabels=False)
    fig.update_yaxes(title_text="MDS dim 2", row=1, col=1, showticklabels=False)
    fig.update_xaxes(tickangle=-40, row=1, col=2)
    fig.update_xaxes(title_text="Cluster", row=2, col=1)
    fig.update_yaxes(title_text="% of cluster", row=2, col=1)

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Cluster map visualization (similarity + composition)."
    )
    parser.add_argument("--input-csv", default=DEFAULT_CONFIG["input_csv"])
    parser.add_argument("--output-html", default=DEFAULT_CONFIG["output_html"])
    parser.add_argument("--task-name", default=DEFAULT_CONFIG["task_name"])
    parser.add_argument(
        "--grouping-level",
        type=int,
        choices=[1, 2, 3],
        default=DEFAULT_CONFIG["grouping_level"],
        help="Hierarchy level for composition grouping (1=Corpus, 2=Sub-corpus, 3=Book).",
    )
    parser.add_argument(
        "--top-texts",
        type=int,
        default=DEFAULT_CONFIG["top_texts"],
        help="Number of top texts per cluster to include in the breakdown CSV.",
    )
    args = parser.parse_args()

    task = Task.init(project_name="MIDRASH", task_name=args.task_name)
    task.connect(vars(args))
    print("ClearML task initialized")

    script_path = os.path.abspath(__file__)
    task.upload_artifact(name="Source Code", artifact_object=script_path)

    resolved_csv = resolve_input_csv(args.input_csv)
    print(f"Using input CSV: {resolved_csv}")

    df = pd.read_csv(resolved_csv)
    df = prepare_dataframe(df, grouping_level=args.grouping_level)
    df = merge_small_groups(df)
    print(f"Loaded {len(df)} chunks, {df['scluster'].nunique()} clusters")

    # --- Centroids & similarity ---
    cluster_ids, centroids, sizes = compute_cluster_centroids(df)
    similarity_matrix = cosine_similarity(centroids)
    print("Computed cluster centroids and similarity matrix")

    # --- 2D map ---
    positions = compute_cluster_map_positions(similarity_matrix)
    print("Computed MDS positions for cluster map")

    # --- Top n-grams per cluster ---
    df_ngrams = add_cluster_ctfidf_ngrams(df, ngram_n=3, top_k=20)
    ngrams_by_cluster = (
        df_ngrams.drop_duplicates("scluster").set_index("scluster")["cluster_top_ngrams"].to_dict()
    )

    # --- Composition ---
    composition = compute_cluster_composition(df, group_col="grouping_label")

    # --- Build figure ---
    fig = build_figure(
        cluster_ids,
        similarity_matrix,
        positions,
        sizes,
        ngrams_by_cluster,
        composition,
        group_col="grouping_label",
    )

    os.makedirs(os.path.dirname(args.output_html) or ".", exist_ok=True)
    fig.write_html(args.output_html, auto_open=False)
    print(f"Saved visualization to: {args.output_html}")

    # --- CSVs ---
    pad = len(str(max(cluster_ids)))
    labels = [f"Cluster {str(cid).zfill(pad)}" for cid in cluster_ids]
    sim_df = pd.DataFrame(similarity_matrix, index=labels, columns=labels)
    sim_csv_path = args.output_html.replace(".html", "_similarity.csv")
    sim_df.to_csv(sim_csv_path, encoding="utf-8-sig")
    print(f"Saved similarity matrix to: {sim_csv_path}")

    top_texts_df = compute_top_texts_per_cluster(df, top_n=args.top_texts)
    top_texts_csv_path = args.output_html.replace(".html", "_top_texts.csv")
    top_texts_df.to_csv(top_texts_csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved top texts per cluster to: {top_texts_csv_path}")

    # --- ClearML reporting ---
    task.upload_artifact(name="Visualization HTML", artifact_object=args.output_html)
    task.upload_artifact(name="Similarity Matrix", artifact_object=sim_df)
    task.upload_artifact(name="Top Texts Per Cluster", artifact_object=top_texts_df)
    task.get_logger().report_table(
        title="Top Texts Per Cluster",
        series="composition",
        iteration=0,
        table_plot=top_texts_df,
    )

    task.get_logger().flush()
    task.flush(wait_for_uploads=True)
    task.close()
    print("ClearML task closed")


if __name__ == "__main__":
    main()
