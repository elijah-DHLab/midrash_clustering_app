"""Cluster visualization with original density calculation and ClearML reporting.

Combines:
- Original density computation from chunk_clustering.py (histogram2d, bins=[50,50])
- Visualization structure from local_density_visualization.py (text boundaries,
  discrete cluster colors, c-TF-IDF n-grams panel, etc.)
- ClearML integration for logging plots, tables, and artifacts.
"""

import argparse
import ast
import os

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ruptures as rpt
from clearml import Task
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "input_csv": "Results/rabani_unsmoothed_table.csv",
    "output_html": "Results/cluster_visualization_report.html",
    "task_name": "cluster_visualization_report",
    "grouping_level": 2,
    "show_text_boundaries": True,
    "max_text_labels": 20,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_hierarchy_list(value):
    if pd.isna(value):
        return tuple()
    text = str(value).strip()
    if not text:
        return tuple()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (tuple, list)):
            return tuple(int(x) for x in parsed)
    except (ValueError, SyntaxError):
        pass
    return tuple()


def _build_collection_name(filename):
    name = str(filename)
    marker = "-."
    idx = name.find(marker)
    return name[:idx] if idx != -1 else name


def _grouping_label(filename, level):
    """Extract a label from filename at the requested hierarchy level (1-3).

    Level 1 -> Corpus                   e.g. 'Talmud'
    Level 2 -> Corpus-Sub               e.g. 'Talmud-Bavli'
    Level 3 -> Corpus-Sub-Book          e.g. 'Talmud-Bavli-Seder Moed'
    """
    base = str(filename)
    if base.endswith("-.json"):
        base = base[: -len("-.json")]
    elif base.endswith(".json"):
        base = base[: -len(".json")]
    parts = base.split("-")
    return "-".join(parts[:level])


def merge_small_groups(df, min_span_fraction=0.025):
    """Collapse grouping_label entries whose chronological span is too small
    into their level-1 parent label."""
    total_span = float(
        df["chronological_index"].max() - df["chronological_index"].min()
    )
    min_span = total_span * min_span_fraction

    spans = df.groupby("grouping_label")["chronological_index"].agg(
        lambda x: x.max() - x.min()
    )
    small_groups = set(spans[spans < min_span].index)

    if small_groups:
        df = df.copy()
        df["grouping_label"] = df["grouping_label"].map(
            lambda label: label.split("-")[0] if label in small_groups else label
        )
    return df


def resolve_input_csv(input_csv):
    if os.path.exists(input_csv):
        return input_csv

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        os.path.join(script_dir, input_csv),
        os.path.join(script_dir, "Results", "rabani_unsmoothed_table.csv"),
        os.path.join(script_dir, "Inputs", "rabani_unsmoothed_table.csv"),
        "G:/My Drive/Haifa/MIDRASH/Results/rabani_unsmoothed_table.csv",
    ]
    for candidate in candidate_paths:
        if os.path.exists(candidate):
            return candidate

    tried = [input_csv] + candidate_paths
    tried_list = "\n".join(f"- {p}" for p in tried)
    raise FileNotFoundError(
        f"Input CSV was not found in any known location. Tried:\n{tried_list}\n"
        "Pass a valid file explicitly with --input-csv."
    )


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def prepare_dataframe(df, grouping_level=2):
    required_cols = {"filename", "scluster"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    if "chronological_index" not in df.columns:
        if "hierarchy_list" in df.columns:
            tmp = df.copy()
            tmp["_hierarchy_tuple"] = tmp["hierarchy_list"].apply(_parse_hierarchy_list)
            tmp = tmp.sort_values(
                by=["_hierarchy_tuple", "filename", "chunk_number"],
                kind="mergesort",
            ).reset_index(drop=True)
            tmp["chronological_index"] = np.arange(len(tmp), dtype=int)
            df = tmp.drop(columns=["_hierarchy_tuple"])
        else:
            df = df.copy().reset_index(drop=True)
            df["chronological_index"] = np.arange(len(df), dtype=int)

    if "chunk_number" not in df.columns:
        df["chunk_number"] = df.groupby("filename").cumcount() + 1

    if "chunk" not in df.columns:
        df["chunk"] = ""

    df["collection"] = df["filename"].apply(_build_collection_name)
    df["text_start"] = df["chunk"].astype(str).str.slice(0, 120)
    df["grouping_label"] = df["filename"].apply(
        lambda f: _grouping_label(f, grouping_level)
    )

    df["scluster"] = pd.to_numeric(df["scluster"], errors="coerce")
    df = df.dropna(subset=["scluster"]).copy()
    df["scluster"] = df["scluster"].astype(int)

    # Always recompute chronological_index from 0 after filtering
    df = df.sort_values("chronological_index").reset_index(drop=True)
    df["chronological_index"] = np.arange(len(df), dtype=int)

    return df


# ---------------------------------------------------------------------------
# Original density calculation (from chunk_clustering.py)
# ---------------------------------------------------------------------------


def compute_density_original(df):
    """Density via 2D histogram with fixed 50x50 bins — the original method."""
    hist, xedges, yedges = np.histogram2d(
        df["chronological_index"],
        df["scluster"],
        bins=[50, 50],
    )

    x_bin_indices = np.digitize(df["chronological_index"], xedges) - 1
    y_bin_indices = np.digitize(df["scluster"], yedges) - 1

    x_bin_indices[x_bin_indices >= hist.shape[0]] = hist.shape[0] - 1
    y_bin_indices[y_bin_indices >= hist.shape[1]] = hist.shape[1] - 1

    density = hist[x_bin_indices, y_bin_indices]
    df = df.copy()
    df["density_raw"] = density.astype(float)

    # Normalize for marker sizing
    d_min = df["density_raw"].min()
    d_max = df["density_raw"].max()
    if d_max > d_min:
        df["density_normalized"] = (df["density_raw"] - d_min) / (d_max - d_min)
    else:
        df["density_normalized"] = 0.0

    df["marker_size"] = 4.0 + 8.0 * np.sqrt(df["density_normalized"])
    return df


# ---------------------------------------------------------------------------
# Per-chunk cluster affinities (own cluster vs. nearest neighboring cluster)
# ---------------------------------------------------------------------------


def _parse_vec_fast(vec_str):
    """Parse a '[0.1, 0.2, ...]' string to a float array without ast.literal_eval."""
    cleaned = str(vec_str).strip().strip("[]")
    if not cleaned:
        return None
    return np.fromstring(cleaned, sep=",")


def compute_chunk_cluster_affinities(df):
    """For each chunk, compute cosine similarity to its own cluster's
    centroid and to the nearest neighboring cluster's centroid (i.e. the
    second-best matching cluster), plus the margin between them."""
    if "vec" not in df.columns:
        print("WARNING: 'vec' column not found — skipping chunk affinities.")
        return df

    vecs = df["vec"].apply(_parse_vec_fast)
    valid = vecs.apply(lambda v: v is not None and v.size > 0)
    if not valid.all():
        print(
            f"WARNING: {(~valid).sum()} chunks have unparsable vectors — "
            "affinities set to NaN for them."
        )

    df = df.copy()
    cluster_ids = sorted(df["scluster"].unique())
    centroids = []
    for cid in cluster_ids:
        mask = valid & (df["scluster"] == cid)
        cluster_vecs = np.stack(vecs[mask].values)
        centroids.append(cluster_vecs.mean(axis=0))
    centroid_matrix = np.stack(centroids)

    chunk_matrix = np.stack(vecs[valid].values)
    sim_matrix = cosine_similarity(chunk_matrix, centroid_matrix)

    cluster_id_to_col = {cid: i for i, cid in enumerate(cluster_ids)}
    own_col = df.loc[valid, "scluster"].map(cluster_id_to_col).to_numpy()
    row_idx = np.arange(len(sim_matrix))

    own_sim = sim_matrix[row_idx, own_col]

    masked = sim_matrix.copy()
    masked[row_idx, own_col] = -np.inf
    nearest_other_col = np.argmax(masked, axis=1)
    nearest_other_sim = masked[row_idx, nearest_other_col]
    nearest_other_cluster = np.array(cluster_ids)[nearest_other_col]

    for col in ("own_cluster_sim", "nearest_other_cluster", "nearest_other_sim", "cluster_margin"):
        df[col] = np.nan

    df.loc[valid, "own_cluster_sim"] = own_sim
    df.loc[valid, "nearest_other_cluster"] = nearest_other_cluster
    df.loc[valid, "nearest_other_sim"] = nearest_other_sim
    df.loc[valid, "cluster_margin"] = own_sim - nearest_other_sim

    return df


# ---------------------------------------------------------------------------
# c-TF-IDF n-grams per cluster
# ---------------------------------------------------------------------------


def add_cluster_ctfidf_ngrams(df, ngram_n=3, top_k=20):
    if "chunk" not in df.columns or df["chunk"].fillna("").eq("").all():
        df = df.copy()
        df["cluster_top_ngrams"] = "No n-grams available"
        return df

    cluster_docs = (
        df.groupby("scluster")["chunk"]
        .apply(
            lambda chunks: " ".join(
                str(c) for c in chunks if str(c).strip()
            )
        )
        .sort_index()
    )

    vectorizer = CountVectorizer(
        ngram_range=(ngram_n, ngram_n),
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b",
        min_df=1,
    )

    counts = vectorizer.fit_transform(cluster_docs.values)
    if counts.shape[1] == 0:
        df = df.copy()
        df["cluster_top_ngrams"] = "No n-grams available"
        return df

    counts_dense = (
        counts.toarray().astype(float)
        if hasattr(counts, "toarray")
        else np.asarray(counts, dtype=float)
    )
    row_sums = counts_dense.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    tf = counts_dense / row_sums

    doc_freq = (counts_dense > 0).sum(axis=0)
    n_classes = counts_dense.shape[0]
    idf = np.log((1 + n_classes) / (1 + doc_freq)) + 1.0
    ctfidf = tf * idf

    feature_names = vectorizer.get_feature_names_out()
    top_ngrams_by_cluster = {}
    for row_idx, cluster_id in enumerate(cluster_docs.index):
        scores = ctfidf[row_idx]
        nonzero = np.flatnonzero(scores > 0)
        if len(nonzero) == 0:
            top_ngrams_by_cluster[int(cluster_id)] = "No n-grams available"
            continue
        top_indices = nonzero[np.argsort(scores[nonzero])[::-1][:top_k]]
        top_terms = [str(feature_names[i]) for i in top_indices]
        top_ngrams_by_cluster[int(cluster_id)] = "<br>".join(top_terms)

    df = df.copy()
    df["cluster_top_ngrams"] = df["scluster"].map(top_ngrams_by_cluster)
    return df


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def add_text_boundaries(fig, df):
    col_boundaries = (
        df.groupby("grouping_label", as_index=False)
        .agg(
            x_start=("chronological_index", "min"),
            x_end=("chronological_index", "max"),
        )
        .sort_values("x_start")
        .reset_index(drop=True)
    )

    if col_boundaries.empty:
        return fig

    for i, row in col_boundaries.iterrows():
        if i % 2 == 0:
            fig.add_vrect(
                x0=float(row["x_start"]),
                x1=float(row["x_end"]),
                fillcolor="rgba(120, 120, 120, 0.06)",
                line_width=0,
                layer="below",
            )

    for _, row in col_boundaries.iterrows():
        fig.add_vline(
            x=float(row["x_start"]),
            line_width=1.5,
            line_color="rgba(60, 60, 60, 0.35)",
        )
    fig.add_vline(
        x=float(col_boundaries.iloc[-1]["x_end"]),
        line_width=1.5,
        line_color="rgba(60, 60, 60, 0.35)",
    )

    total_span = float(col_boundaries["x_end"].max()) - float(
        col_boundaries["x_start"].min()
    )
    min_span = total_span * 0.025

    tick_vals = []
    tick_text = []
    for _, row in col_boundaries.iterrows():
        span = float(row["x_end"]) - float(row["x_start"])
        if span < min_span:
            continue
        x_pos = float(row["x_start"])
        label = str(row["grouping_label"])
        if len(label) > 30:
            label = label[:27] + "..."
        tick_vals.append(x_pos)
        tick_text.append(label)

    fig.update_xaxes(
        tickvals=tick_vals,
        ticktext=tick_text,
        tickangle=-40,
        tickfont=dict(size=11),
    )
    return fig


def build_figure(df, show_text_boundaries=True, change_points=None, render_mode="svg"):
    """`render_mode="webgl"` draws the chunk layer on the GPU instead of as SVG.

    At a few thousand points SVG is fine and exports as vector. Past roughly twenty
    thousand every hover, pan and zoom has to touch that many DOM nodes, and the
    figure stops being usable; WebGL makes those interactions immediate. Nothing
    about the data or the clustering changes — only how the same points are drawn.
    Default stays "svg" so the command-line pipeline is unaffected.
    """
    n_clusters = int(df["scluster"].max()) + 1
    pad = len(str(n_clusters - 1))

    df = df.copy()
    df["scluster_str"] = "Cluster " + df["scluster"].astype(str).str.zfill(pad)
    legend_order = [
        f"Cluster {str(cid).zfill(pad)}"
        for cid in sorted(df["scluster"].unique(), reverse=True)
    ]
    df["scluster_str"] = pd.Categorical(
        df["scluster_str"], categories=legend_order, ordered=True
    )

    cluster_ids = sorted(df["scluster"].unique())
    if len(cluster_ids) == 1:
        positions = [0.5]
    else:
        positions = [cid / max(cluster_ids) for cid in cluster_ids]
    sampled_colors = px.colors.sample_colorscale(
        px.colors.sequential.Plasma, positions
    )
    color_map = {
        f"Cluster {str(cid).zfill(pad)}": color
        for cid, color in zip(cluster_ids, sampled_colors)
    }

    fig = px.scatter(
        df,
        x="chronological_index",
        y="scluster",
        color="scluster_str",
        color_discrete_map=color_map,
        category_orders={"scluster_str": legend_order},
        size="marker_size",
        size_max=14,
        # Plotly Express switches to WebGL (Scattergl) automatically above
        # 1000 rows. The boundary lines/rects (add_vline/add_vrect) are
        # always drawn in the SVG shape layer, which is redrawn on a
        # different pipeline than a WebGL canvas — during interactive
        # pan/zoom the two layers fall a frame or more out of sync, which
        # looks like the lines drifting relative to the dots. Forcing SVG
        # keeps both layers on the same rendering path.
        render_mode=render_mode,
        custom_data=[
            "collection",
            "chunk_number",
            "text_start",
            "density_normalized",
            "chronological_index",
            "own_cluster_sim",
            "nearest_other_cluster",
            "nearest_other_sim",
            "cluster_margin",
        ],
        title="Cluster Scatter Over Time (Original Density)",
        labels={
            "chronological_index": "Chronological Index",
            "scluster": "Cluster ID",
            "density_raw": "Density",
            "scluster_str": "Cluster",
        },
    )

    row_height = 35
    fixed_margins = 60 + 160  # top + bottom margin, holds regardless of k
    fig.update_layout(
        autosize=True,
        height=max(500, fixed_margins + n_clusters * row_height),
        margin=dict(l=70, r=160, t=60, b=160),
        legend_title_text="Cluster",
        dragmode="pan",
    )
    fig.update_yaxes(dtick=1, tick0=0)
    fig.update_traces(
        marker=dict(opacity=0.82, line=dict(width=0)),
        hovertemplate="<b>Chronological index:</b> %{customdata[4]}<br>"
        + "<b>Chunk index in text:</b> %{customdata[1]}<br>"
        + "<b>Cluster:</b> %{y}<br>"
        + "<b>Collection:</b> %{customdata[0]}<br>"
        + "<b>Text:</b> %{customdata[2]}<br>"
        + "<b>Density:</b> %{customdata[3]:.2f}<br>"
        + "<b>Similarity to own cluster:</b> %{customdata[5]:.3f}<br>"
        + "<b>Nearest other cluster:</b> %{customdata[6]:.0f} "
        + "(sim: %{customdata[7]:.3f})<br>"
        + "<b>Margin (own − nearest other):</b> %{customdata[8]:.3f}<br>",
    )

    if show_text_boundaries:
        fig = add_text_boundaries(fig, df)

    # --- CPD change-point lines with score annotations & cluster contributions ---
    if change_points:
        y_max = float(df["scluster"].max())
        y_min = float(df["scluster"].min())
        for cp_entry in change_points:
            cp_x, cp_score = cp_entry[0], cp_entry[1]
            contributions = cp_entry[2] if len(cp_entry) > 2 else []

            fig.add_vline(
                x=int(cp_x),
                line_width=2.5,
                line_dash="dash",
                line_color="red",
                opacity=0.75,
            )
            fig.add_annotation(
                x=int(cp_x),
                y=y_max + 0.8,
                text=f"{cp_score:.2f}",
                showarrow=True,
                arrowhead=0,
                arrowcolor="red",
                ax=0,
                ay=-20,
                font=dict(size=10, color="red"),
                xanchor="center",
                yanchor="bottom",
            )

            # Invisible hover trace on the red line showing top contributing clusters
            if contributions:
                top_n = contributions[:5]
                hover_lines = [f"<b>Change point at chunk {cp_x}</b>",
                               f"<b>Score: {cp_score:.3f}</b>", ""]
                for cid, delta in top_n:
                    direction = "+" if delta > 0 else ""
                    hover_lines.append(
                        f"Cluster {cid}: {direction}{delta:.3f}"
                    )
                hover_text = "<br>".join(hover_lines)

                y_positions = np.linspace(y_min, y_max, min(8, int(y_max - y_min) + 1))
                fig.add_trace(
                    go.Scatter(
                        x=[int(cp_x)] * len(y_positions),
                        y=y_positions.tolist(),
                        mode="markers",
                        marker=dict(size=12, color="rgba(0,0,0,0)", line=dict(width=0)),
                        hovertemplate=hover_text + "<extra></extra>",
                        showlegend=False,
                    )
                )

    # --- Side panel with hoverable cluster dots showing top n-grams ---
    panel_x = 0.5
    fig.update_layout(
        xaxis=dict(domain=[0.0, 0.88]),
        xaxis2=dict(
            domain=[0.90, 0.98],
            range=[0.0, 1.0],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            visible=False,
        ),
        showlegend=False,
    )

    # Compute cluster sizes (number of chunks per cluster)
    cluster_sizes = df.groupby("scluster").size().rename("cluster_size")

    cluster_profiles = (
        df.sort_values(by="scluster", ascending=False)
        .drop_duplicates(subset=["scluster"])[
            ["scluster", "scluster_str", "cluster_top_ngrams"]
        ]
        .sort_values(by="scluster", ascending=False)
        .merge(cluster_sizes, left_on="scluster", right_index=True)
    )

    # Scale panel dot size by cluster size
    max_size = cluster_profiles["cluster_size"].max()
    cluster_profiles["panel_dot_size"] = (
        8.0 + 14.0 * np.sqrt(cluster_profiles["cluster_size"] / max_size)
    )

    for _, row in cluster_profiles.iterrows():
        cluster_label = str(row["scluster_str"])
        fig.add_trace(
            go.Scatter(
                x=[panel_x],
                y=[int(row["scluster"])],
                mode="markers+text",
                text=[str(int(row["scluster"]))],
                textposition="middle center",
                textfont=dict(color="white", size=10, family="Arial Black"),
                marker=dict(
                    size=row["panel_dot_size"],
                    color=color_map.get(cluster_label, "gray"),
                    line=dict(width=0.5, color="rgba(0,0,0,0.4)"),
                ),
                customdata=[
                    [cluster_label, str(row["cluster_top_ngrams"]), int(row["cluster_size"])]
                ],
                hovertemplate="<b>%{customdata[0]}</b><br>"
                + "<b>Size:</b> %{customdata[2]} chunks<br>"
                + "<b>Top 3-grams:</b><br>%{customdata[1]}<extra></extra>",
                showlegend=False,
                xaxis="x2",
                yaxis="y",
            )
        )

    fig.add_annotation(
        x=panel_x,
        y=float(df["scluster"].max()) + 0.6,
        xref="x2",
        yref="y",
        text="Clusters",
        showarrow=False,
        xanchor="center",
        font=dict(size=12),
    )

    return fig


# ---------------------------------------------------------------------------
# Change-point detection (PELT on block-level cluster-composition vectors)
# ---------------------------------------------------------------------------


def compute_change_points(df, block_size, penalty):
    """Run PELT change-point detection on block-level cluster-composition
    vectors, and assign a 1-indexed 'segment' column to df.

    Returns (df, change_points) where change_points is a list of
    (chronological_index, score, cluster_contributions) tuples, sorted by
    position, and cluster_contributions is a list of (cluster_id, signed
    delta) pairs sorted by |delta| descending.
    """
    cluster_seq = df.sort_values("chronological_index")["scluster"].to_numpy()
    n_clusters_cpd = int(cluster_seq.max()) + 1
    n_blocks = len(cluster_seq) // block_size
    distributions = np.zeros((n_blocks, n_clusters_cpd), dtype=float)
    block_edges = np.zeros(n_blocks, dtype=int)
    for b in range(n_blocks):
        lo = b * block_size
        counts = np.bincount(
            cluster_seq[lo : lo + block_size], minlength=n_clusters_cpd
        ).astype(float)
        distributions[b] = counts / counts.sum()
        block_edges[b] = lo

    model = rpt.Pelt(model="l2", min_size=2, jump=1).fit(distributions)
    cp_block_indices = model.predict(pen=penalty)
    if cp_block_indices and cp_block_indices[-1] == n_blocks:
        cp_block_indices = cp_block_indices[:-1]

    change_points = []
    for bi in cp_block_indices:
        cp_x = int(block_edges[bi]) if bi < n_blocks else int(len(cluster_seq))
        dist_before = distributions[bi - 1] if bi > 0 else distributions[0]
        dist_after = distributions[bi] if bi < n_blocks else distributions[-1]
        diff = dist_after - dist_before  # signed per-cluster change
        score = float(np.linalg.norm(diff))
        abs_diff = np.abs(diff)
        top_indices = np.argsort(abs_diff)[::-1]
        cluster_contributions = [
            (int(c), float(diff[c])) for c in top_indices if abs_diff[c] > 1e-6
        ]
        change_points.append((cp_x, score, cluster_contributions))

    # --- Assign segment index to each chunk ---
    cp_positions = sorted(cp[0] for cp in change_points)
    seg_boundaries = [0] + cp_positions + [len(df)]
    df = df.copy()
    df["segment"] = 0
    for seg_idx in range(len(seg_boundaries) - 1):
        mask = (df["chronological_index"] >= seg_boundaries[seg_idx]) & (
            df["chronological_index"] < seg_boundaries[seg_idx + 1]
        )
        df.loc[mask, "segment"] = seg_idx + 1

    return df, change_points


# ---------------------------------------------------------------------------
# Segment composition table (between CPD change points)
# ---------------------------------------------------------------------------


def build_segment_composition_table(df, change_points):
    """Build a table showing the text composition of each segment between CPD
    change points.

    For each segment, lists which texts (filenames) it contains, what percentage
    of the segment each text occupies, and what percentage of each text falls
    within that segment.
    """
    if not change_points:
        return None

    df_sorted = df.sort_values("chronological_index").reset_index(drop=True)
    total_chunks = len(df_sorted)

    # Build segment boundaries: [0, cp1, cp2, ..., end]
    # Sort change points by position; each entry is (cp_x, score, contributions)
    sorted_cps = sorted(change_points, key=lambda cp: cp[0])
    cp_positions = [cp[0] for cp in sorted_cps]
    boundaries = [0] + cp_positions + [total_chunks]

    # Total chunks per filename (for computing % of text in segment)
    total_per_file = df_sorted.groupby("filename").size()

    rows = []
    for seg_idx in range(len(boundaries) - 1):
        seg_start = boundaries[seg_idx]
        seg_end = boundaries[seg_idx + 1]

        seg_df = df_sorted[
            (df_sorted["chronological_index"] >= seg_start)
            & (df_sorted["chronological_index"] < seg_end)
        ]
        seg_size = len(seg_df)
        if seg_size == 0:
            continue

        # Count chunks per filename in this segment
        file_counts = seg_df.groupby("filename").size().sort_index()

        composition_parts = []
        for fname, count in file_counts.items():
            pct_of_segment = 100.0 * count / seg_size
            pct_of_text = 100.0 * count / total_per_file[fname]
            if pct_of_text >= 99.5:
                text_coverage = "100%"
            else:
                text_coverage = f"{pct_of_text:.0f}%"
            composition_parts.append(
                f"{text_coverage} of {fname} ({pct_of_segment:.1f}% of segment)"
            )

        composition_str = " + ".join(composition_parts)

        # Cluster frequencies within this segment
        cluster_freq = seg_df["scluster"].value_counts(normalize=True).sort_index()

        # Top contributing clusters at the change point that STARTS this segment
        # (the first segment has no preceding change point)
        cp_contributors = ""
        if seg_idx > 0:
            cp_entry = sorted_cps[seg_idx - 1]
            contributions = cp_entry[2] if len(cp_entry) > 2 else []
            top5 = contributions[:5]
            parts = []
            for cid, delta in top5:
                direction = "+" if delta > 0 else ""
                freq_pct = cluster_freq.get(cid, 0.0) * 100
                parts.append(f"Cluster {cid} ({direction}{delta:.3f}, freq: {freq_pct:.1f}%)")
            cp_contributors = ", ".join(parts)

        rows.append(
            {
                "Segment": seg_idx + 1,
                "Start Chunk": seg_start,
                "End Chunk": seg_end - 1,
                "Num Chunks": seg_size,
                "Composition": composition_str,
                "Top Contributing Clusters": cp_contributors,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Nearest neighbors (non-chronological similarity)
# ---------------------------------------------------------------------------


def _parse_vec_string(vec_str):
    """Parse a vector stored as a string back to a numpy array."""
    try:
        parsed = ast.literal_eval(str(vec_str))
        return np.array(parsed, dtype=float)
    except (ValueError, SyntaxError):
        return None


def compute_segment_nearest_neighbors(df, k=5):
    """For each segment (between CPD change points), compute an average vector
    from all its chunks, then find the k most similar segments by cosine
    similarity — ignoring chronological order.

    Requires 'segment' and 'vec' columns in df.
    """
    if "vec" not in df.columns:
        print("WARNING: 'vec' column not found — skipping segment neighbors.")
        return None
    if "segment" not in df.columns:
        print("WARNING: 'segment' column not found — run with --show-change-points.")
        return None

    # Parse vectors
    vecs = df["vec"].apply(_parse_vec_string)
    valid_mask = vecs.apply(lambda v: v is not None)
    df_valid = df[valid_mask].copy()
    df_valid["_vec"] = vecs[valid_mask]

    segments = sorted(df_valid["segment"].unique())
    if len(segments) < 2:
        print("WARNING: fewer than 2 segments — skipping segment neighbors.")
        return None

    # Compute average vector per segment
    seg_vecs = {}
    seg_info = {}
    for seg_id in segments:
        seg_df = df_valid[df_valid["segment"] == seg_id]
        seg_matrix = np.stack(seg_df["_vec"].values).astype(np.float32)
        seg_vecs[seg_id] = seg_matrix.mean(axis=0)
        seg_info[seg_id] = {
            "start": int(seg_df["chronological_index"].min()),
            "end": int(seg_df["chronological_index"].max()),
            "n_chunks": len(seg_df),
            "collections": ", ".join(sorted(seg_df["collection"].unique())),
        }

    seg_ids = list(seg_vecs.keys())
    seg_matrix = np.stack([seg_vecs[s] for s in seg_ids])

    # Cosine similarity
    sim_matrix = cosine_similarity(seg_matrix)
    np.fill_diagonal(sim_matrix, -1.0)

    rows = []
    for i, seg_id in enumerate(seg_ids):
        info = seg_info[seg_id]
        top_idx = np.argsort(sim_matrix[i])[::-1][:k]
        row_data = {
            "Segment": seg_id,
            "Chunks": f"{info['start']}-{info['end']}",
            "Num Chunks": info["n_chunks"],
            "Collections": info["collections"],
        }
        for rank, j in enumerate(top_idx):
            neighbor_id = seg_ids[j]
            neighbor_info = seg_info[neighbor_id]
            sim_val = sim_matrix[i, j]
            row_data[f"neighbor_{rank+1}"] = (
                f"Segment {neighbor_id} "
                f"(chunks {neighbor_info['start']}-{neighbor_info['end']}, "
                f"sim={sim_val:.3f})"
            )
        rows.append(row_data)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Cluster visualization with original density and ClearML reporting."
    )
    parser.add_argument(
        "--input-csv",
        default=DEFAULT_CONFIG["input_csv"],
        help="Path to clustered chunks CSV.",
    )
    parser.add_argument(
        "--output-html",
        default=DEFAULT_CONFIG["output_html"],
        help="Output path for Plotly HTML.",
    )
    parser.add_argument(
        "--task-name",
        default=DEFAULT_CONFIG["task_name"],
        help="ClearML task name.",
    )
    parser.add_argument(
        "--grouping-level",
        type=int,
        choices=[1, 2, 3],
        default=DEFAULT_CONFIG["grouping_level"],
        help="Hierarchy level for X-axis labels (1=Corpus, 2=Sub-corpus, 3=Book).",
    )
    parser.add_argument(
        "--hide-text-boundaries",
        action="store_true",
        help="Disable text boundary lines and labels on X axis.",
    )
    parser.add_argument(
        "--show-change-points",
        action="store_true",
        help="Run PELT change-point detection and overlay results as red dashed lines.",
    )
    parser.add_argument(
        "--cpd-penalty",
        type=float,
        default=0.1,
        help="PELT penalty for CPD. Lower = more change points (default: 0.1).",
    )
    parser.add_argument(
        "--cpd-block-size",
        type=int,
        default=50,
        help="Block size for CPD distribution aggregation (default: 50 chunks).",
    )
    parser.add_argument(
        "--show-nearest-neighbors",
        action="store_true",
        help="Compute top-5 nearest neighbors per chunk (cosine similarity on vectors).",
    )
    parser.add_argument(
        "--nn-k",
        type=int,
        default=5,
        help="Number of nearest neighbors to find (default: 5).",
    )
    args = parser.parse_args()

    # --- ClearML ---
    task_name = args.task_name
    if args.show_change_points:
        task_name = task_name + "_CPD"
    task = Task.init(project_name="MIDRASH", task_name=task_name)
    task.connect(vars(args))
    print("ClearML task initialized")

    # Upload this script as artifact
    script_path = os.path.abspath(__file__)
    task.upload_artifact(name="Source Code", artifact_object=script_path)
    print(f"Uploaded {script_path} as artifact")

    # --- Load & prepare ---
    resolved_csv = resolve_input_csv(args.input_csv)
    print(f"Using input CSV: {resolved_csv}")

    df = pd.read_csv(resolved_csv)
    df = prepare_dataframe(df, grouping_level=args.grouping_level)
    df = merge_small_groups(df)
    print(
        f"Grouping level {args.grouping_level}: "
        f"{df['grouping_label'].nunique()} groups (after merging small groups)"
    )

    # --- Original density (bins=[50,50]) ---
    df = compute_density_original(df)
    print("Computed density (original 50x50 histogram2d)")

    # --- Per-chunk cluster affinities (own vs. nearest neighboring cluster) ---
    df = compute_chunk_cluster_affinities(df)
    print("Computed per-chunk cluster affinities")

    # --- c-TF-IDF n-grams ---
    df = add_cluster_ctfidf_ngrams(df, ngram_n=3, top_k=20)
    print("Computed c-TF-IDF n-grams per cluster")

    # --- Optional CPD ---
    change_points = None
    if args.show_change_points:
        df, change_points = compute_change_points(
            df, block_size=args.cpd_block_size, penalty=args.cpd_penalty
        )
        print(
            f"CPD: found {len(change_points)} change points "
            f"(penalty={args.cpd_penalty}, block_size={args.cpd_block_size})"
        )
        print(f"Assigned segment index (1-{int(df['segment'].max())}) to each chunk")

        # --- Segment composition table ---
        segment_table = build_segment_composition_table(df, change_points)
        if segment_table is not None:
            seg_csv_path = args.output_html.replace(".html", "_segments.csv")
            segment_table.to_csv(seg_csv_path, index=False, encoding="utf-8-sig")
            print(f"Saved segment composition table to: {seg_csv_path}")
            task.upload_artifact(
                name="Segment Composition Table", artifact_object=segment_table
            )
            task.get_logger().report_table(
                title="Segment Composition",
                series="CPD segments",
                iteration=0,
                table_plot=segment_table,
            )

    # --- Optional segment nearest neighbors ---
    if args.show_nearest_neighbors:
        nn_table = compute_segment_nearest_neighbors(df, k=args.nn_k)
        if nn_table is not None:
            nn_csv_path = args.output_html.replace(".html", "_segment_neighbors.csv")
            nn_table.to_csv(nn_csv_path, index=False, encoding="utf-8-sig")
            print(f"Saved segment neighbors table to: {nn_csv_path}")
            task.upload_artifact(
                name="Segment Neighbors Table", artifact_object=nn_table
            )
            task.get_logger().report_table(
                title="Segment Nearest Neighbors (Top-5)",
                series="similarity",
                iteration=0,
                table_plot=nn_table.head(20),
            )

    # --- Build figure ---
    fig = build_figure(
        df,
        show_text_boundaries=not args.hide_text_boundaries,
        change_points=change_points,
    )

    # --- Save HTML ---
    os.makedirs(os.path.dirname(args.output_html) or ".", exist_ok=True)
    fig.write_html(args.output_html, auto_open=False)
    print(f"Saved visualization to: {args.output_html}")

    # --- Report to ClearML ---
    # Upload full HTML as artifact (user will open in browser)
    task.upload_artifact(
        name="Visualization HTML", artifact_object=args.output_html
    )

    # Report summary table
    summary_cols = [
        c for c in df.columns if c not in ("vec", "cluster_top_ngrams")
    ]
    task.get_logger().report_table(
        title="Chunks Summary",
        series="data",
        iteration=0,
        table_plot=df[summary_cols].head(10),
    )
    task.upload_artifact(name="Chunks DataFrame", artifact_object=df)

    task.get_logger().flush()
    task.flush(wait_for_uploads=True)
    task.close()
    print("ClearML task closed")


if __name__ == "__main__":
    main()
