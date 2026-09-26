# Diachronic-Stylistic Clustering of Rabbinic Literature — Pipeline Description

**Status:** covers the core pipeline (corpus → chunking → embedding → clustering → cluster map) and the chronological mapping / change-point detection stage that sits on top of it.

## 1. Research framing

The corpus of rabbinic and midrashic literature is treated as a single, ordered sequence of text spanning multiple historical strata (e.g. Apocrypha, Halakhah, Midrash, Talmud, later commentaries). The guiding idea is **diachronic-stylistic analysis**: represent the corpus as a sequence of short text segments placed along its canonical chronological order, embed each segment, and group segments by distributional similarity (clustering) independent of chronology. The resulting clusters function as a data-driven proxy for recurring stylistic/thematic modes in the literature, which can then be examined against the chronological axis (in later stages of the pipeline) to look for points of stylistic or thematic transition between literary strata/periods.

This document describes the pipeline up to and including the **cluster map** — i.e., the static structure of the clusters themselves (how many there are, how similar they are to one another, and what each is composed of) — before any temporal analysis is layered on top.

## 2. Corpus and metadata

- **Source**: Sefaria digitized rabbinic/midrashic texts (`SefariasData/rabbinic`), one JSONL file per work. Each line is a JSON object with a `"sentence"` field; a file's full text is the concatenation of its sentences.
- **Filename convention**: `Corpus-SubCorpus-Book-.json` (e.g. `Halakhah-Mishneh Torah-Sefer Ahavah-...`), which doubles as a hierarchical grouping label (corpus / sub-corpus / book) used later for composition analysis.
- **Metadata table** (`Inputs/avg_text_length_per_file.xlsx`): one row per file, with:
  - `hierarchy` — an ordinal encoding of the file's position in the canonical chronological order of the corpus. This is the backbone that turns an unordered file collection into a single ordered sequence.
  - `avg_text_length` — mean text length per file.
  - Six boolean "Experiment" columns (`Experiment 1`–`Experiment 6`), each defining a different sub-corpus selection for analysis (e.g. all rabbinic literature, midrashic literature only, Tanhuma with/without Yalkut, midrashic literature excluding anthologies, early halakhic/amoraic/tanhumaic literature). A given analysis run selects one of these as its working dataset.

**TODO**: document precisely how `hierarchy` was assigned (source/methodology for the canonical chronological ordering) and what principled basis defines each Experiment subset — needed for a reproducible Methods section.

## 3. Preprocessing: segmentation into chunks

For a chosen sub-corpus (Experiment N):
1. Files are filtered to those flagged for that Experiment, then sorted by `hierarchy` so the entire sub-corpus forms one chronologically ordered stream of text.
2. Each file's sentences are concatenated and split into whitespace-delimited tokens.
3. The token stream is partitioned into **non-overlapping chunks of 50 words**, in order. This is a fixed word-count window, not a linguistic unit (sentence/paragraph) — chunk boundaries can fall mid-sentence.
4. Each chunk retains its source filename, its `hierarchy`, its within-file `chunk_number`, and a corpus-wide `chronological_index` (its position in the fully ordered, cross-file sequence). `chronological_index` is the single axis that anchors every chunk in "corpus time," independent of which file it came from.

**Design note**: fixed-length word windows (rather than sentence/verse units) were chosen for embedding uniformity; this trades linguistic naturalness for a constant unit size going into the embedding step.

## 4. Text representation: embeddings

Each 50-word chunk is embedded as a single dense vector using a pretrained **FastText** model (`get_sentence_vector`), producing one point per chunk in a shared vector space. FastText was chosen presumably for its subword handling, which is favorable for Hebrew's rich morphology, though word-level tokenization (via naive `.split()`) is used upstream rather than any morphological segmentation.

**TODO**: record the FastText model's provenance — training corpus, vector dimensionality, and hyperparameters — for the Methods section (currently referenced only as an external `.bin` file with no documented training details in this repo).

## 5. Clustering

- **Algorithm**: K-Means (`sklearn.cluster.KMeans`), applied to the full chunk-embedding matrix, with `random_state=42` for reproducibility.
- **Number of clusters (k)**: a default of k=10 is used for the main run; a dedicated model-selection script sweeps k over a range (2–30) and reports both inertia (elbow method) and mean silhouette score (computed on a subsample for tractability) to support a principled choice of k.
- **Optional temporal smoothing**: an experimental flag can apply a symmetric convolution kernel across the embedding dimensions before clustering (intended to smooth embeddings using neighboring chunks' values). The main/reported results use this **disabled** (unsmoothed).
- **Output**: each chunk is assigned a discrete cluster label. This label is a similarity-based grouping — it does not use `chronological_index` at all — so clusters can and do draw chunks from many different points in the chronological sequence.

The full per-chunk table (filename, hierarchy, chunk text, embedding vector, `chronological_index`, cluster label) is persisted as the canonical intermediate artifact that all downstream analysis reads from.

## 6. Cluster characterization

For each cluster:
- **Centroid**: mean embedding vector of all chunks assigned to it.
- **Textual signature**: top n-grams (unigrams/bigrams/trigrams, and specifically top trigrams for the map) ranked by **class-based TF-IDF (c-TF-IDF)** — term frequency within the cluster's concatenated text, weighted by inverse document frequency across clusters (treating each cluster as one "document"). This surfaces vocabulary that is both frequent in, and distinctive to, a given cluster, rather than merely frequent overall.

## 7. Cluster map construction

The cluster map turns the clustering result into an interpretable 2D layout of inter-cluster structure (not a temporal view):

1. **Inter-cluster similarity**: pairwise cosine similarity between cluster centroids.
2. **2D projection**: classical multidimensional scaling (MDS) on the centroid similarity matrix (converted to a distance matrix as `1 − cosine similarity`), placing more similar clusters closer together in the plane.
3. **Cluster size encoding**: marker size proportional to the number of chunks in the cluster (√-scaled).
4. **Composition analysis**: for each cluster, the percentage breakdown of its member chunks by source corpus/sub-corpus (using the filename-derived hierarchical grouping label), visualized as a stacked bar chart — showing which parts of the literature contribute to each stylistic/thematic cluster.
5. **Supporting outputs**: a full pairwise cosine-similarity heatmap across clusters, and a per-cluster table of the top contributing source texts (share of the cluster occupied by each text, and share of each text falling into that cluster).

Together, steps 1–5 answer: *how many distinct stylistic/thematic modes does K-Means recover, how distinct are they from one another, and which parts of the literature contribute to each one* — as a static structure, prior to relating any of it back to chronological position.

## 8. Chronological mapping and change-point detection

This stage relates the (chronology-agnostic) clustering result back to the corpus's chronological axis, in order to detect points where the literature's stylistic/thematic composition shifts.

### 8.1 The chronological axis

Every chunk carries a `chronological_index`: its position in the fully ordered, cross-file, cross-corpus sequence produced in §3 (chunks sorted by `hierarchy`, then by file, then by within-file order). This single integer axis is what "chronological" means throughout the pipeline — it is a rank/order over chunks, not a calendar date.

### 8.2 Chunk-level view: cluster label over time

The most direct mapping plots each chunk as a point at `(chronological_index, cluster label)`. Read left to right, this shows — chunk by chunk — which stylistic/thematic cluster the text belongs to at each position in the corpus's chronological order. On its own this is noisy at the single-chunk level (clusters interleave locally), so it is read together with the block-level composition signal below rather than in isolation. Marker density (via a 2D histogram over chronological index × cluster) is used to visually emphasize where points concentrate.

### 8.3 Block-level view: cluster *composition* over time

To see "what is this stretch of the corpus made of" rather than "what is this one chunk," the chronological sequence of per-chunk cluster labels is aggregated into fixed-size, non-overlapping **blocks** (a configurable number of consecutive chunks, e.g. 50 chunks per block). For each block *b*, a **cluster-composition vector** is computed:

> distribution(b) = [ proportion of block *b*'s chunks in cluster 0, cluster 1, …, cluster k−1 ]

This turns the corpus into a sequence of probability vectors over the *k* clusters — one vector per block, indexed by the block's starting `chronological_index`. This distribution sequence, not the raw per-chunk labels, is the signal that change-point detection actually operates on.

### 8.4 Detecting change points

Two complementary measures are used over the block-distribution sequence:

- **Jensen-Shannon divergence (JSD)** between each pair of *adjacent* blocks — a bounded, symmetric divergence between two probability distributions. Plotted as a curve alongside the chunk scatter, local peaks in JSD flag abrupt shifts in cluster composition between consecutive blocks.
- **PELT (Pruned Exact Linear Time)** change-point detection (via the `ruptures` library), run on the full sequence of block-distribution vectors with an L2 cost model. Unlike pairwise adjacent-block JSD, PELT finds a globally optimal set of breakpoints that best explain shifts in the *whole* distribution sequence, subject to a penalty term controlling sensitivity (lower penalty → more, finer-grained change points; higher penalty → fewer, coarser ones).

Each detected change point is mapped back from its block index to the underlying `chronological_index`, so it can be drawn directly on the chunk-level scatter (§8.2) as a vertical marker.

### 8.5 Interpreting a change point

For every detected change point, in addition to its position, two further quantities characterize *what* changed:

- A **magnitude score** — the L2 norm of the difference between the composition vector just before and just after the change point (how much the overall composition moved).
- **Per-cluster contributions** — the signed change in each cluster's proportion (`distribution_after − distribution_before`), ranked by absolute magnitude. This identifies, for a given change point, which specific clusters grew and which shrank (e.g. "Cluster 3 dropped by 0.31, Cluster 7 rose by 0.22"), giving the change point a stylistic/thematic reading rather than just a location.

### 8.6 Segments between change points

Consecutive change points partition the chronological axis into **segments**. For each segment, a composition table records which source texts (filenames) fall within it, what percentage of the segment each text occupies, and what percentage of each text's own chunks fall inside that segment (i.e. whether the segment contains a text in full, or only overlaps part of it). This grounds each detected stylistic segment back in the actual books/corpora that compose it.

As a secondary, explicitly *non-chronological* analysis, each segment's chunks are also averaged into a single segment-level embedding, and cosine similarity between all segment embeddings is computed to find each segment's nearest neighbors elsewhere in the corpus — surfacing cases where two chronologically distant segments are nonetheless stylistically similar (recurrence of a style/mode across different periods, rather than a one-way linear drift).

### 8.7 Implementation status

The maintained implementation of §8.1–§8.6 is `cluster_visualization_report.py` (flags `--show-change-points`, `--cpd-block-size`, `--cpd-penalty`, `--show-nearest-neighbors`), which produces the combined chunk-scatter + change points + segment tables in one report. A separate standalone script, `change_point_detection.py`, implements the same JSD/PELT core (block signal construction and PELT detection are self-contained) but its visualization layer imports helper functions from a `local_density_visualization.py` module that no longer exists in the repository, so the standalone script currently fails at import time. It should be treated as superseded by `cluster_visualization_report.py` unless/until it is repaired.

## 9. Reproducibility / tooling notes

- All pipeline runs are logged via ClearML (project `MIDRASH`): configuration, tables, plots, and artifacts (including the source script itself) are versioned per run.
- `random_state=42` is fixed throughout for K-Means and MDS.
- Primary data/model paths point to a shared Google Drive location; local `Results/`/`Inputs/` folders are used for repo-relative fallbacks.

## Open items / TODO before this is paper-ready

- [ ] Document `hierarchy` construction methodology and cite its source.
- [ ] Document FastText model provenance (training corpus, dimensions, hyperparameters).
- [ ] Justify/cite the 50-word chunk size choice (vs. alternatives).
- [ ] Report and justify the final choice of k (via the elbow/silhouette sweep) for the reported results.
- [ ] Decide whether smoothed or unsmoothed clustering is the one reported in the paper, and justify.
- [ ] Precisely define each of the six "Experiment" sub-corpora for a Data section (what's included/excluded and why).
- [ ] Justify/report the block size used to aggregate cluster-composition vectors for change-point detection (§8.3), and the PELT penalty value used for the reported results (§8.4) — currently a tunable default, not a chosen/justified value.
- [ ] Decide whether the paper reports JSD peaks, PELT breakpoints, or both, and how discrepancies between the two (if any) are reconciled/discussed.
- [ ] Fix or retire `change_point_detection.py` (broken import of `local_density_visualization.py`) so there is one unambiguous implementation to cite/reproduce from.
