"""Shared, reusable pipeline core: metadata loading, chunking, embedding, clustering.

Extracted from chunk_clustering_MAIN.py / num_cluster_optimization.py so the
Streamlit app (and any future entry point) doesn't duplicate this logic a
third time. Chunk size, previously hardcoded at 50 words in both scripts, is
a real parameter here.

Paths are env-var overridable so the same code works locally (defaults match
today's CLI scripts) and, later, on a different machine (e.g. a lab server)
by just setting different env vars.
"""

import bisect
import hashlib
import json
import os

import fasttext
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from cluster_visualization_report import _grouping_label

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))

# Since October 2026 the corpus is Sefaria act_jsonl_v7 (s3://elasticsearch-back-c/jsons/
# sefaria-jsons/act_jsonl_v7/, documented in Inputs/v7_README.html) and the input table is
# Inputs/corpus_catalog.xlsx (Inputs/build_catalog.py). The old corpus and table -
# SefariasData/rabbinic and avg_text_length_per_file.xlsx - are reachable through the env vars.
MODEL_PATH = os.environ.get(
    "MIDRASH_MODEL_PATH", "G:/My Drive/Haifa/MIDRASH/models/fasttext_v7_2026-10.bin"
)
DATA_DIR = os.environ.get("MIDRASH_DATA_DIR", os.path.join(_HERE, "corpus_v7"))
METADATA_PATH = os.environ.get(
    "MIDRASH_METADATA_PATH", os.path.join(_HERE, "Inputs", "corpus_catalog.xlsx")
)
RESULTS_DIR = os.environ.get("MIDRASH_RESULTS_DIR", "Results")
CACHE_DIR = os.environ.get("MIDRASH_CACHE_DIR", "cache")

EXPERIMENT_OPTIONS = {
    "Experiment 1": "Experiment 1: Rabbinic Literature",
    "Experiment 2": "Experiment 2: Midrashic Literature",
    "Experiment 3": "Experiment 3: Tanhuma",
    "Experiment 4": "Experiment 4: Tanhuma",
    "Experiment 5": "Experiment 5: Midrashic Literature w/o anthologies",
    "Experiment 6": "Experiment 6: Early Main Literature (halakhic, amoraic, tanhumah)",
}


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def load_metadata(metadata_path: str = None) -> pd.DataFrame:
    """Load the file metadata table and add level 1/2/3 grouping-label
    columns (Corpus / Corpus-SubCorpus / Corpus-SubCorpus-Book), reusing the
    same convention the rest of the pipeline uses for `grouping_level`, so a
    hierarchical file selector can be built directly from these columns.

    Rows with no `hierarchy` value are dropped: they have no place in the
    corpus-wide chronological order (build_chunks requires it), and none of
    the six preset Experiment columns ever include such a file — about half
    the rows (Halakhah, Liturgy, Kabbalah, most Apocrypha) fall in this
    bucket, so this only matters once files are selectable outside those
    presets, as they are here.
    """
    path = metadata_path or METADATA_PATH
    df = pd.read_excel(path, dtype=str)   # the first sheet: `catalog` in corpus_catalog.xlsx
    if "filename" not in df.columns and "file" in df.columns:
        df = df.rename(columns={"file": "filename"})
    df = df.dropna(subset=["hierarchy"]).reset_index(drop=True)
    df["level1"] = df["filename"].apply(lambda f: _grouping_label(f, 1))
    df["level2"] = df["filename"].apply(lambda f: _grouping_label(f, 2))
    df["level3"] = df["filename"].apply(lambda f: _grouping_label(f, 3))
    return df


def files_for_experiment(metadata: pd.DataFrame, experiment_key: str) -> list:
    """Filenames flagged under the given preset Experiment column (e.g. 'Experiment 1')."""
    column = EXPERIMENT_OPTIONS[experiment_key]
    return metadata.loc[metadata[column] == "1", "filename"].tolist()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def location_prefix(filename: str) -> str:
    """The part of every `location` in a file that only repeats the file's name.

    v7:  Midrash-Aggadic_Midrash-Midrash_Rabbah-Bereishit_Rabbah-Sefaria.jsonl
         -> Midrash_Aggadic_Midrash_Midrash_Rabbah_Bereishit_Rabbah_Sefaria_   (then 52_10)
    old: Apocrypha-Book of Tobit-.json -> Apocrypha_Book_of_Tobit__            (then 1_1)
    """
    stem = filename[: -len(".jsonl")] if filename.endswith(".jsonl") else filename[: -len(".json")]
    return stem.replace("-", "_").replace(" ", "_") + "_"


def _parse_hierarchy_tuple(value):
    return tuple(int(part.split("-")[0]) for part in str(value).split("."))


def build_chunks(
    file_rows: pd.DataFrame, data_dir: str = None, chunk_size: int = 50
) -> pd.DataFrame:
    """Read each selected file's JSONL, concatenate its sentences, and split
    into fixed-size word chunks, sorted into one corpus-wide chronological
    sequence by `hierarchy`.

    `file_rows` must have at least `filename` and `hierarchy` columns (a
    subset of the metadata table, e.g. from load_metadata()).
    """
    resolved_data_dir = data_dir or DATA_DIR

    dataset = file_rows.copy()
    dataset["hierarchy_list"] = dataset["hierarchy"].apply(_parse_hierarchy_tuple)
    dataset = dataset.sort_values(by="hierarchy_list").reset_index(drop=True)

    available = set(os.listdir(resolved_data_dir))

    records = []
    for _, row in dataset.iterrows():
        filename = row["filename"]
        if filename not in available:
            continue
        filepath = os.path.join(resolved_data_dir, filename)

        full_text = []
        # Where each source row starts in full_text, and where that row is in the book.
        row_starts, row_refs, row_segments = [], [], []
        # The text unit a row was cut from, and the row's first word within it. v7 splits a
        # long Sefaria unit into rows whose from_word continues the previous row's to_word;
        # the unit is named by the location of its first row, because in some books (Pesikta
        # Rabbati, the Yalkut, ...) every part's location also gets a part-number suffix.
        row_units, row_from = [], []
        prefix = location_prefix(filename)
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Error in file {filename}: {e}")
                    continue
                words = obj.get("sentence", "").split()
                if not words:
                    continue
                row_starts.append(len(full_text))
                loc = str(obj.get("location", ""))
                row_refs.append(loc[len(prefix):] if loc.startswith(prefix) else loc)
                row_segments.append(obj.get("segment", ""))
                from_word = int(obj.get("from_word") or 1)
                continues = from_word > 1 and row_units
                row_units.append(row_units[-1] if continues else row_refs[-1])
                row_from.append(from_word)
                full_text.extend(words)

        def row_of(word_index):
            return bisect.bisect_right(row_starts, word_index) - 1

        def place(word_index):
            """(row, word in the v7 row, text unit, word in the unit) of one word, all 1-based."""
            r = row_of(word_index)
            offset = word_index - row_starts[r]
            return r, offset + 1, row_units[r], row_from[r] + offset

        def ref_spans(start, end, first, last):
            """'1–23: 1_7 | 24–50: 1_10' - which words of the chunk (1-based) come from which
            location. Consecutive rows with the same location (a v7 row split in two) are one span."""
            spans = []
            for r in range(first, last + 1):
                lo = max(row_starts[r], start) - start + 1
                hi = (min(row_starts[r + 1] - 1, end) if r + 1 < len(row_starts) else end) - start + 1
                if spans and spans[-1][2] == row_refs[r]:
                    spans[-1][1] = hi
                else:
                    spans.append([lo, hi, row_refs[r]])
            return " | ".join(f"{lo}–{hi}: {ref}" for lo, hi, ref in spans)

        for idx, start in enumerate(range(0, len(full_text), chunk_size)):
            end = min(start + chunk_size, len(full_text)) - 1
            first, first_word, first_unit, first_unit_word = place(start)
            last, last_word, last_unit, last_unit_word = place(end)
            records.append(
                {
                    "filename": filename,
                    "hierarchy": row["hierarchy"],
                    "hierarchy_list": row["hierarchy_list"],
                    "chunk": " ".join(full_text[start : end + 1]),
                    "chunk_number": idx + 1,
                    # `location` gives a row's first Sefaria section only (v7 README), so a
                    # chunk is placed by the rows its first and last word come from.
                    "start_ref": row_refs[first],
                    "end_ref": row_refs[last],
                    "ref_spans": ref_spans(start, end, first, last),
                    # Word-exact ends of the chunk: in the v7 rows (segment id and word
                    # within the row), and in Sefaria's text unit (from_word + offset).
                    "start_segment": row_segments[first],
                    "start_segment_word": first_word,
                    "end_segment": row_segments[last],
                    "end_segment_word": last_word,
                    "start_unit": first_unit,
                    "start_unit_word": first_unit_word,
                    "end_unit": last_unit,
                    "end_unit_word": last_unit_word,
                }
            )

    df_chunks = pd.DataFrame(records)
    if df_chunks.empty:
        return df_chunks
    df_chunks = df_chunks.sort_values(
        by=["hierarchy_list", "filename", "chunk_number"]
    ).reset_index(drop=True)
    df_chunks["chronological_index"] = df_chunks.index
    return df_chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def load_fasttext_model(model_path: str = None):
    return fasttext.load_model(model_path or MODEL_PATH)


def embed_chunks(df_chunks: pd.DataFrame, model) -> pd.DataFrame:
    df_chunks = df_chunks.copy()
    df_chunks["vec"] = df_chunks["chunk"].apply(model.get_sentence_vector)
    return df_chunks


# ---------------------------------------------------------------------------
# On-disk embedding cache
# ---------------------------------------------------------------------------
#
# The CLI pipeline (chunk_clustering_MAIN.py) computes embeddings once and
# writes them into a CSV that every downstream script just reads — it never
# recomputes. The Streamlit app's in-memory st.session_state cache only
# lasts for one browser session, so a fresh page load or server restart
# loses it; this disk cache gives the app the same "compute once" behavior
# as the CLI, keyed by the exact (file selection, chunk size) pair.


def _embedding_cache_key(filenames, chunk_size: int) -> str:
    payload = json.dumps({"files": sorted(filenames), "chunk_size": chunk_size})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _embedding_cache_path(filenames, chunk_size: int) -> str:
    key = _embedding_cache_key(filenames, chunk_size)
    return os.path.join(CACHE_DIR, f"embed_{key}.pkl")


def load_cached_embeddings(filenames, chunk_size: int):
    """Return the cached, already-embedded chunks DataFrame for this exact
    (file selection, chunk size), or None if nothing is cached yet."""
    path = _embedding_cache_path(filenames, chunk_size)
    if os.path.exists(path):
        return pd.read_pickle(path)
    return None


def save_cached_embeddings(filenames, chunk_size: int, df_embedded: pd.DataFrame) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _embedding_cache_path(filenames, chunk_size)
    df_embedded.to_pickle(path)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def cluster_chunks(
    df_chunks: pd.DataFrame,
    k: int,
    apply_smoothing: bool = False,
    random_state: int = 42,
) -> pd.DataFrame:
    df_chunks = df_chunks.copy()
    XX = np.stack(df_chunks["vec"].values)

    if apply_smoothing:
        weights = np.array([0.2, 0.5, 0.6, 0.7, 0.8, 1, 0.8, 0.7, 0.6, 0.5, 0.2])
        features = np.array(
            [np.convolve(x, weights, mode="same") / sum(weights) for x in XX.T]
        ).T
    else:
        features = XX

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    df_chunks["scluster"] = kmeans.fit_predict(features)
    return df_chunks


def stringify_vectors(df_chunks: pd.DataFrame) -> pd.DataFrame:
    """Convert the `vec` column from numpy arrays to the string
    representation (`"[0.1, 0.2, ...]"`) the rest of the pipeline expects.

    cluster_visualization_report.py's vector parsing (`_parse_vec_fast`,
    `_parse_vec_string`) was written for `vec` as loaded back from a CSV
    (chunk_clustering_MAIN.py stringifies it before saving); calling those
    functions on live numpy arrays fails, so this bridges the two
    representations. Call this once, right after clustering, before handing
    the DataFrame to any cluster_visualization_report function.
    """
    df_chunks = df_chunks.copy()
    df_chunks["vec"] = df_chunks["vec"].apply(lambda v: v.tolist()).astype(str)
    return df_chunks
