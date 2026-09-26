"""Export n-gram frequency tables per cluster from an existing clustered CSV.

Output: one Excel file with 3 sheets (מילים, זוגות, שלשות).
Each sheet has one column per cluster with the top 100 n-grams.

Usage:
    python export_ngrams.py
    python export_ngrams.py --input-csv "G:/My Drive/Haifa/MIDRASH/Results/rabani_unsmoothed_table.csv"
"""

import argparse
import os
from collections import Counter

import pandas as pd
from clearml import Task

TOP_N = 100


def export_ngrams(input_csv: str, output_path: str) -> None:
    task = Task.init(project_name="MIDRASH", task_name="ngram_export")
    task.upload_artifact(name="Source: chunk_clustering_MAIN.py", artifact_object="chunk_clustering_MAIN.py")
    task.upload_artifact(name="Source: export_ngrams.py", artifact_object="export_ngrams.py")

    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)

    required = {"chunk", "scluster"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}. Found: {list(df.columns)}")

    df["scluster"] = pd.to_numeric(df["scluster"], errors="coerce")
    df = df.dropna(subset=["scluster"]).copy()
    df["scluster"] = df["scluster"].astype(int)
    cluster_ids = sorted(df["scluster"].unique())

    config = {
        "input_csv": input_csv,
        "output_path": output_path,
        "clustering_algorithm": "KMeans",
        "num_clusters": len(cluster_ids),
        "chunk_size_words": 50,
        "apply_smoothing": False,
        "total_chunks": len(df),
        "top_n": TOP_N,
    }
    task.connect(config)
    print(f"Experiment config: {config}")

    uni_cols = {}
    bi_cols = {}
    tri_cols = {}

    for cluster_id in cluster_ids:
        cluster_chunks = df[df["scluster"] == cluster_id]["chunk"]

        unigrams: Counter = Counter()
        bigrams: Counter = Counter()
        trigrams: Counter = Counter()

        for chunk in cluster_chunks:
            tokens = str(chunk).split()
            unigrams.update(tokens)
            bigrams.update(zip(tokens, tokens[1:]))
            trigrams.update(zip(tokens, tokens[1:], tokens[2:]))

        col = f"Cluster {cluster_id}"
        uni_cols[col] = [w for w, _ in unigrams.most_common(TOP_N)]
        bi_cols[col] = [" ".join(k) for k, _ in bigrams.most_common(TOP_N)]
        tri_cols[col] = [" ".join(k) for k, _ in trigrams.most_common(TOP_N)]

        print(f"  processed cluster {cluster_id}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame(uni_cols).to_excel(writer, sheet_name="מילים", index=True)
        pd.DataFrame(bi_cols).to_excel(writer, sheet_name="זוגות", index=True)
        pd.DataFrame(tri_cols).to_excel(writer, sheet_name="שלשות", index=True)

    task.upload_artifact(name="NGrams All Clusters", artifact_object=output_path)
    print(f"\nSaved → {output_path}")
    task.flush(wait_for_uploads=True)
    task.close()


def main():
    parser = argparse.ArgumentParser(description="Export n-gram frequencies per cluster.")
    parser.add_argument(
        "--input-csv",
        default=r"G:\My Drive\Haifa\MIDRASH\Results\rabani_unsmoothed_table.csv",
    )
    parser.add_argument(
        "--output-path",
        default=r"G:\My Drive\Haifa\MIDRASH\Results\ngrams\all_clusters_ngrams.xlsx",
    )
    args = parser.parse_args()
    export_ngrams(args.input_csv, args.output_path)


if __name__ == "__main__":
    main()
