"""Generate the embedding store: every source file, at each chunk size.

Run this where the FastText model and the Sefaria dump are reachable. The web app
never loads the model; it reads only what this writes.

    python webapp/build_store.py --sizes 25 50 75 100
    python webapp/build_store.py --sizes 50 --corpora Midrash Tanaitic
    python webapp/build_store.py --sizes 50 --model /path/to/fasttext_rabbinic_dim100.bin

A file already present in the store for a given size is skipped, so an interrupted
run resumes and a newly added size costs only its own pass.
"""
import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import pipeline_core                     # noqa: E402
from explorer.store import STORE_DIR, safe_name   # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", required=True)
    ap.add_argument("--corpora", nargs="*", default=None,
                    help="restrict to these top-level corpora, e.g. Midrash Tanaitic")
    ap.add_argument("--files-from", default=None,
                    help="a text file of source filenames, one per line; the exact set to build")
    ap.add_argument("--model", default=None, help="FastText .bin; defaults to pipeline_core.MODEL_PATH")
    ap.add_argument("--label", required=True,
                    help="store subdirectory for this model, e.g. rabbinic or zohar")
    ap.add_argument("--title", default=None, help="how the app should name this model")
    ap.add_argument("--note", default="", help="one line of provenance shown in the app")
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    metadata = pipeline_core.load_metadata()
    metadata = metadata[metadata["filename"].notna()]
    if args.corpora:
        keep = metadata["filename"].astype(str).str.split("-").str[0].isin(args.corpora)
        metadata = metadata[keep]
    if args.files_from:
        wanted = {l.strip() for l in open(args.files_from, encoding="utf-8") if l.strip()}
        metadata = metadata[metadata["filename"].astype(str).isin(wanted)]
        missing = wanted - set(metadata["filename"].astype(str))
        if missing:
            print("not in the metadata table: %d (%s)" % (len(missing), sorted(missing)[0][:60]))
    print("files in scope: %d" % len(metadata), flush=True)

    print("loading FastText ...", flush=True)
    model = pipeline_core.load_fasttext_model(args.model)
    dim = model.get_dimension()
    print("  dim %d" % dim, flush=True)

    model_dir = STORE_DIR / args.label
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.json").write_text(json.dumps({
        "label": args.title or args.label,
        "note": args.note,
        "model_file": str(args.model or pipeline_core.MODEL_PATH),
        "dim": dim,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    for size in args.sizes:
        out_dir = model_dir / str(size)
        out_dir.mkdir(parents=True, exist_ok=True)
        index_path = out_dir / "index.json"
        index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}

        t0, done, chunks_total = time.time(), 0, 0
        for _, row in metadata.iterrows():
            filename = str(row["filename"])
            target = out_dir / (safe_name(filename) + ".npz")
            if filename in index and target.exists():
                continue

            one = pd.DataFrame([row])
            chunks = pipeline_core.build_chunks(one, data_dir=args.data_dir, chunk_size=size)
            if chunks.empty:
                continue
            vecs = np.vstack([model.get_sentence_vector(c) for c in chunks["chunk"]]).astype(np.float32)
            np.savez_compressed(target,
                                vec=vecs,
                                chunk=chunks["chunk"].to_numpy(dtype=object),
                                chunk_number=chunks["chunk_number"].to_numpy())
            index[filename] = {
                "n_chunks": int(len(chunks)),
                "words": int(sum(len(c.split()) for c in chunks["chunk"])),
                "hierarchy": str(row.get("hierarchy", "")),
                "dim": dim,
            }
            index_path.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
            done += 1
            chunks_total += len(chunks)
            if done % 25 == 0:
                print("  size %d: %d files, %s chunks, %.0fs"
                      % (size, done, format(chunks_total, ","), time.time() - t0), flush=True)

        size_mb = sum(p.stat().st_size for p in out_dir.glob("*.npz")) / 1e6
        print("size %d: %d files in the store, %.0f MB, %.0fs"
              % (size, len(index), size_mb, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
