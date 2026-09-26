"""Fill the store from the pickles the CLI pipeline already cached.

The cached frames hold chunks that were embedded with whatever model was in use
at the time, so a store built this way is only as good as that model; it exists
to give the app real data to show before the new embeddings are generated.

    python webapp/import_cache.py ../cache/embed_789aed26c9ee7f37f8ae2182.pkl --chunk-size 50
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from explorer.store import STORE_DIR, safe_name   # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pickle")
    ap.add_argument("--chunk-size", type=int, required=True)
    args = ap.parse_args()

    df = pd.read_pickle(args.pickle)
    out_dir = STORE_DIR / str(args.chunk_size)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}

    for filename, rows in df.groupby("filename"):
        rows = rows.sort_values("chunk_number")
        vecs = np.vstack(rows["vec"].to_numpy()).astype(np.float32)
        np.savez_compressed(out_dir / (safe_name(str(filename)) + ".npz"),
                            vec=vecs,
                            chunk=rows["chunk"].to_numpy(dtype=object),
                            chunk_number=rows["chunk_number"].to_numpy())
        index[str(filename)] = {
            "n_chunks": int(len(rows)),
            "words": int(rows["chunk"].astype(str).str.split().str.len().sum()),
            "hierarchy": str(rows["hierarchy"].iloc[0]),
            "dim": int(vecs.shape[1]),
            "source": "cache import: %s" % pathlib.Path(args.pickle).name,
        }
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    size_mb = sum(p.stat().st_size for p in out_dir.glob("*.npz")) / 1e6
    print("chunk size %d: %d files, %s chunks, %.0f MB -> %s"
          % (args.chunk_size, len(index), format(len(df), ","), size_mb, out_dir))


if __name__ == "__main__":
    main()
