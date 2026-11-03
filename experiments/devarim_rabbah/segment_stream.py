"""Change points on the whole task corpus as ONE stream, in the chronological (hierarchy) order
the clustering uses: does the algorithm, told nothing about where works begin, find the
boundaries between works by itself - and seams inside works?

The 219 works are concatenated in hierarchy order (store.load_selection's order), the chunk
vectors standardised over the whole stream, and segment.segment() run on it: recursive
binary split, each split kept only if it beats every one of R block-shuffled copies (p<0.01).
Afterwards each change point is located: which work, which v7 unit, and whether it falls
within TOL words of a boundary between works.

    python experiments/devarim_rabbah/segment_stream.py [--size 50]
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grid      # noqa: E402
import segment   # noqa: E402

TOL = 300


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=50)
    ap.add_argument("--without-target", action="store_true", help="leave Devarim Rabbah out of the stream")
    args = ap.parse_args()
    size = args.size
    groups, target = grid.corpus_files()
    names = grid.store.titles_for(grid.STORE)
    df = grid.store.load_selection(list(groups) + ([] if args.without_target else [target]), size, grid.STORE)
    files = df["filename"].to_numpy()
    x = segment.standardise(np.vstack(df["vec"].to_numpy()).astype(np.float32))
    print(f"stream: {len(df):,} chunks of {size} words, {len(set(files))} works", flush=True)

    # Where each work starts in the stream, in chunks.
    starts = [0] + [i for i in range(1, len(files)) if files[i] != files[i - 1]]
    order = [files[i] for i in starts]
    t0 = time.time()
    cps = segment.segment(x, size)
    print(f"{len(cps)} change points in {time.time() - t0:.0f}s", flush=True)

    units = df["start_unit"].astype(str).to_numpy()
    chunk_no = df["chunk_number"].to_numpy()
    chunks = df["chunk"].astype(str).to_numpy()
    tol = TOL // size
    out_cps = []
    for c, p, depth, strength in cps:
        nearest_start = min(starts, key=lambda s: abs(s - c))
        out_cps.append({
            "i": int(c), "file": files[c], "word_in_work": int((chunk_no[c] - 1) * size),
            "unit": units[c], "depth": depth, "strength": round(float(strength), 2),
            "work_boundary": abs(nearest_start - c) <= tol and nearest_start != 0,
            "after": " ".join(chunks[c].split()[:14]), "before": " ".join(chunks[c - 1].split()[-14:]),
        })
    detected = {s for s in starts[1:] if any(abs(s - c) <= tol for c, *_ in cps)}
    result = {
        "size": size, "chunks": len(df),
        "works": [{"file": f, "title": names.get(f, {}).get("he") or f, "genre": groups.get(f, "target"),
                   "start": int(s), "found_start": s in detected} for f, s in zip(order, starts)],
        "cps": out_cps,
    }
    tag = "_without_devarim" if args.without_target else ""
    (HERE / "results" / f"stream_changepoints_s{size}{tag}.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    on = sum(c["work_boundary"] for c in out_cps)
    print(f"work boundaries found: {len(detected)} of {len(starts) - 1}; "
          f"change points on a work boundary: {on} of {len(out_cps)}", flush=True)


if __name__ == "__main__":
    main()
