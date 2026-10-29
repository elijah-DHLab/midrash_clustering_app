"""Significant change points, on the chunk vectors alone, for every work of the task corpus
(the agreed reference corpus and Devarim Rabbah), for the visualization.

Each work is segmented on its own: its 25-word chunk vectors are standardised within the work
and split recursively wherever a split survives the block-permutation test (segment.py: p<0.01).
Nothing about genres or known boundaries is used. For each work the output keeps its change
points (word position, the v7 unit, the first words after the point) and the starts of its
top-level units (parasha, chapter, daf) for drawing.

    python experiments/devarim_rabbah/segment_corpus.py
"""
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grid      # noqa: E402
import segment   # noqa: E402

SIZE = 25
OUT = HERE / "results" / "corpus_changepoints.json"


def main():
    groups, target = grid.corpus_files()
    names = grid.store.titles_for(grid.STORE)
    works = list(groups) + [target]
    done = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    t0 = time.time()
    for i, f in enumerate(works, 1):
        if f in done:
            continue
        df = grid.store.load_selection([f], SIZE, grid.STORE)
        x = segment.standardise(np.vstack(df["vec"].to_numpy()).astype(np.float32))
        cps = segment.segment(x, SIZE)
        units = df["start_unit"].astype(str).tolist()
        chunks = df["chunk"].astype(str).tolist()
        tops, last = [], None
        for j, u in enumerate(units):
            top = u.split("_")[0]
            if top != last:
                tops.append([j * SIZE, top])
                last = top
        done[f] = {
            "title": names.get(f, {}).get("he") or f,
            "genre": groups.get(f, "target"),
            "words": len(units) * SIZE,
            "units": tops,
            "cps": [{"word": c * SIZE, "unit": units[c], "after": " ".join(chunks[c].split()[:14]),
                     "before": " ".join(chunks[c - 1].split()[-14:])} for c, _, _ in cps],
        }
        if i % 10 == 0 or len(units) > 2000:
            OUT.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
            print(f"{i}/{len(works)} works, {time.time() - t0:.0f}s - last: {done[f]['title']} "
                  f"({len(units)} chunks, {len(cps)} change points)", flush=True)
    OUT.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
    print("done", len(done), "works", flush=True)


if __name__ == "__main__":
    main()
