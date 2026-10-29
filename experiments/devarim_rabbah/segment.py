"""Change points decided by significance, not by a count given in advance.

Recursive binary segmentation with a permutation test at every split (in the spirit of
circular binary segmentation, Olshen et al. 2004): in a stretch of the work, find the split
that most separates the mean signal before it from the mean after it; compare its strength
with the strongest split in R block-shuffled copies of the same stretch (blocks of BLOCK chunks
keep the short-range dependence between neighbouring chunks, while the shuffle destroys any
real change); keep the split only if p < ALPHA, then search each half the same way. The number
of change points is whatever survives.

Three signals for Devarim Rabbah, none told where 11:5 is:
- vectors:  the chunk vectors themselves, standardised (no clustering, no genres);
- profile:  each chunk's cosine similarity to the centroid of each of the six reference genres
            (a continuous genre profile - the middle way between vectors and clusters);
- labels:   the chunk's cluster (one-hot), for the configurations that did best in detect.py.

Each surviving change point is reported with its p-value, the v7 unit it falls in, whether it
sits on a parasha boundary, and the late-midrash lean of the stretch before and after it.

    python experiments/devarim_rabbah/segment.py
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import detect   # noqa: E402
import grid     # noqa: E402

ALPHA, R, BLOCK_WORDS, MIN_WORDS = 0.01, 199, 250, 500
OUT = HERE / "results"


def best_split(x, min_size):
    """The split maximising the between-segment sum of squares n1*n2/n*|mean1-mean2|^2."""
    n = len(x)
    if n < 2 * min_size:
        return None, 0.0
    cum = np.cumsum(x, axis=0)
    total = cum[-1]
    t = np.arange(min_size, n - min_size + 1)
    left = cum[t - 1] / t[:, None]
    right = (total - cum[t - 1]) / (n - t)[:, None]
    stat = t * (n - t) / n * ((left - right) ** 2).sum(axis=1)
    i = int(stat.argmax())
    return int(t[i]), float(stat[i])


def split_p(x, obs, min_size, block, rng):
    n = len(x)
    blocks = [x[i:i + block] for i in range(0, n, block)]
    hits = 0
    for _ in range(R):
        order = rng.permutation(len(blocks))
        _, s = best_split(np.concatenate([blocks[j] for j in order]), min_size)
        hits += s >= obs
    return (1 + hits) / (R + 1)


def segment(x, size, seed=0):
    """All significant change points of x, as (index, p, depth), found top-down."""
    rng = np.random.default_rng(seed)
    min_size, block = max(2, MIN_WORDS // size), max(2, BLOCK_WORDS // size)
    found, stack = [], [(0, len(x), 0)]
    while stack:
        a, b, depth = stack.pop()
        t, obs = best_split(x[a:b], min_size)
        if t is None:
            continue
        p = split_p(x[a:b], obs, min_size, block, rng)
        if p < ALPHA:
            found.append((a + t, p, depth))
            stack += [(a, a + t, depth + 1), (a + t, b, depth + 1)]
    return sorted(found)


def standardise(x):
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-9)


def main():
    groups, target = grid.corpus_files()
    prefix = target[: -len(".jsonl")].replace("-", "_") + "_"
    rows, starts, w, seen = [], [], 0, set()
    for line in open(grid.ROOT / "corpus_v7" / target, encoding="utf-8"):
        r = json.loads(line)
        unit = r["location"][len(prefix):]
        if unit.split("_")[0] not in seen:
            seen.add(unit.split("_")[0])
            starts.append(w)
        rows.append((w, unit))
        w += len(r["sentence"].split())
    internal = starts[1:]
    unit_at = lambda word: [u for s, u in rows if s <= word][-1]   # noqa: E731
    b115 = grid.boundary_word(target)

    out = []
    for size in (25, 50):
        df = grid.store.load_selection(list(groups) + [target], size, grid.STORE)
        files = df["filename"].to_numpy()
        vecs = np.vstack(df["vec"].to_numpy()).astype(np.float32)
        genre = np.array([groups.get(f, "target") for f in files])
        is_target = files == target
        tv = vecs[is_target]

        # Late lean per chunk, for the direction (as in profile.py: k=15, projected, balanced).
        seq_ref, lift_ref, _ = detect.cluster(df, groups, target, size, 15, False, "projected", "balanced")
        late = lift_ref[seq_ref]

        unit_vec = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
        cents = np.vstack([unit_vec[genre == g].mean(axis=0) for g in grid.GROUPS])
        cents /= np.linalg.norm(cents, axis=1, keepdims=True)
        signals = {"vectors": standardise(tv), "profile": standardise(unit_vec[is_target] @ cents.T)}
        for k, mode in ((15, "projected"), (20, "included")):
            seq, _, _ = detect.cluster(df, groups, target, size, k, False, mode, "balanced")
            signals[f"labels k{k} {mode}"] = np.eye(k)[seq]

        for name, x in signals.items():
            cps = segment(x, size)
            edges = [0] + [c for c, _, _ in cps] + [len(x)]
            print(f"\nsize {size} | {name}: {len(cps)} significant change points", flush=True)
            for (c, p, depth) in cps:
                word = c * size
                i = edges.index(c)
                before, after = late[edges[i - 1]:c].mean(), late[c:edges[i + 1]].mean()
                on_par = min(abs(word - s) for s in internal) <= 300
                row = {"chunk_size": size, "signal": name, "word": word, "unit": unit_at(word), "p": round(p, 3),
                       "depth": depth, "parasha_boundary": on_par, "near_11_5": abs(word - b115) <= 600,
                       "late_before": round(float(before), 2), "late_after": round(float(after), 2)}
                out.append(row)
                print(f"   word {word:6d}  {row['unit']:7}  p={p:.3f}  depth {depth}  "
                      f"{'parasha boundary ' if on_par else ''}{'<-- 11:5 area ' if row['near_11_5'] else ''}"
                      f"late {before:.2f} -> {after:.2f}", flush=True)
    pd.DataFrame(out).to_csv(OUT / "segment.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
