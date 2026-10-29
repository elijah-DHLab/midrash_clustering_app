"""Unsupervised change-point detection in Devarim Rabbah: does an algorithm that is not told
where to look find the known shift at 11:5 by itself, and is the stretch after what it finds
closer to late midrash?

grid.py tested the given point. Here the point is never given. Binary segmentation finds
change points in order of prominence - the first is the strongest change in the work, the
second the next, up to MAX_BKPS - on two kinds of signal:

- "labels": Devarim Rabbah's sequence of clusters (one-hot), for each clustering
  configuration (chunk size, k, smoothing, included/projected, natural/balanced);
- "vectors": the chunk vectors themselves, standardised, with no clustering at all - the
  detection then owes nothing to genres, the catalog or the reference corpus.

Only afterwards is the result held against the literature: at which step does a change point
land within TOL words of 11:5 (and of 10:1, the split grid.py kept finding)? A hit at step 1
means the algorithm's strongest change is the known one. Chance level: the probability that m
change points placed at random would put one within TOL of a given word.
Then the direction: each segment the algorithm found gets its mean lean towards late midrash
(cluster lift, as in profile.py).

    python experiments/devarim_rabbah/detect.py
"""
import itertools
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import ruptures as rpt
from sklearn.cluster import KMeans

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grid   # noqa: E402

OUT = HERE / "results"
SIZES = [25, 50, 75, 100]
KS = [6, 10, 15, 20]
SMOOTHING = [False, True]
MODES = ["included", "projected"]
BALANCE = ["natural", "balanced"]
MAX_BKPS = 6
TOL = 500          # words: about two or three v7 rows
MIN_SIZE_WORDS = 500


def discovery(cps_by_m, target_word, size):
    """The first step m at which a change point lands within TOL words of target_word."""
    for m, cps in cps_by_m.items():
        if any(abs(c * size - target_word) <= TOL for c in cps):
            return m
    return None


def binseg(signal, model, size):
    min_size = max(2, MIN_SIZE_WORDS // size)
    algo = rpt.Binseg(model=model, min_size=min_size, jump=1).fit(signal)
    out = {}
    for m in range(1, MAX_BKPS + 1):
        try:
            out[m] = [c for c in algo.predict(n_bkps=m) if c < len(signal)]
        except Exception:
            break
    return out


def segment_leans(cps, late, size):
    edges = [0] + sorted(cps) + [len(late)]
    return [{"from_word": a * size, "to_word": b * size, "late_lean": round(float(late[a:b].mean()), 3)}
            for a, b in zip(edges, edges[1:]) if b > a]


def chance(m, length):
    """P(at least one of m uniformly placed points falls within TOL of a given word)."""
    p = min(1.0, 2 * TOL / length)
    return 1 - (1 - p) ** m


def cluster(df, groups, target, size, k, smoothing, mode, balance, seed=42):
    files = df["filename"].to_numpy()
    vecs = np.vstack(df["vec"].to_numpy()).astype(np.float32)
    if smoothing:
        vecs = grid.smooth_within(vecs, files)
    is_target = files == target
    genre = np.array([groups.get(f, "target") for f in files])
    fit_mask = np.ones(len(df), bool) if mode == "included" else ~is_target
    if balance == "balanced":
        rng = np.random.default_rng(0)
        m = min(int(((genre == g) & fit_mask).sum()) for g in grid.GROUPS)
        chosen = [rng.choice(np.flatnonzero((genre == g) & fit_mask), m, replace=False) for g in grid.GROUPS]
        if mode == "included":
            chosen.append(np.flatnonzero(is_target))
        fit_idx = np.concatenate(chosen)
    else:
        fit_idx = np.flatnonzero(fit_mask)
    km = KMeans(n_clusters=k, random_state=seed, n_init=3).fit(vecs[fit_idx])
    labels = km.predict(vecs)
    ref = fit_idx[genre[fit_idx] != "target"]
    late_share = (genre[ref] == "late").mean()
    lift = np.array([((genre[ref][labels[ref] == c] == "late").mean() / late_share)
                     if (labels[ref] == c).any() else 0.0 for c in range(k)])
    return labels[is_target], lift, vecs[is_target]


def main():
    OUT.mkdir(exist_ok=True)
    groups, target = grid.corpus_files()
    b115 = grid.boundary_word(target)
    b101 = 28583
    length = 33194
    rows = []
    for size in SIZES:
        t0 = time.time()
        df = grid.store.load_selection(list(groups) + [target], size, grid.STORE)
        print(f"size {size}: {len(df):,} chunks in {time.time() - t0:.0f}s", flush=True)

        # Vectors alone: no clustering, so no k, mode or balance. The late lean of each
        # segment is read from one reference clustering (k=15, projected, balanced).
        _, lift_ref, _ = cluster(df, groups, target, size, 15, False, "projected", "balanced")
        seq_ref, _, _ = cluster(df, groups, target, size, 15, False, "projected", "balanced")
        late_ref = lift_ref[seq_ref]
        for smoothing in SMOOTHING:
            files = df["filename"].to_numpy()
            vecs = np.vstack(df["vec"].to_numpy()).astype(np.float32)
            if smoothing:
                vecs = grid.smooth_within(vecs, files)
            v = vecs[files == target]
            v = (v - v.mean(axis=0)) / (v.std(axis=0) + 1e-9)
            for model in ("l2", "rbf"):
                cps = binseg(v, model, size)
                rows.append(record("vectors-" + model, size, None, smoothing, None, None,
                                   cps, late_ref, b115, b101, length))
                print(f"  vectors-{model:3} s{size} {'sm' if smoothing else 'raw'}: "
                      f"11_5 at step {rows[-1]['step_11_5']}, 10_1 at step {rows[-1]['step_10_1']}", flush=True)

        for k, smoothing, mode, balance in itertools.product(KS, SMOOTHING, MODES, BALANCE):
            seq, lift, _ = cluster(df, groups, target, size, k, smoothing, mode, balance)
            onehot = np.eye(k)[seq]
            cps = binseg(onehot, "l2", size)
            rows.append(record("labels", size, k, smoothing, mode, balance, cps, lift[seq], b115, b101, length))
            r = rows[-1]
            print(f"  labels s{size} k{k} {'sm' if smoothing else 'raw'} {mode} {balance}: "
                  f"11_5 at step {r['step_11_5']}, 10_1 at step {r['step_10_1']}, "
                  f"first cp {r['cp_order_words'][:1]}", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "detect.csv", index=False, encoding="utf-8-sig")
    print("done", flush=True)


def record(method, size, k, smoothing, mode, balance, cps_by_m, late, b115, b101, length):
    last = cps_by_m.get(max(cps_by_m)) if cps_by_m else []
    # The order in which binary segmentation added its change points = their prominence.
    order, seen = [], set()
    for m in sorted(cps_by_m):
        for c in cps_by_m[m]:
            if c not in seen:
                seen.add(c)
                order.append(c * size)
    s115, s101 = discovery(cps_by_m, b115, size), discovery(cps_by_m, b101, size)
    # The direction: lean of the segment after the change point nearest 11_5, against before it.
    near = min(last, key=lambda c: abs(c * size - b115)) if last else None
    after = before = None
    if near is not None:
        after = round(float(late[near:].mean()), 3)
        before = round(float(late[:near].mean()), 3)
    return {
        "method": method, "chunk_size": size, "k": k, "smoothing": smoothing, "mode": mode, "balance": balance,
        "step_11_5": s115, "step_10_1": s101,
        "chance_at_step": round(chance(s115, length), 3) if s115 else None,
        "cp_order_words": order,
        "nearest_to_11_5_words": None if near is None else near * size - b115,
        "late_lean_before_nearest": before, "late_lean_after_nearest": after,
        "segments_at_6": json.dumps(segment_leans(last, late, size), ensure_ascii=False),
    }


if __name__ == "__main__":
    main()
