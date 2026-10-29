"""Devarim Rabbah 11:5 - which clustering parameters show the known stylistic shift most clearly.

Prior research: from 11:5 to the end, printed Devarim Rabbah shifts style and is closer to
late aggadic midrash. In act_jsonl_v7 that is the text unit `11_5`, which starts at word
30,603 of 33,194: only the last 7.8% of the work follows it.

The reference corpus is the list agreed on 2026-10-26 (GROUPS below). For every combination
of chunk size, k, smoothing, whether Devarim Rabbah takes part in fitting the clusters
("included") or is assigned to clusters fitted without it ("projected"), and whether the
genres are fitted as they are or at equal size ("natural" / "balanced"), the script clusters
the corpus and reads Devarim Rabbah's sequence of clusters:

- split statistic: chi-square of the before/after cluster table at each possible split; where
  the best split falls, and how 11_5 ranks among all splits;
- p_shift: how often a circularly shifted label sequence (same clusters, same autocorrelation,
  boundary moved) scores at 11_5 as high as the real one - the significance of the split;
- late shift: whether the chunks after 11_5 sit in clusters that lean towards late midrash
  more than the chunks before it do (cluster lift of the "late" genre), with the same p;
- PELT, the app's change-point method, on Devarim Rabbah alone, over block sizes and penalties:
  every change point it returns, and the one nearest 11_5.

Smoothing here is within each work (the app smooths across the whole selection, which mixes
the end of one work with the start of the next).

    python experiments/devarim_rabbah/grid.py            # full grid, ~30-60 min on 4 cores
    python experiments/devarim_rabbah/grid.py --quick    # a few configurations, to check
"""
import argparse
import itertools
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import ruptures as rpt
from scipy.ndimage import convolve1d
from sklearn.cluster import KMeans

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "webapp"))
from explorer import store   # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "results"
STORE = "v7"
TARGET = "דברים רבה"
BOUNDARY_UNIT = "11_5"

GROUPS = {
    "classic": {"titles": ["בראשית רבה", "ויקרא רבה", "שיר השירים רבה", "פסיקתא דרב כהנא", "רות רבה", "איכה רבה"]},
    "late": {"category": "מדרש אגדה - מאוחר", "except": ["מדרש תהילים", "פסיקתא רבתי", "אסתר רבה"]},
    "tannaitic": {"category": "מדרשי התנאים", "except": ["סדר עולם רבה", "מגילת תענית"]},
    "mishnah_tosefta": {"category": "משנה-תוספתא", "except": []},
    "bavli": {"category": "תלמוד בבלי", "except": ["מסכת עבדים", "מסכת דרך ארץ רבה", "מסכת דרך ארץ זוטא"]},
    "yerushalmi": {"category": "תלמוד ירושלמי", "except": []},
}
SIZES = [25, 50, 75, 100]
KS = [4, 6, 8, 10, 12, 15, 20]
SMOOTHING = [False, True]
MODES = ["included", "projected"]
BALANCE = ["natural", "balanced"]
PELT_BLOCKS = [1, 2, 5, 10]
PELT_PENALTIES = [0.2, 0.5, 1.0, 2.0]
WEIGHTS = np.array([0.2, 0.5, 0.6, 0.7, 0.8, 1, 0.8, 0.7, 0.6, 0.5, 0.2])   # the app's smoothing kernel


def corpus_files():
    catalog = pd.read_excel(ROOT / "Inputs" / "corpus_catalog.xlsx", dtype=str)
    catalog = catalog[catalog["in_experiments"] == "True"]
    groups = {}
    for name, rule in GROUPS.items():
        if "titles" in rule:
            rows = catalog[catalog["he_title"].isin(rule["titles"])]
            assert len(rows) == len(rule["titles"]), name
        else:
            rows = catalog[(catalog["scholarly_category"] == rule["category"])
                           & ~catalog["he_title"].isin(rule["except"])]
        for f in rows["file"]:
            groups[f] = name
    target = catalog.loc[catalog["he_title"] == TARGET, "file"].iloc[0]
    return groups, target


def boundary_word(target):
    """The word (0-based, in the order the pipeline reads them) where unit 11_5 begins."""
    prefix = target[: -len(".jsonl")].replace("-", "_") + "_"
    words = 0
    for line in open(ROOT / "corpus_v7" / target, encoding="utf-8"):
        row = json.loads(line)
        if row["location"][len(prefix):] == BOUNDARY_UNIT:
            return words
        words += len(row.get("sentence", "").split())
    raise ValueError("unit not found")


def smooth_within(vecs, files):
    out = np.empty_like(vecs)
    for f in pd.unique(files):
        idx = np.flatnonzero(files == f)
        # Zero-padded like the app's np.convolve(mode="same"), but keeping the work's length
        # when it has fewer chunks than the kernel.
        out[idx] = convolve1d(vecs[idx], WEIGHTS, axis=0, mode="constant", cval=0.0) / WEIGHTS.sum()
    return out


def chi2_split(onehot_cum, total, t, n):
    """Chi-square of the 2 x K table: clusters before chunk t against after it."""
    before = onehot_cum[t - 1]
    after = total - before
    table = np.vstack([before, after])
    keep = table.sum(axis=0) > 0
    table = table[:, keep]
    expected = table.sum(axis=1, keepdims=True) * table.sum(axis=0, keepdims=True) / n
    return float(((table - expected) ** 2 / expected).sum())


def split_profile(labels, k, min_seg):
    n = len(labels)
    onehot = np.eye(k)[labels]
    cum = onehot.cumsum(axis=0)
    total = cum[-1]
    ts = np.arange(min_seg, n - min_seg + 1)
    return ts, np.array([chi2_split(cum, total, t, n) for t in ts])


def pelt(labels, k, block, penalty):
    """The app's method: cluster proportions in blocks of `block` chunks, PELT with an l2 cost."""
    n_blocks = len(labels) // block
    if n_blocks < 4:
        return []
    dist = np.array([np.bincount(labels[i * block:(i + 1) * block], minlength=k) / block
                     for i in range(n_blocks)])
    cps = rpt.Pelt(model="l2", min_size=2, jump=1).fit(dist).predict(pen=penalty)
    return [c * block for c in cps if c < n_blocks]


def run_one(df, groups, target, b_word, size, k, smoothing, mode, balance, rng_seed=0, km_seed=42):
    files = df["filename"].to_numpy()
    vecs = np.vstack(df["vec"].to_numpy()).astype(np.float32)
    if smoothing:
        vecs = smooth_within(vecs, files)
    is_target = files == target
    genre = np.array([groups.get(f, "target") for f in files])

    fit_mask = np.ones(len(df), bool) if mode == "included" else ~is_target
    if balance == "balanced":
        # Equal chunks per genre in the fit; the target, when included, enters whole.
        rng = np.random.default_rng(rng_seed)
        sizes = {g: int(((genre == g) & fit_mask).sum()) for g in GROUPS}
        m = min(sizes.values())
        chosen = [rng.choice(np.flatnonzero((genre == g) & fit_mask), m, replace=False) for g in GROUPS]
        if mode == "included":
            chosen.append(np.flatnonzero(is_target))
        fit_idx = np.concatenate(chosen)
    else:
        fit_idx = np.flatnonzero(fit_mask)

    km = KMeans(n_clusters=k, random_state=km_seed, n_init=3).fit(vecs[fit_idx])
    labels = km.predict(vecs)

    # How much each cluster leans towards late midrash, read off the reference chunks
    # the clusters were fitted on (the balanced sample, or all of them).
    ref = fit_idx[genre[fit_idx] != "target"]
    late_share = (genre[ref] == "late").mean()
    lift = np.zeros(k)
    for c in range(k):
        members = ref[labels[ref] == c]
        lift[c] = ((genre[members] == "late").mean() / late_share) if len(members) else 0.0

    seq = labels[is_target]
    n = len(seq)
    b = int(round(b_word / size))
    min_seg = max(5, int(0.02 * n))
    ts, stats = split_profile(seq, k, min_seg)
    observed = stats[ts == b][0]
    best_t = int(ts[stats.argmax()])
    rank = int((stats > observed).sum()) + 1

    late = lift[seq]
    late_delta = late[b:].mean() - late[:b].mean()
    # Circular shifts keep the clusters and their autocorrelation but move the boundary.
    shifted_stats, shifted_late = [], []
    onehot = np.eye(k)
    for r in range(1, n):
        s = np.roll(seq, r)
        cum = onehot[s].cumsum(axis=0)
        shifted_stats.append(chi2_split(cum, cum[-1], b, n))
        ls = lift[s]
        shifted_late.append(ls[b:].mean() - ls[:b].mean())
    p_shift = (1 + sum(x >= observed for x in shifted_stats)) / n
    p_late = (1 + sum(x >= late_delta for x in shifted_late)) / n

    pelt_rows = []
    for block, pen in itertools.product(PELT_BLOCKS, PELT_PENALTIES):
        cps = pelt(seq, k, block, pen)
        nearest = min(cps, key=lambda c: abs(c - b)) if cps else None
        pelt_rows.append({"block": block, "penalty": pen, "n_cps": len(cps),
                          "nearest_cp_words": None if nearest is None else abs(nearest - b) * size,
                          "cps_words": [c * size for c in cps]})

    return {
        "chunk_size": size, "k": k, "smoothing": smoothing, "mode": mode, "balance": balance,
        "n_target_chunks": n, "boundary_chunk": b,
        "chi2_at_boundary": round(observed, 2), "boundary_rank": rank, "splits": len(ts),
        "best_split_word": best_t * size, "best_split_off_words": (best_t - b) * size,
        "p_shift": round(p_shift, 4),
        "late_delta": round(float(late_delta), 3), "p_late": round(p_late, 4),
        "late_lift_after": round(float(late[b:].mean()), 3), "late_lift_before": round(float(late[:b].mean()), 3),
    }, pelt_rows, seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    groups, target = corpus_files()
    b_word = boundary_word(target)
    print("corpus files:", len(groups), "| target:", target, "| 11_5 starts at word", b_word, flush=True)

    sizes, ks = (SIZES, KS) if not args.quick else ([50], [8])
    rows, pelt_all, sequences = [], [], {}
    for size in sizes:
        t0 = time.time()
        df = store.load_selection(list(groups) + [target], size, STORE)
        print(f"size {size}: {len(df):,} chunks loaded in {time.time() - t0:.0f}s", flush=True)
        for k, smoothing, mode, balance in itertools.product(ks, SMOOTHING, MODES, BALANCE):
            t1 = time.time()
            row, pelt_rows, seq = run_one(df, groups, target, b_word, size, k, smoothing, mode, balance)
            rows.append(row)
            key = f"s{size}_k{k}_{'sm' if smoothing else 'raw'}_{mode}_{balance}"
            sequences[key] = seq.tolist()
            for p in pelt_rows:
                pelt_all.append({**{c: row[c] for c in ("chunk_size", "k", "smoothing", "mode", "balance")}, **p})
            print(f"  {key:38} rank {row['boundary_rank']:>3}/{row['splits']}  p_shift {row['p_shift']:.4f}  "
                  f"best split {row['best_split_off_words']:+6d} words  late {row['late_delta']:+.3f} "
                  f"(p {row['p_late']:.3f})  {time.time() - t1:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "grid.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(pelt_all).to_csv(OUT / "pelt.csv", index=False, encoding="utf-8-sig")
        (OUT / "sequences.json").write_text(json.dumps(sequences), encoding="utf-8")
    print("done:", OUT)


if __name__ == "__main__":
    main()
