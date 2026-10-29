"""Reading the Devarim Rabbah grid (grid.py): what holds after correcting for 224 tests, where
the change points actually fall, and whether 10:1 - the split the grid keeps finding - is as
strong as 11:5.

    python experiments/devarim_rabbah/analyze.py
"""
import collections
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grid   # noqa: E402

RES = HERE / "results"


def bh(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = np.minimum(adj, 1)
    return out


def units_and_words(target):
    """(start word, unit) of every v7 row of the target."""
    prefix = target[: -len(".jsonl")].replace("-", "_") + "_"
    rows, w = [], 0
    for line in open(grid.ROOT / "corpus_v7" / target, encoding="utf-8"):
        r = json.loads(line)
        rows.append((w, r["location"][len(prefix):]))
        w += len(r.get("sentence", "").split())
    return rows


def unit_at(rows, word):
    return [u for s, u in rows if s <= word][-1]


def split_p(seq, k, b):
    """chi-square at split b, and its circular-shift p (as in grid.py)."""
    n = len(seq)
    onehot = np.eye(k)
    cum = onehot[seq].cumsum(axis=0)
    obs = grid.chi2_split(cum, cum[-1], b, n)
    hits = 0
    for r in range(1, n):
        c = onehot[np.roll(seq, r)].cumsum(axis=0)
        hits += grid.chi2_split(c, c[-1], b, n) >= obs
    return obs, (1 + hits) / n


def key_of(r):
    return f"s{r.chunk_size}_k{r.k}_{'sm' if r.smoothing else 'raw'}_{r['mode']}_{r.balance}"


def main():
    g = pd.read_csv(RES / "grid.csv")
    pelt = pd.read_csv(RES / "pelt.csv")
    seqs = json.loads((RES / "sequences.json").read_text(encoding="utf-8"))
    groups, target = grid.corpus_files()
    rows = units_and_words(target)
    b101 = next(s for s, u in rows if u == "10_1")

    # 1. Multiple comparisons over the whole grid.
    g["q_shift"] = bh(g["p_shift"])
    g["q_late"] = bh(g["p_late"])
    raw = g[~g["smoothing"]]
    print(json.dumps({
        "configs": len(g),
        "11_5 split p<.05": int((g.p_shift < .05).sum()), "11_5 split q<.05": int((g.q_shift < .05).sum()),
        "late shift p<.05": int((g.p_late < .05).sum()), "late shift q<.05": int((g.q_late < .05).sum()),
        "late_delta>0 (unsmoothed)": f"{int((raw.late_delta > 0).sum())}/{len(raw)}",
    }, ensure_ascii=False))
    print("\nby factor (unsmoothed): share 11_5 split p<.05 | share late q<.05 | median late_delta")
    for f in ["chunk_size", "mode", "balance", "k"]:
        t = raw.groupby(f).agg(split_sig=("p_shift", lambda x: (x < .05).mean()),
                               late_sig=("q_late", lambda x: (x < .05).mean()),
                               late=("late_delta", "median"))
        print(t.round(3).to_string(), "\n")

    # 2. Where the best split falls, across the unsmoothed configurations.
    where = collections.Counter(unit_at(rows, w) for w in raw["best_split_word"])
    print("best split by unit (unsmoothed):", where.most_common(8))

    # 3. 10_1 tested like 11_5, on the saved label sequences.
    t101 = []
    for _, r in g.iterrows():
        obs, p = split_p(np.array(seqs[key_of(r)]), int(r.k), int(round(b101 / r.chunk_size)))
        t101.append(round(p, 4))
    g["p_split_10_1"] = t101
    raw = g[~g["smoothing"]]
    print("\n10_1 split: p<.05 in", int((raw.p_split_10_1 < .05).sum()), "of", len(raw), "unsmoothed;",
          "p<.025 (Bonferroni, 2 candidates) in", int((raw.p_split_10_1 < .025).sum()),
          "| median p", round(raw.p_split_10_1.median(), 3))
    print("11_5 split: p<.05 in", int((raw.p_shift < .05).sum()), "of", len(raw),
          "| median p", round(raw.p_shift.median(), 3))

    # 4. PELT, the app's method: where its change points fall in Devarim Rabbah.
    p = pelt[(~pelt.smoothing) & (pelt.n_cps.between(1, 8))]
    by_parasha = collections.Counter()
    for cps in p["cps_words"]:
        for w in json.loads(cps):
            by_parasha[unit_at(rows, w).split("_")[0]] += 1
    print("\nPELT change points by parasha (unsmoothed, 1-8 points):",
          sorted(by_parasha.items(), key=lambda x: -x[1]))
    near = p[p.nearest_cp_words.notna() & (p.nearest_cp_words <= 300)]
    print("PELT settings with a point within 300 words of 11_5:", len(near), "of", len(p),
          "| (block, penalty):", near.groupby(["block", "penalty"]).size().sort_values(ascending=False).head(4).to_dict())

    g.to_csv(RES / "grid_analyzed.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
