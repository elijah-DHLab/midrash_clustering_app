"""The leading configurations of the grid, rerun with other random seeds (KMeans initialisation
and, when balanced, the genre subsample): does the late shift after 11:5 hold, and does the
best split stay at 10:1?

    python experiments/devarim_rabbah/stability.py
"""
import pathlib
import sys

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grid   # noqa: E402

CONFIGS = [(25, 20, "projected", "balanced"), (25, 20, "included", "natural"),
           (25, 15, "projected", "balanced"), (50, 8, "included", "natural")]
SEEDS = [1, 2, 3, 4, 5]

groups, target = grid.corpus_files()
b_word = grid.boundary_word(target)
out, loaded = [], {}
for size, k, mode, balance in CONFIGS:
    if size not in loaded:
        loaded[size] = grid.store.load_selection(list(groups) + [target], size, grid.STORE)
    for seed in SEEDS:
        row, _, _ = grid.run_one(loaded[size], groups, target, b_word, size, k, False, mode, balance,
                                 rng_seed=seed, km_seed=seed)
        row["seed"] = seed
        out.append(row)
        print(size, k, mode, balance, seed, "late", row["late_delta"], "p", row["p_late"],
              "| 11_5 p", row["p_shift"], "| best split", row["best_split_word"], flush=True)
pd.DataFrame(out).to_csv(HERE / "results" / "stability.csv", index=False, encoding="utf-8-sig")
