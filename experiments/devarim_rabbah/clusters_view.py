"""What the clusters are, seen through the first method: for one configuration (25-word
chunks, k=20, balanced fit, Devarim Rabbah projected - the one profile.py draws), each
cluster's genre make-up, its characteristic words, and how Devarim Rabbah's chunks are spread
over the clusters before 10:1, between 10:1 and 11:5, and after 11:5.

    python experiments/devarim_rabbah/clusters_view.py
"""
import collections
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grid   # noqa: E402

SIZE, K, SEED = 25, 20, 1
B101, B115 = 28583, 30603
GENRE_HE = {"classic": "קלאסי", "late": "מאוחר", "tannaitic": "תנאים",
            "mishnah_tosefta": "משנה-תוספתא", "bavli": "בבלי", "yerushalmi": "ירושלמי"}


def main():
    groups, target = grid.corpus_files()
    df = grid.store.load_selection(list(groups) + [target], SIZE, grid.STORE)
    files = df["filename"].to_numpy()
    vecs = np.vstack(df["vec"].to_numpy()).astype(np.float32)
    genre = np.array([groups.get(f, "target") for f in files])
    is_target = files == target
    names = list(grid.GROUPS)

    rng = np.random.default_rng(SEED)
    m = min(int((genre == g).sum()) for g in names)
    fit_idx = np.concatenate([rng.choice(np.flatnonzero(genre == g), m, replace=False) for g in names])
    km = KMeans(n_clusters=K, random_state=SEED, n_init=3).fit(vecs[fit_idx])
    labels = km.predict(vecs)

    # Genre make-up of each cluster, read off the balanced sample it was fitted on.
    comp = pd.crosstab(labels[fit_idx], genre[fit_idx]).reindex(columns=names, fill_value=0)
    share = comp.div(comp.sum(axis=1), axis=0)

    # Characteristic words: frequent in the cluster's chunks, rare elsewhere (ratio of rates).
    texts = df["chunk"].to_numpy()
    overall = collections.Counter()
    per = {c: collections.Counter() for c in range(K)}
    for i in fit_idx:
        words = str(texts[i]).split()
        overall.update(words)
        per[labels[i]].update(words)
    total = sum(overall.values())

    def top_words(c, n=8):
        tot_c = sum(per[c].values())
        scored = [(w, (cnt / tot_c) / (overall[w] / total)) for w, cnt in per[c].items() if cnt >= 30]
        return " ".join(w for w, _ in sorted(scored, key=lambda x: -x[1])[:n])

    # Devarim Rabbah over the clusters, in three stretches.
    seq = labels[is_target]
    words = np.arange(len(seq)) * SIZE
    parts = {"before_10_1": seq[words < B101], "10_1_to_11_5": seq[(words >= B101) & (words < B115)],
             "after_11_5": seq[words >= B115]}
    dist = {p: np.bincount(s, minlength=K) / len(s) for p, s in parts.items()}

    rows = []
    for c in range(K):
        main_genre = share.loc[c].idxmax()
        rows.append({
            "cluster": c,
            "main_genre": GENRE_HE[main_genre], "main_share": round(share.loc[c, main_genre] * 100),
            "late_share": round(share.loc[c, "late"] * 100), "classic_share": round(share.loc[c, "classic"] * 100),
            "devarim_before_10_1": round(dist["before_10_1"][c] * 100, 1),
            "devarim_10_1_to_11_5": round(dist["10_1_to_11_5"][c] * 100, 1),
            "devarim_after_11_5": round(dist["after_11_5"][c] * 100, 1),
            "words": top_words(c),
        })
    out = pd.DataFrame(rows).sort_values("devarim_before_10_1", ascending=False)
    out.to_csv(HERE / "results" / "clusters_view.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    print(out.to_string(index=False))
    print("chunks per stretch:", {p: len(s) for p, s in parts.items()})


if __name__ == "__main__":
    main()
