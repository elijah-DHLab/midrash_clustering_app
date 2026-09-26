"""Blind single-seam detection on two works with a known philological seam (ch 14|15)."""
import sys, re, collections
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
D = sys.argv[1]; B = 200; SEAM_CH = 15
LET = re.compile(r"[^א-ת]")

def step_r2(X):
    """R^2 of a one-step model at every split t (segments [:t], [t:]), X blocks x features, standardized."""
    n = len(X); tot = ((X - X.mean(0)) ** 2).sum()
    cs = np.cumsum(X, 0); cs2 = np.cumsum((X ** 2).sum(1))
    r2 = np.full(n, np.nan)
    for t in range(1, n):
        left = cs2[t-1] - (cs[t-1] ** 2).sum() / t
        rs = cs[-1] - cs[t-1]; right = (cs2[-1] - cs2[t-1]) - (rs ** 2).sum() / (n - t)
        r2[t] = 1 - (left + right) / tot
    return r2

def zs(X):
    X = np.asarray(X, float); X = X.reshape(len(X), -1)
    sd = X.std(0); keep = sd > 0
    return (X[:, keep] - X[:, keep].mean(0)) / sd[keep]

for work in ["shemot_rabbah", "bamidbar_rabbah"]:
    w = pd.read_csv("%s/%s_lm.csv" % (D, work), encoding="utf-8")
    assert w["surprisal_bits"].notna().all()
    nb = len(w) // B; w = w.iloc[:nb * B].copy(); w["block"] = np.arange(len(w)) // B
    g = w.groupby("block")
    norm = w["word"].map(lambda s: LET.sub("", str(s)))
    top_fw = [x for x, _ in collections.Counter(norm[norm != ""]).most_common(100)]
    tri = collections.Counter(t[i:i+3] for t in norm for i in range(len(t) - 2))
    top_tri = [x for x, _ in tri.most_common(300)]
    fw = np.zeros((nb, len(top_fw))); ct = np.zeros((nb, len(top_tri)))
    fwi = {x: i for i, x in enumerate(top_fw)}; tti = {x: i for i, x in enumerate(top_tri)}
    for b, (blk, s) in enumerate(zip(w["block"], norm)):
        if s in fwi: fw[blk, fwi[s]] += 1
        for i in range(len(s) - 2):
            j = tti.get(s[i:i+3])
            if j is not None: ct[blk, j] += 1
    ct = ct / ct.sum(1, keepdims=True).clip(1)

    feats = {
        "LM  surprisal (median)":      g["surprisal_bits"].median().values,
        "LM  entropy at word start":   g["entropy_bits"].mean().values,
        "LM  top-1 prob":              g["top1_prob"].median().values,
        "LM  first token = argmax":    g["rank_first_token"].apply(lambda r: (r == 1).mean()).values,
        "base tokens per word":        g["n_tokens"].mean().values,
        "base word length":            norm.str.len().groupby(w["block"]).mean().values,
        "base 100 function words":     fw,
        "base 300 char trigrams":      ct,
        "ctrl shared with Tanchuma":   g["tanchuma_shared"].mean().values,
    }
    # surprisal with the Tanchuma-overlap share regressed out across blocks
    y = feats["LM  surprisal (median)"]; x = feats["ctrl shared with Tanchuma"]
    beta = np.polyfit(x, y, 1); feats["LM  surprisal | Tanchuma share"] = y - np.polyval(beta, x)

    blk_ch = g["chapter"].first().values
    true_t = int(np.argmax(blk_ch >= SEAM_CH))
    lo = max(2, int(0.05 * nb))
    print("\n=== %s: %d blocks of %d words; seam (ch %d starts) at block %d = %.0f%% of the work"
          % (work, nb, B, SEAM_CH, true_t, 100 * true_t / nb))
    print("  %-32s %8s %11s %9s %10s %12s" % ("layer", "R2@seam", "seam pctile", "best cut", "cut in ch", "part1-part2"))
    for name, X in feats.items():
        Z = zs(X); r2 = step_r2(Z); valid = np.arange(nb); valid = (valid >= lo) & (valid <= nb - lo)
        best = int(np.nanargmax(np.where(valid, r2, np.nan)))
        pct = 100 * (np.nansum(r2[valid] < r2[true_t])) / valid.sum()
        diff = ""
        if np.ndim(X) == 1:
            a, bb = X[:true_t], X[true_t:]
            diff = "%+.2f sd" % ((a.mean() - bb.mean()) / np.sqrt((a.var() + bb.var()) / 2))
        print("  %-32s %8.3f %10.1f%% %9d %10d %12s" % (name, r2[true_t], pct, best, blk_ch[best], diff))
    print("  Tanchuma share -> surprisal slope: %+.2f bits per 100%% shared" % beta[0])
