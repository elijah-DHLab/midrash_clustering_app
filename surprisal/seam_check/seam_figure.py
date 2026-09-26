"""Spectrogram-style view of the two known seams: layer strips + single-cut profiles."""
import sys, re, collections
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
D = sys.argv[1]; B = 200; SMOOTH = 10; LET = re.compile(r"[^א-ת]")
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
SER = {"LM entropy": "#2a78d6", "100 function words": "#eb6834", "300 char trigrams": "#1baf7a"}
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])

def zs(X):
    X = np.asarray(X, float).reshape(len(X), -1); sd = X.std(0); k = sd > 0
    return (X[:, k] - X[:, k].mean(0)) / sd[k]

def step_r2(X):
    n = len(X); tot = ((X - X.mean(0)) ** 2).sum(); cs = np.cumsum(X, 0); cs2 = np.cumsum((X ** 2).sum(1))
    r2 = np.full(n, np.nan)
    for t in range(1, n):
        rs = cs[-1] - cs[t-1]
        r2[t] = 1 - ((cs2[t-1] - (cs[t-1] ** 2).sum() / t) + ((cs2[-1] - cs2[t-1]) - (rs ** 2).sum() / (n - t))) / tot
    return r2

def profiles(norm, block, nb, top_n, fn):
    c = collections.Counter(x for s in norm for x in fn(s)); top = {x: i for i, (x, _) in enumerate(c.most_common(top_n))}
    M = np.zeros((nb, len(top)))
    for b, s in zip(block, norm):
        for x in fn(s):
            if x in top: M[b, top[x]] += 1
    return M / M.sum(1, keepdims=True).clip(1)

works = [("shemot_rabbah", "Shemot Rabbah", [1, 5, 15, 20, 30, 40, 52]),
         ("bamidbar_rabbah", "Bamidbar Rabbah", [1, 7, 12, 15, 23])]
strips = [("surprisal_bits", "median", "Surprisal"), ("entropy_bits", "mean", "Entropy at word start"),
          ("top1_prob", "median", "Top-1 probability"), ("tanchuma_shared", "mean", "Shared with Tanchuma")]
fig, axes = plt.subplots(len(strips) + 1, 2, figsize=(13, 6.6), facecolor=SURF,
                         gridspec_kw=dict(height_ratios=[1] * len(strips) + [3.2], hspace=0.12, wspace=0.08))
for col, (work, title, ticks) in enumerate(works):
    w = pd.read_csv("%s/%s_lm.csv" % (D, work), encoding="utf-8")
    nb = len(w) // B; w = w.iloc[:nb * B].copy(); w["block"] = np.arange(len(w)) // B
    g = w.groupby("block"); ch = g["chapter"].first().values; seam = int(np.argmax(ch >= 15))
    x = np.arange(nb)
    for row, (c, how, label) in enumerate(strips):
        ax = axes[row, col]
        v = pd.Series(getattr(g[c], how)().values).rolling(SMOOTH, center=True, min_periods=1).mean()
        z = ((v - v.mean()) / v.std()).clip(-1.5, 1.5).values
        ax.imshow(z[None, :], aspect="auto", cmap=DIV, vmin=-1.5, vmax=1.5, extent=(0, nb, 0, 1), interpolation="nearest")
        ax.axvline(seam, color=INK, lw=1.6); ax.set_yticks([]); ax.set_xticks([])
        for s in ax.spines.values(): s.set_visible(False)
        if col == 0: ax.set_ylabel(label, rotation=0, ha="right", va="center", color=INK2, fontsize=9)
        if row == 0: ax.set_title(title, color=INK, fontsize=11, loc="left", pad=6)
    ax = axes[-1, col]
    norm = w["word"].map(lambda s: LET.sub("", str(s)))
    series = {"LM entropy": zs(g["entropy_bits"].mean().values),
              "100 function words": zs(profiles(norm, w["block"], nb, 100, lambda s: [s] if s else [])),
              "300 char trigrams": zs(profiles(norm, w["block"], nb, 300, lambda s: [s[i:i+3] for i in range(len(s) - 2)]))}
    lo = int(0.05 * nb)
    for name, X in series.items():
        r = step_r2(X); r[:lo] = np.nan; r[nb - lo:] = np.nan; r = r / np.nanmax(r)
        ax.plot(x, r, color=SER[name], lw=2, label=name)
        b = int(np.nanargmax(r)); ax.plot([b], [1], "o", ms=6, color=SER[name], mec=SURF, mew=2)
    ax.axvline(seam, color=INK, lw=1.6)
    ax.text(seam + nb * 0.008, 0.04, "known seam\n(ch. 15 begins)", color=INK, fontsize=8.5, va="bottom")
    tick_pos = [int(np.argmax(ch >= t)) for t in ticks]
    ax.set_xticks(tick_pos, ["ch. %d" % t for t in ticks], fontsize=8.5, color=MUTED)
    ax.set_xlim(0, nb); ax.set_ylim(0, 1.08); ax.set_facecolor(SURF)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.tick_params(length=0)
    for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    if col == 0:
        ax.set_ylabel("How well one cut here\nsplits the layer\n(scaled to its best)", rotation=0, ha="right", va="center", color=INK2, fontsize=9)
        ax.set_yticks([0, 0.5, 1], ["0", ".5", "1"], fontsize=8.5, color=MUTED)
    else:
        ax.set_yticks([0, 0.5, 1], ["", "", ""])
h, l = axes[-1, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, -0.04))
fig.text(0.125, 0.965, "Strips: 200-word blocks, 10-block rolling mean, z-scored (blue low, red high). "
         "Dots: where each layer would place a single cut.", color=INK2, fontsize=9)
fig.savefig(sys.argv[2], dpi=160, bbox_inches="tight", facecolor=SURF)
print("->", sys.argv[2])
