"""How many candidates does the semantic measure need?

Recomputes the paper's Hebrew semantic distance (beam + lemma, probability-weighted,
DictaLM layer-0 embeddings) from the candidates stored in its word_level.csv, keeping
only the k most probable lemma groups, and compares each truncation with the full set.
Read-only with respect to the paper: it reads a copy of the measurement file.

    python topk_check.py he_paper_word_level.csv pos_tags_poetry.csv pos_tags_prose.csv
"""
import re, sys, time
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr, pearsonr
from footer_filter import strip_footer_rows

sys.stdout.reconfigure(encoding="utf-8")
CLEAN = re.compile(r"[^א-תa-zA-Z]")
KS = [1, 3, 5, 10, 20, 50]
CONTENT = {"NOUN", "VERB", "ADJ", "ADV"}
MODEL = "dicta-il/DictaLM-3.0-1.7B-Base"
dev = "cuda"

t0 = time.time()
df = pd.read_csv(sys.argv[1], encoding="utf-8-sig", low_memory=False)
df["text_id"] = df["text_id"].astype(str).str.replace(".txt", "", regex=False)
rows = df[["text_id", "actual_word", "word_position"]].to_dict("records")
for i, r in enumerate(rows): r["_i"] = i
kept = strip_footer_rows(rows)
df = df.iloc[[r["_i"] for r in kept]].reset_index(drop=True)
pos = pd.concat([pd.read_csv(p, encoding="utf-8-sig") for p in sys.argv[2:4]])
pos["text_id"] = pos["text_id"].astype(str).str.replace(".txt", "", regex=False)
df = df.merge(pos[["text_id", "word_position", "pos"]], on=["text_id", "word_position"], how="left")
df["clean"] = df["actual_word"].astype(str).map(lambda w: CLEAN.sub("", w))
df["content"] = df["pos"].isin(CONTENT) & df["clean"].ne("") & (df["word_position"] > 3)
n = len(df)
print("rows after footer strip: %d (content %d); texts %d; %.0fs" % (n, df.content.sum(), df.text_id.nunique(), time.time() - t0), flush=True)

pred = df[["pred_%d" % i for i in range(1, 51)]].fillna("").astype(str).map(str.strip).to_numpy()
prob = df[["prob_%d" % i for i in range(1, 51)]].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float64).copy()
prob[pred == ""] = 0

# ---- lemmatise every form, as recompute_lemma_cosine.py does ----
from transformers import AutoModel, AutoTokenizer
forms = sorted(set(df["clean"]) | set(pred[pred != ""].ravel()) - {""})
print("distinct forms: %d" % len(forms), flush=True)
ltok = AutoTokenizer.from_pretrained("dicta-il/dictabert-lex")
lmod = AutoModel.from_pretrained("dicta-il/dictabert-lex", trust_remote_code=True).eval().to(dev)
# The remote code's predict() hands lex_parse_logits the tokenizer's dict instead of
# the id lists it indexes, so every call raises KeyError under this transformers
# version. Run the forward here and pass the ids explicitly; the parsing is theirs.
lex_parse_logits = sys.modules[type(lmod).__module__].lex_parse_logits
lemma, failed = {}, 0
with torch.no_grad():
    for i in range(0, len(forms), 96):
        chunk = forms[i:i + 96]
        enc = ltok(chunk, padding="longest", truncation=True, return_tensors="pt")
        logits = lmod(**{k: v.to(dev) for k, v in enc.items()}, return_dict=True).logits
        res = lex_parse_logits(enc["input_ids"].tolist(), chunk, ltok, logits)
        for w, r in zip(chunk, res):
            lem = w
            try:
                if r:
                    for pair in r:
                        if len(pair) >= 2 and pair[1] and pair[1] != "[BLANK]":
                            lem = pair[1]
            except Exception:
                failed += 1
            lemma[w] = lem
changed = sum(1 for w, l in lemma.items() if l != w)
print("lemmas differing from the form: %d of %d (%.0f%%); parse failures %d" % (changed, len(lemma), 100 * changed / len(lemma), failed), flush=True)
print("lemmatised in %.0fs" % (time.time() - t0), flush=True)
del lmod; torch.cuda.empty_cache()

# ---- layer-0 embeddings of every lemma: mean over its subword tokens ----
from transformers import AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(MODEL)
E = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).get_input_embeddings().weight.detach().to(dev)
lemmas = sorted(set(lemma.values()) | {""})
lidx = {l: i for i, l in enumerate(lemmas)}
ids = tok([" " + l for l in lemmas], add_special_tokens=False)["input_ids"]
flat = torch.tensor([t for s in ids for t in s], device=dev)
offs = torch.tensor(np.concatenate([[0], np.cumsum([len(s) for s in ids])[:-1]]), device=dev)
L = torch.nn.functional.embedding_bag(flat, E, offs, mode="mean")
L = torch.nn.functional.normalize(L, dim=1).half()
empty = torch.tensor([len(s) == 0 or l == "" for s, l in zip(ids, lemmas)], device=dev)

a_idx = torch.tensor([lidx[lemma.get(w, w)] if w else lidx[""] for w in df["clean"]], device=dev)
p_idx = torch.tensor(np.vectorize(lambda w: lidx[lemma.get(w, w)] if w else -1)(pred), device=dev)
P = torch.tensor(prob, device=dev)
D = torch.empty((n, 50), dtype=torch.float64, device=dev)
for s in range(0, n, 4096):
    pi = p_idx[s:s + 4096].clamp(min=0)
    cos = torch.einsum("rkd,rd->rk", L[pi].float(), L[a_idx[s:s + 4096]].float())
    D[s:s + 4096] = (1 - cos).double()
valid_a = ~empty[a_idx]
print("distances in %.0fs" % (time.time() - t0), flush=True)

cum_w = torch.cumsum(P, 1); cum_wd = torch.cumsum(P * D, 1)
same = (p_idx == a_idx[:, None]) & (p_idx >= 0)
out = {}
for k in KS:
    c = (cum_wd[:, k - 1] / cum_w[:, k - 1]).cpu().numpy()
    c[~valid_a.cpu().numpy() | (cum_w[:, k - 1] <= 0).cpu().numpy()] = np.nan
    df["cos_k%d" % k] = c
    df["inlist_k%d" % k] = same[:, :k].any(1).cpu().numpy()

ok = df["cos_dist_centroid"].notna() & df["cos_k50"].notna()
dd = (df.loc[ok, "cos_k50"] - df.loc[ok, "cos_dist_centroid"]).abs()
print("\nrecompute vs stored cos_dist_centroid (k=50): median |d| %.5f, %.1f%% within 1e-3, max %.4f"
      % (dd.median(), 100 * (dd < 1e-3).mean(), dd.max()))
print("groups actually used by the paper's procedure (n_predictions): median %d, 75th pct %d"
      % (df["n_predictions"].median(), df["n_predictions"].quantile(.75)))

for label, sub in [("all words", df), ("content words (paper filter)", df[df.content])]:
    print("\n=== %s: %d words" % (label, len(sub)))
    print("  %-4s %10s %10s %8s | %10s %10s %10s %10s %10s" % (
        "k", "rho(word)", "med|dcos|", "in-list", "poetry rho", "prose rho", "gap", "r(text rho)", "med|drho|"))
    per = {}
    for k in KS + [0]:
        col = "cos_k%d" % k if k else "cos_dist_centroid"
        g = sub.dropna(subset=[col, "surprisal_bits"]).groupby("text_id")
        rho = g.apply(lambda d: spearmanr(d["surprisal_bits"], d[col])[0] if len(d) >= 20 else np.nan, include_groups=False)
        per[k] = rho
    genre = sub.groupby("text_id")["genre"].first()
    for k in KS:
        x = sub[["cos_k%d" % k, "cos_k50"]].dropna()
        r = per[k].dropna(); r50 = per[50].reindex(r.index)
        pm = r[genre.reindex(r.index) == "poetry"].median(); qm = r[genre.reindex(r.index) == "prose"].median()
        print("  %-4d %10.3f %10.4f %7.1f%% | %10.3f %10.3f %10.3f %10.3f %10.3f" % (
            k, spearmanr(x.iloc[:, 0], x.iloc[:, 1])[0], (x.iloc[:, 0] - x.iloc[:, 1]).abs().median(),
            100 * sub["inlist_k%d" % k].mean(), pm, qm, qm - pm,
            pearsonr(r, r50)[0], (r - r50).abs().median()))
print("\ndone in %.0fs" % (time.time() - t0))
