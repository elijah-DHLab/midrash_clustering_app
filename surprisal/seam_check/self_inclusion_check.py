"""Is the Hebrew coupling a restatement of probability? (self-inclusion check)

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


same = (p_idx == a_idx[:, None]) & (p_idx >= 0)
w_all = P.sum(1); w_self = (P * same).sum(1); wd_all = (P * D).sum(1)
va = valid_a.cpu().numpy()
df["cos_full"] = np.where(va & (w_all > 0).cpu().numpy(), (wd_all / w_all).cpu().numpy(), np.nan)
rest = (w_all - w_self)
df["cos_excl"] = np.where(va & (rest > 1e-12).cpu().numpy(), ((wd_all - (P * D * same).sum(1)) / rest).cpu().numpy(), np.nan)
df["q_self"] = (w_self / w_all.clamp(min=1e-12)).cpu().numpy()
df["inlist"] = same.any(1).cpu().numpy()
sub = df[df.content].copy()
print()
print("content words %d; in-list %.1f%% (poetry %.1f%%, prose %.1f%%)" % (len(sub), 100*sub.inlist.mean(),
      100*sub[sub.genre=="poetry"].inlist.mean(), 100*sub[sub.genre=="prose"].inlist.mean()))
il = sub[sub.inlist].dropna(subset=["cos_full"])
print("in-list words: Pearson(cos, q_self) = %.3f ; Spearman(cos, surprisal) = %.3f" % (
      pearsonr(il.cos_full, il.q_self)[0], spearmanr(il.cos_full, il.surprisal_bits)[0]))
ol = sub[~sub.inlist].dropna(subset=["cos_full"])
print("out-of-list words: Spearman(cos, surprisal) = %.3f" % spearmanr(ol.cos_full, ol.surprisal_bits)[0])
def per_text(frame, col, minn=20):
    g = frame.dropna(subset=[col]).groupby("text_id")
    return g.apply(lambda d: spearmanr(d.surprisal_bits, d[col])[0] if len(d) >= minn else np.nan, include_groups=False)
genre = sub.groupby("text_id")["genre"].first()
rng = np.random.default_rng(0)
print()
print("  %-34s %8s %8s %8s %10s" % ("per-text rho (surprisal vs semantic)", "poetry", "prose", "gap", "perm p"))
for label, frame, col in [("full measure (paper)", sub, "cos_full"),
                          ("word removed from its own set", sub, "cos_excl"),
                          ("out-of-list words only", sub[~sub.inlist], "cos_full"),
                          ("in-list words only", sub[sub.inlist], "cos_full")]:
    r = per_text(frame, col).dropna(); gg = genre.reindex(r.index)
    obs = r[gg=="prose"].median() - r[gg=="poetry"].median()
    lab = gg.to_numpy(); vals = r.to_numpy(); hits = 0
    for _ in range(5000):
        perm = rng.permutation(lab)
        hits += (np.median(vals[perm=="prose"]) - np.median(vals[perm=="poetry"])) >= obs
    print("  %-34s %8.3f %8.3f %8.3f %10.4f   (texts: %d poetry, %d prose)" % (label, r[gg=="poetry"].median(),
          r[gg=="prose"].median(), obs, (hits+1)/5001, (gg=="poetry").sum(), (gg=="prose").sum()))
print()
print("done in %.0fs" % (time.time() - t0))
