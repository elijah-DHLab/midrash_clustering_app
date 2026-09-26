"""One causal-LM pass over a running word stream, sliding fixed-length context.

Every scored token sees at least L-STRIDE tokens of preceding text (except the
first window of the work). Words joined by a single space; no trailing-space
token. Per word: surprisal (sum over its tokens), and at its first token the
entropy of the next-token distribution, the top-1 probability, and the rank of
the actual token.
"""
import math, pickle, sys, time
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "dicta-il/DictaLM-3.0-1.7B-Base"
L, STRIDE, BATCH = 512, 256, 16
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
LN2 = math.log(2)

for work, d in pickle.load(open(sys.argv[1], "rb")).items():
    t0 = time.time()
    words = d["words"]
    starts, pos = [], 0
    for w in words:
        starts.append(pos); pos += len(w) + 1
    enc = tok(" ".join(words), return_offsets_mapping=True, add_special_tokens=False)
    ids = np.array(enc["input_ids"]); N = len(ids)
    tw = np.searchsorted(np.array(starts), np.array([e - 1 for s, e in enc["offset_mapping"]]), side="right") - 1
    lp = np.full(N, np.nan); ent = np.full(N, np.nan); top1 = np.full(N, np.nan); rank = np.full(N, np.nan)

    wins = []
    s = 0
    while True:
        e = min(s + L, N); wins.append((s, e))
        if e == N: break
        s += STRIDE
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id   # Qwen-style: <|endoftext|> opens a document
    for b in range(0, len(wins), BATCH):
        chunk = wins[b:b + BATCH]
        width = max(e - s for s, e in chunk) + 1
        x = torch.full((len(chunk), width), tok.pad_token_id or 0, dtype=torch.long)
        att = torch.zeros_like(x)
        for k, (s, e) in enumerate(chunk):
            x[k, 0] = bos; x[k, 1:e - s + 1] = torch.from_numpy(ids[s:e]); att[k, :e - s + 1] = 1
        with torch.inference_mode():
            hidden = model.model(input_ids=x.cuda(), attention_mask=att.cuda()).last_hidden_state
        for k, (s, e) in enumerate(chunk):
            first = s if s == 0 else s + (L - STRIDE)
            js = torch.arange(first - s, e - s, device="cuda")          # logits[j] predicts token s+j
            with torch.inference_mode():
                logp = torch.log_softmax(model.lm_head(hidden[k, js]).float(), -1)
            tgt = torch.from_numpy(ids[first:e]).cuda()
            a = logp.gather(1, tgt[:, None])[:, 0]
            p = logp.exp()
            lp[first:e] = a.cpu().numpy()
            ent[first:e] = (-(p * logp).sum(-1) / LN2).cpu().numpy()
            top1[first:e] = p.max(-1).values.cpu().numpy()
            rank[first:e] = ((logp > a[:, None]).sum(-1) + 1).cpu().numpy()

    df = pd.DataFrame({"w": tw, "lp": lp, "ent": ent, "top1": top1, "rank": rank})
    g = df.groupby("w")
    out = pd.DataFrame({
        "surprisal_bits": -g["lp"].sum() / LN2,
        "n_tokens": g.size(),
        "entropy_bits": g["ent"].first(),
        "top1_prob": g["top1"].first(),
        "rank_first_token": g["rank"].first(),
    }).reindex(range(len(words)))
    out.insert(0, "word", words); out.insert(1, "chapter", d["chap"]); out.insert(2, "tanchuma_shared", d["shared"])
    out.to_csv("%s_lm.csv" % work, index_label="word_index", encoding="utf-8")
    print("%-16s %d words, %d tokens, %d windows, %.0fs; median surprisal %.2f bits"
          % (work, len(words), N, len(wins), time.time() - t0, out["surprisal_bits"].median()), flush=True)
