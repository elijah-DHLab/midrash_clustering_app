"""Content check: each experiment book (rows with `hierarchy`) old JSON vs. its v7 JSONL."""
import json, csv, collections, pandas as pd
OLD = "G:/My Drive/Haifa/MIDRASH/SefariasData/rabbinic/"
NEW = "corpus_v7/"

def load(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: pass
    return rows

def norm_loc(l):   # v7 adds an edition tag and drops apostrophes; compare the trailing ref numbers
    return tuple(l.split("_")[-2:])

df = pd.read_excel("Inputs/avg_text_length_per_file.xlsx", dtype=str).dropna(subset=["hierarchy"])
match = {x["old_file"]: x["v7_match"] for x in csv.DictReader(open("Inputs/old_vs_v7_names.csv", encoding="utf-8-sig"))}
out = []
for f in df.filename:
    o, n = load(OLD + f), load(NEW + match[f])
    ow = [w for r in o for w in (r.get("sentence") or "").split()]
    nw = [w for r in n for w in (r.get("sentence") or "").split()]
    oc, nc = collections.Counter(ow), collections.Counter(nw)
    shared = sum((oc & nc).values())
    heb = sum(any("\u05d0" <= ch <= "\u05ea" for ch in w) for w in nw[:2000]) / max(1, min(2000, len(nw)))
    out.append(dict(filename=f, v7=match[f], old_rows=len(o), v7_rows=len(n), old_words=len(ow), v7_words=len(nw),
                    word_ratio=round(len(nw) / max(1, len(ow)), 3),
                    token_overlap=round(shared / max(1, max(len(ow), len(nw))), 3),
                    v7_hebrew_share=round(heb, 3),
                    same_first_text=(o[0].get("sentence", "")[:80] == n[0].get("sentence", "")[:80]) if o and n else False))
res = pd.DataFrame(out)
res.to_csv("Inputs/old_vs_v7_content.csv", index=False, encoding="utf-8-sig")
print(res[["word_ratio", "token_overlap", "v7_hebrew_share"]].describe().round(3).to_string())
print("same first text:", res.same_first_text.sum(), "/", len(res))
print("total words old/new:", res.old_words.sum(), res.v7_words.sum())
