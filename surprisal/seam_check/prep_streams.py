"""Running word streams for Shemot and Bamidbar Rabbah, plus a per-word flag for
text shared with printed Tanchuma (5-gram shingles), cleaned as build_corpus.py."""
import json, re, pickle, collections, sys
sys.stdout.reconfigure(encoding="utf-8")
BASE = "G:/My Drive/Haifa/MIDRASH/SefariasData/rabbinic/Midrash-Aggadic Midrash-"
PAREN = re.compile(r"\([^)]{1,60}\)"); NIKKUD = re.compile(r"[\u0591-\u05C7]")
BIDI = re.compile(r"[\u200e\u200f\u202a-\u202e]"); LET = re.compile(r"[^א-ת]")

def stream(name, need_chapter=True):
    words, chap = [], []
    for line in open(BASE + name + "-.json", encoding="utf-8"):
        if not line.strip(): continue
        r = json.loads(line)
        m = re.search(r"__(\d+)_(\d+)$", r["location"])
        if not m and need_chapter: continue
        t = BIDI.sub("", NIKKUD.sub("", PAREN.sub(" ", str(r.get("orig_sentence", "")))))
        w = t.split(); words += w; chap += [int(m.group(1)) if m else 0] * len(w)
    return words, chap

norm = lambda ws: [LET.sub("", w) for w in ws]
tw, _ = stream("Midrash Tanchuma", need_chapter=False); print("tanchuma words", len(tw)); tn = [w for w in norm(tw) if w]
T5 = set(tuple(tn[i:i+5]) for i in range(len(tn) - 4))
out = {}
for key, name in [("shemot_rabbah", "Midrash Rabbah-Shemot Rabbah"),
                  ("bamidbar_rabbah", "Midrash Rabbah-Bamidbar Rabbah")]:
    words, chap = stream(name)
    n = norm(words); shared = [0] * len(words)
    idx = [i for i, w in enumerate(n) if w]
    for j in range(len(idx) - 4):
        if tuple(n[i] for i in idx[j:j+5]) in T5:
            for i in idx[j:j+5]: shared[i] = 1
    cnt = collections.Counter(chap); p1 = sum(v for k, v in cnt.items() if k <= 14)
    sh1 = sum(s for s, c in zip(shared, chap) if c <= 14) / p1
    sh2 = sum(s for s, c in zip(shared, chap) if c > 14) / (len(words) - p1)
    print("%-16s %6d words, part1 %.1f%%, chapters %d-%d | shared with Tanchuma: part1 %.1f%%, part2 %.1f%%"
          % (key, len(words), 100*p1/len(words), min(cnt), max(cnt), 100*sh1, 100*sh2))
    out[key] = dict(words=words, chap=chap, shared=shared)
pickle.dump(out, open(sys.argv[1], "wb"))
