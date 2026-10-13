"""Old SefariasData/rabbinic JSON names vs. act_jsonl_v7 names, with a fuzzy best match."""
import os, re, csv, json, difflib
OLD = "G:/My Drive/Haifa/MIDRASH/SefariasData/rabbinic"
old = sorted(f for f in os.listdir(OLD) if f.endswith(".json"))
new = [l.strip() for l in open("Inputs/v7_files.txt", encoding="utf-8") if l.strip()]
store = set(json.load(open("embedding_store/rabbinic/50/index.json", encoding="utf-8")))

def norm(s):
    s = re.sub(r"\.jsonl?$", "", s)
    s = re.sub(r"[-_]?([A-Z0-9_]*[A-Z]{3,}[A-Z0-9_]*_)?Sefaria$", "", s)   # edition + Sefaria suffix
    return re.sub(r"[\s_\-]+", " ", s).strip().lower()

nn = {norm(n): n for n in new}
# Checked by hand: the fuzzy guess is wrong for these.
MANUAL = {
    "Midrash-Aggadic Midrash-Derech Eretz Zuta-.json":
        "Tanaitic-Minor_Tractates-Tractate_Derekh_Eretz_Zuta-VILNA1883_Sefaria.jsonl",
    "Midrash-Aggadic Midrash-Ein Yaakov-.json": "",   # v7 has only the Glick edition
}
rows, used = [], set()
for o in old:
    k = norm(o)
    if o in MANUAL: m, sc = MANUAL[o], (1.0 if MANUAL[o] else 0)
    elif k in nn: m, sc = nn[k], 1.0
    else:
        c = difflib.get_close_matches(k, nn, n=1, cutoff=0.0)
        m, sc = (nn[c[0]], difflib.SequenceMatcher(None, k, c[0]).ratio()) if c else ("", 0)
    if sc >= 0.85: used.add(m)
    rows.append([o.split("-")[0], o, "yes" if o in store else "", m if sc >= 0.6 else "",
                 round(sc, 2), "exact" if sc == 1 else "likely" if sc >= 0.85 else "check" if sc >= 0.6 else "none"])
with open("Inputs/old_vs_v7_names.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(["category", "old_file", "in_store", "v7_match", "score", "status"]); w.writerows(rows)
with open("Inputs/v7_unmatched.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(["category", "v7_file"])
    for n in new:
        if n not in used: w.writerow([n.split("-")[0], n])
from collections import Counter
print(Counter(r[5] for r in rows)); print("in store:", Counter(r[5] for r in rows if r[2]))
print("by cat (status):"); 
for cat in sorted({r[0] for r in rows}): print(" ", cat, dict(Counter(r[5] for r in rows if r[0]==cat)))
print("v7 not matched:", len(new)-len(used))
