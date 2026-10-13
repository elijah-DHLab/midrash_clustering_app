"""Build Inputs/corpus_catalog.xlsx: one row per act_jsonl_v7 file, with everything known about it.

Replaces avg_text_length_per_file.xlsx as the pipeline's input table. Sefaria-side columns are
read from the files themselves (see v7_README.html for the field definitions); our own columns
- hierarchy, the canonical order, the notes column and the Experiment flags - are carried over
from the old table through the old->v7 name mapping in old_vs_v7_names.csv.

    python Inputs/build_catalog.py
"""
import csv
import json
import os
import re

import pandas as pd

V7_DIR = "corpus_v7"
OLD_TABLE = "Inputs/avg_text_length_per_file.xlsx"
NAME_MAP = "Inputs/old_vs_v7_names.csv"
OUT = "Inputs/corpus_catalog.xlsx"

HEBREW = re.compile(r"[א-ת]")
OUR_COLUMNS = {   # old column -> catalog column
    "hierarchy": "hierarchy",
    "Chronological - canonical order": "canonical_order",
    "Unnamed: 11": "notes",
}


def parse_categories(cats):
    """Split v7 `categories` into its parts, by the order the README gives:
    [Sefaria path...] [Hebrew scholarly mapping] [composition id?] [book] "Sefaria" [type]."""
    src = len(cats) - 1 - cats[::-1].index("Sefaria") if "Sefaria" in cats else len(cats) - 1
    book = cats[src - 1] if src >= 1 else ""
    head = cats[:max(src - 1, 0)]
    heb = next((i for i, c in enumerate(head) if HEBREW.search(c)), None)
    if heb is None:
        return head, "", "", book
    return head[:heb], head[heb], " ".join(head[heb + 1:]), book


def edition_of(filename):
    stem = filename[: -len(".jsonl")]
    if stem.endswith("-Sefaria"):
        return ""
    return stem.rsplit("-", 1)[-1][: -len("_Sefaria")]


def scan(filename):
    prefix = filename[: -len(".jsonl")].replace("-", "_") + "_"
    first = last = None
    segments = words = 0
    for line in open(os.path.join(V7_DIR, filename), encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        first = first or r
        last = r
        segments += 1
        words += r.get("text_length") or len(r.get("sentence", "").split())

    def ref(r):
        loc = r["location"]
        return loc[len(prefix):] if loc.startswith(prefix) else "?" + loc

    path, scholarly, comp_id, book = parse_categories(first.get("categories") or [])
    return {
        "file": filename,
        "he_title": first.get("he_title", ""),
        "title": first.get("title", ""),
        "cat1": path[0] if len(path) > 0 else "",
        "cat2": path[1] if len(path) > 1 else "",
        "cat3": path[2] if len(path) > 2 else "",
        "sefaria_path": " > ".join(path),
        "scholarly_category": scholarly,
        "composition_id": comp_id,
        "book": book,
        "edition": edition_of(filename),
        "type": first.get("type", ""),
        "segments": segments,
        "words": words,
        "first_ref": ref(first),
        "last_ref": ref(last),
    }


def main():
    files = sorted(f for f in os.listdir(V7_DIR) if f.endswith(".jsonl"))
    cat = pd.DataFrame([scan(f) for f in files])

    old = pd.read_excel(OLD_TABLE, dtype=str)
    exp_cols = [c for c in old.columns if c.startswith("Experiment")]
    match = {r["old_file"]: r for r in csv.DictReader(open(NAME_MAP, encoding="utf-8-sig"))}
    old["file"] = old["filename"].map(
        lambda f: match[f]["v7_match"] if f in match and match[f]["status"] in ("exact", "likely") else "")

    carried = {}
    for _, row in old[old["file"] != ""].iterrows():
        rec = {new: row.get(o) for o, new in OUR_COLUMNS.items()}
        rec.update({c: row.get(c) for c in exp_cols})
        rec["old_filename"] = row["filename"]
        prev = carried.get(row["file"])
        if prev is None:
            carried[row["file"]] = rec
            continue
        # Two old rows on one v7 file (Derekh Eretz Zuta, marked as a duplicate in the old
        # table): keep the first row's hierarchy, take the union of the Experiment flags.
        for c in exp_cols:
            if pd.notna(rec[c]):
                prev[c] = rec[c]
        prev["notes"] = "; ".join(str(x) for x in (
            prev.get("notes"), f"מוזג עם {rec['old_filename']} (hierarchy {rec['hierarchy']}, {rec['notes']})")
            if pd.notna(x) and x)
        prev["old_filename"] += " | " + rec["old_filename"]

    ours = pd.DataFrame.from_dict(carried, orient="index")
    cat = cat.merge(ours, left_on="file", right_index=True, how="left")
    cat["in_experiments"] = cat["hierarchy"].notna()

    order = (["file", "he_title", "title", "hierarchy", "canonical_order", "notes", "in_experiments"]
             + exp_cols
             + ["cat1", "cat2", "cat3", "sefaria_path", "scholarly_category", "composition_id",
                "book", "edition", "type", "segments", "words", "first_ref", "last_ref", "old_filename"])
    cat = cat[order].sort_values(["in_experiments", "file"], ascending=[False, True])
    old_only = old[old["file"] == ""].drop(columns=["file"])

    columns_doc = pd.DataFrame([
        ("file", "שם הקובץ ב-act_jsonl_v7 - המפתח של הטבלה"),
        ("he_title / title", "שם הספר בעברית / באנגלית, מתוך הקובץ"),
        ("hierarchy", "שלנו: המיקום בסדר הכרונולוגי-מחקרי. קובע את ציר הזמן. ריק = לא נכנס לניסויים"),
        ("canonical_order", "שלנו: הסדר הכרונולוגי-קנוני (מהעמודה 'Chronological - canonical order')"),
        ("notes", "שלנו: העמודה שהייתה ללא כותרת (Unnamed: 11) בטבלה הישנה"),
        ("in_experiments", "יש לקובץ hierarchy, ולכן הוא נבחר בצנרת"),
        ("Experiment ...", "שלנו: 1 = הקובץ שייך לניסוי"),
        ("cat1-cat3, sefaria_path", "היררכיית הקטגוריות של ספריא (מתוך categories)"),
        ("scholarly_category", "המיפוי המדעי בעברית (מתוך categories)"),
        ("composition_id", "מזהה החיבור מהרשימה הרשמית של חיבורים מדרשיים (מתוך categories), אם נמצא"),
        ("book", "שם הספר בעמדה הקבועה שלפני 'Sefaria' ב-categories"),
        ("edition", "קיצור מהדורת הדפוס מתוך שם הקובץ; ריק = born digital"),
        ("type", "print edition / born digital"),
        ("segments / words", "מספר השורות והמילים בקובץ (סכום text_length)"),
        ("first_ref / last_ref", "המקום בספר (location בלי קידומת הקובץ) של השורה הראשונה והאחרונה"),
        ("old_filename", "שם הקובץ בקורפוס הישן (SefariasData/rabbinic), אם היה"),
    ], columns=["עמודה", "משמעות"])

    with pd.ExcelWriter(OUT, engine="openpyxl") as xl:
        cat.to_excel(xl, sheet_name="catalog", index=False)
        old_only.to_excel(xl, sheet_name="old_only", index=False)
        columns_doc.to_excel(xl, sheet_name="columns", index=False)
    print(f"{OUT}: {len(cat)} files, {int(cat['in_experiments'].sum())} in experiments; "
          f"{len(old_only)} old rows without a v7 file")
    bad = cat[cat["first_ref"].str.startswith("?")]
    print("locations not starting with the file prefix:", len(bad), bad["file"].head(5).tolist())
    print("no Hebrew scholarly mapping:", int((cat["scholarly_category"] == "").sum()),
          "| with composition id:", int((cat["composition_id"] != "").sum()))


if __name__ == "__main__":
    main()
