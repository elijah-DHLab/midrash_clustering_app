"""Old corpus (SefariasData/rabbinic) vs. new (act_jsonl_v7), one Excel workbook to send.

Every old file and every new file appears exactly once: a pair on one row when they are the
same text, alone with "no counterpart" when not. Matches come from old_vs_v7_names.csv where the
name match was exact or near-exact, and from MANUAL, checked by hand; the fuzzy guesses that were
wrong are dropped there. For each pair the word counts and the overlap of their vocabularies are
given, so that a text that changed between the two corpora can be seen.

    python Inputs/build_comparison_for_moshe.py
"""
import collections
import csv
import json
import os

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

OLD_DIR = "G:/My Drive/Haifa/MIDRASH/SefariasData/rabbinic"
NEW_DIR = "corpus_v7"
OUT = "Inputs/old_vs_new_corpus_comparison.xlsx"

# Checked by hand. A value of None means: no counterpart, although the name matcher offered one.
MANUAL = {
    "Midrash-Aggadic Midrash-Derech Eretz Zuta-.json":
        "Tanaitic-Minor_Tractates-Tractate_Derekh_Eretz_Zuta-VILNA1883_Sefaria.jsonl",
    "Midrash-Aggadic Midrash-Ein Yaakov-.json": None,
    "Kabbalah-Sefer HaKana-.json": "Kabbalah-Other_Kabbalah_Works-Sefer_HaKanah-Sefaria.jsonl",
    "Kabbalah-Shaarei Orah-.json": "Kabbalah-Other_Kabbalah_Works-Shaarei_Orah-Sefaria.jsonl",
    "Kabbalah-Sefer Yetzirah-.json": "Kabbalah-Sefer_Yetzirah-Sefer_Yetzirah-WARSAW1884_Sefaria.jsonl",
    "Apocrypha-Sefer Yehudit-.json": "Second_Temple-Apocrypha-Book_of_Judith-FRANKEL1885_Sefaria.jsonl",
    "Kabbalah-Megaleh Amukot-.json":
        "Kabbalah-Other_Kabbalah_Works-Megalleh_Amukkot_on_Parashat_VaEtchanan-KRAKOW1637_Sefaria.jsonl",
    "Halakhah-Tur and Commentaries-Tur-.json": "Halakhah-Tur-Tur-Sefaria.jsonl",
}
TO_CHECK = {}   # old file -> why its match is uncertain; shown as S_CHECK
NOTES = {"Kabbalah-Megaleh Amukot-.json":
         "בחדש שמו 'על פרשת ואתחנן', אך הגודל דומה (147 אלף מול 134 אלף מילים) - ככל הנראה אותו חיבור"}
# Categories where the name matcher's 'check' guesses were looked at and are right.
CHECKED_RIGHT = {"Apocrypha"}
CHECKED_RIGHT_FILES = {
    "Kabbalah-Avodat HaKodesh (Gabbai)-.json", "Kabbalah-Ma'arekhet HaElokut-.json",
    "Kabbalah-Maggid Meisharim-.json", "Kabbalah-Or Neerav-.json",
    "Kabbalah-Recanati on the Torah-.json", "Kabbalah-Shaarei Tzedek-.json",
    "Kabbalah-The Beginning of Wisdom-.json", "Kabbalah-Zohar-.json",
}
# Old files that are a second copy of another old file.
DUPLICATES = {
    "Midrash-Aggadic Midrash-Midrash Tanchuma- (1).json": "Midrash-Aggadic Midrash-Midrash Tanchuma-.json",
    "Midrash-Aggadic Midrash-Midrash Tanchuma_Buber-.json": "Midrash-Aggadic Midrash-Midrash Tanchuma Buber-.json",
    "Midrash-Aggadic Midrash-Derech Eretz Zuta-.json": "Tanaitic-Minor Tractates-Tractate Derekh Eretz Zuta-.json",
}

CATEGORY_HE = {
    "Midrash": "מדרש", "Mishnah": "משנה", "Talmud": "תלמוד", "Tanaitic": "ספרות תנאית",
    "Tosefta": "תוספתא - מהדורת ליברמן ומפרשים", "Tanakh": 'תנ"ך', "Halakhah": "הלכה", "Liturgy": "תפילה",
    "Kabbalah": "קבלה", "Apocrypha": "ספרות בית שני", "Second_Temple": "ספרות בית שני",
    "Jewish_Thought": "מחשבה", "Musar": "מוסר", "Reference": "ספרי עזר", "Responsa": 'שו"ת',
}
KIND_HE = {"primary": "טקסט", "commentary": "פרשנות", "modern/translation": "מודרני / תרגום"}

S_SAME = "מקבילה"
S_RENAMED = "מקבילה - שם או קטגוריה שונים"
S_CHECK = "מקבילה אפשרית - לבדוק"
S_DUP = "כפילות בישן - אותו קובץ חדש"
S_NO_NEW = "אין מקבילה בקורפוס החדש"
S_NO_OLD = "אין מקבילה בקורפוס הישן"
CATEGORY_ORDER = ["מדרש", "ספרות תנאית", "משנה", "תלמוד", "תוספתא - מהדורת ליברמן ומפרשים", 'תנ"ך',
                  "ספרות בית שני", "קבלה", "הלכה", "תפילה", "מחשבה", "מוסר", 'שו"ת', "ספרי עזר"]
STATUS_ORDER = [S_SAME, S_RENAMED, S_CHECK, S_DUP, S_NO_NEW, S_NO_OLD]
FILLS = {S_SAME: "E3F1DF", S_RENAMED: "E3F1DF", S_CHECK: "FFF2CC", S_DUP: "EDEDED",
         S_NO_NEW: "F8D7D3", S_NO_OLD: "DCE6F2"}


def words_of(path):
    tokens = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                tokens.extend(json.loads(line).get("sentence", "").split())
            except json.JSONDecodeError:
                pass
    return tokens


def main():
    old_files = sorted(f for f in os.listdir(OLD_DIR) if f.endswith(".json"))
    names = {r["old_file"]: r for r in csv.DictReader(open("Inputs/old_vs_v7_names.csv", encoding="utf-8-sig"))}
    catalog = pd.read_excel("Inputs/corpus_catalog.xlsx", dtype=str).set_index("file")
    kinds = {r["v7_file"]: r["kind"] for r in csv.DictReader(open("Inputs/v7_outside_list.csv", encoding="utf-8-sig"))}
    old_table = pd.read_excel("Inputs/avg_text_length_per_file.xlsx", dtype=str).set_index("filename")

    def match_of(f):
        if f in MANUAL:
            return MANUAL[f]
        r = names.get(f)
        if not r or not r["v7_match"]:
            return None
        if r["status"] in ("exact", "likely"):
            return r["v7_match"]
        if r["status"] == "check" and (r["category"] in CHECKED_RIGHT or f in CHECKED_RIGHT_FILES):
            return r["v7_match"]
        return None

    rows, used = [], set()
    for f in old_files:
        new = match_of(f)
        in_old_exp = f in old_table.index and pd.notna(old_table.loc[f, "hierarchy"])
        if new:
            used.add(new)
            if f in DUPLICATES:
                status = S_DUP
            elif f in TO_CHECK:
                status = S_CHECK
            elif names.get(f, {}).get("status") == "exact" and f not in MANUAL:
                status = S_SAME
            else:
                status = S_RENAMED
        else:
            status = S_NO_NEW
        rows.append({"old": f, "new": new, "status": status, "in_old_exp": in_old_exp})
    for n in sorted(catalog.index):
        if n not in used:
            rows.append({"old": None, "new": n, "status": S_NO_OLD, "in_old_exp": False})

    print("counting words ...", flush=True)
    out = []
    for r in rows:
        old, new = r["old"], r["new"]
        cat = catalog.loc[new] if new else None
        ow = words_of(os.path.join(OLD_DIR, old)) if old else []
        nw = int(cat["words"]) if new else 0
        overlap = ""
        if old and new and r["status"] != S_DUP:
            oc = collections.Counter(ow)
            ncount = collections.Counter(words_of(os.path.join(NEW_DIR, new)))
            overlap = round(100 * sum((oc & ncount).values()) / max(len(ow), sum(ncount.values()), 1))
        prefix = (new or old).split("-")[0]
        note = []
        if old in TO_CHECK:
            note.append(TO_CHECK[old])
        if old in NOTES:
            note.append(NOTES[old])
        if old in DUPLICATES:
            note.append("עותק נוסף של %s בקורפוס הישן" % DUPLICATES[old])
        if old == "Midrash-Aggadic Midrash-Ein Yaakov-.json":
            note.append("בחדש יש רק מהדורת גליק, שמופיעה בשורה משלה")
        if overlap != "" and ow and (overlap < 85 or abs(nw / len(ow) - 1) > 0.08):
            note.append("הבדל ניכר בטקסט בין הקורפוסים")
        out.append({
            "קטגוריה": CATEGORY_HE.get(prefix, prefix),
            "שם החיבור": (cat["he_title"] if new and pd.notna(cat["he_title"]) else ""),
            "קובץ בקורפוס הישן": old or "— אין מקבילה —",
            "קובץ בקורפוס החדש": new or "— אין מקבילה —",
            "סטטוס": r["status"],
            # A duplicate row shares its new file with the original's row; count it from the old side only.
            "בניסויים": "כן" if (r["in_old_exp"] or (new and r["status"] != S_DUP
                                                     and cat["in_experiments"] == "True")) else "",
            "סוג (בחדש)": KIND_HE.get(kinds.get(new, "primary"), "") if new else "",
            "מהדורה (בחדש)": (cat["edition"] if new and pd.notna(cat["edition"]) else ""),
            "מילים בישן": len(ow) if old else None,
            "מילים בחדש": nw if new else None,
            "שינוי באורך %": round(100 * (nw / len(ow) - 1)) if old and new and ow else None,
            "חפיפת אוצר מילים %": overlap if overlap != "" else None,
            "הערות": "; ".join(note),
        })
    df = pd.DataFrame(out)
    # The experiment corpora first, and in each category the pairs before what has none,
    # so the sheet opens on file-against-file rows rather than on a column of absences.
    df["_cat"] = df["קטגוריה"].map(lambda c: CATEGORY_ORDER.index(c) if c in CATEGORY_ORDER
                                   else len(CATEGORY_ORDER))
    df["_status"] = df["סטטוס"].map(STATUS_ORDER.index)
    df = df.sort_values(["_cat", "_status", "בניסויים", "קובץ בקורפוס הישן", "קובץ בקורפוס החדש"],
                        ascending=[True, True, False, True, True]).drop(columns=["_cat", "_status"])

    summary = (df.assign(old=df["קובץ בקורפוס הישן"].ne("— אין מקבילה —"),
                         new=df["קובץ בקורפוס החדש"].ne("— אין מקבילה —"))
                 .groupby("קטגוריה")
                 .apply(lambda g: pd.Series({
                     "קבצים בישן": int(g["old"].sum()),
                     # Unique: a duplicate old file points at a new file already counted.
                     "קבצים בחדש": int(g.loc[g["new"], "קובץ בקורפוס החדש"].nunique()),
                     "עם מקבילה": int((g["old"] & g["new"] & g["סטטוס"].ne(S_DUP)).sum()),
                     "כפילויות בישן": int(g["סטטוס"].eq(S_DUP).sum()),
                     "רק בישן": int((g["old"] & ~g["new"]).sum()),
                     "רק בחדש": int((~g["old"] & g["new"]).sum()),
                     "בניסויים": int(g["בניסויים"].eq("כן").sum()),
                 }), include_groups=False)
                 .reset_index())
    total = summary.drop(columns="קטגוריה").sum()
    summary = pd.concat([summary, pd.DataFrame([{"קטגוריה": "סה\"כ", **total.to_dict()}])], ignore_index=True)

    exp = df[df["בניסויים"].eq("כן")]
    notes = pd.DataFrame({"": [
        "השוואה בין הקורפוס הישן (SefariasData/rabbinic, 529 קבצים) לקורפוס החדש (act_jsonl_v7, 1,417 קבצים).",
        "כל קובץ ישן וכל קובץ חדש מופיעים בדיוק פעם אחת בגיליון 'השוואה'.",
        "",
        "ספרי הניסויים: %d שורות. לכל ספרי הניסויים יש מקבילה בקורפוס החדש." % len(exp),
        "  דרך ארץ זוטא הופיע בישן פעמיים (תחת מדרש ותחת מסכתות קטנות); בחדש קובץ אחד.",
        "",
        "מה שאין לו מקבילה בחדש: כמעט כל ההלכה (בחדש רק הטור), כל ספרי התפילה, עין יעקב (המהדורה הרגילה),",
        "  ומספר חיבורי קבלה ומבואות לתלמוד. אף אחד מהם אינו בניסויים.",
        "מה שאין לו מקבילה בישן: בעיקר פרשנות (ראשונים על התלמוד, מפרשי תנ\"ך, מפרשי המדרש),",
        "  תוספתא במהדורת ליברמן, ותוספות טקסט: למשל אגדת בראשית, בראשית רבתי, מדרש שמואל, שכל טוב, ספרי זוטא.",
        "",
        "איך נקבעו המקבילות: לפי שם הקובץ. התאמות שאינן זהות בשמן נבדקו ידנית; התאמות שגויות של ההשוואה האוטומטית הוסרו.",
        "",
        "חפיפת אוצר מילים: איזה אחוז מהמילים (כולל חזרות) משותף לשני הקבצים.",
        "  גם טקסט זהה לא יגיע ל-100%: הקורפוס החדש מסיר גרשיים (הקב\"ה ← הקבה, ה' ← ה)",
        "  ואת הפניות הפסוקים שבתוך הטקסט. ערכים סביב 90% ומעלה הם הרגילים;",
        "  'הבדל ניכר בטקסט' מסומן מתחת ל-85% או בשינוי אורך של יותר מ-8% - שם ייתכן נוסח או מהדורה אחרים.",
    ]})

    with pd.ExcelWriter(OUT, engine="openpyxl") as xl:
        # The file-by-file sheet first: it is what the workbook is for, and a viewer that
        # opens on the first tab (Drive's preview) should show it.
        df.to_excel(xl, sheet_name="השוואה", index=False)
        summary.to_excel(xl, sheet_name="סיכום", index=False)
        notes.to_excel(xl, sheet_name="הסבר", index=False, header=False)
        for name, ws in xl.sheets.items():
            ws.sheet_view.rightToLeft = True
            for cell in ws[1]:
                cell.font = Font(bold=True)
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            for i, col in enumerate(ws.iter_cols(min_row=1, max_row=ws.max_row), start=1):
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:400])
                ws.column_dimensions[get_column_letter(i)].width = min(max(10, width + 2), 70)
        ws = xl.sheets["השוואה"]
        ws.freeze_panes = "C2"
        ws.auto_filter.ref = ws.dimensions
        status_col = list(df.columns).index("סטטוס") + 1
        for row in ws.iter_rows(min_row=2):
            fill = FILLS.get(row[status_col - 1].value)
            if fill:
                for c in row:
                    c.fill = PatternFill("solid", fgColor=fill)
        xl.sheets["הסבר"].column_dimensions["A"].width = 120

    print(OUT)
    print(summary.to_string(index=False))
    print(df["סטטוס"].value_counts().to_string())


if __name__ == "__main__":
    main()
