"""Builds the change-point atlas page from results/corpus_changepoints.json.

    python experiments/devarim_rabbah/build_changepoints_page.py
"""
import json
import pathlib

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = json.loads((HERE / "results" / "corpus_changepoints.json").read_text(encoding="utf-8"))
OUT = HERE / "results" / "changepoints_atlas.html"

catalog = pd.read_excel(ROOT / "Inputs" / "corpus_catalog.xlsx", dtype=str).set_index("file")
GENRES = [("classic", "מדרש אגדה קלאסי"), ("late", "מדרש אגדה מאוחר"), ("tannaitic", "מדרשי תנאים"),
          ("mishnah_tosefta", "משנה ותוספתא"), ("yerushalmi", "תלמוד ירושלמי"), ("bavli", "תלמוד בבלי")]

works = []
for f, w in DATA.items():
    row = catalog.loc[f] if f in catalog.index else None
    works.append({
        "f": f, "t": w["title"], "g": "classic" if w["genre"] == "target" else w["genre"],
        "target": w["genre"] == "target", "n": w["words"],
        "h": str(row["hierarchy"]) if row is not None else "",
        "u": w["units"], "c": [[c["word"], c["unit"], c["before"], c["after"]] for c in w["cps"]],
    })
payload = json.dumps({"works": works, "genres": GENRES}, ensure_ascii=False, separators=(",", ":"))

html = (HERE / "changepoints_atlas.template.html").read_text(encoding="utf-8").replace("/*DATA*/null", payload)
OUT.write_text(html, encoding="utf-8")
print(OUT, len(works), "works,", sum(len(w["c"]) for w in works), "change points,", round(len(html) / 1024), "KB")
