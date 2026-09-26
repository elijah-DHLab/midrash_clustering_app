"""Attach Hebrew titles to the embedding store, from Sefaria's own table of contents.

The dump's `he_title` field is empty in every file, so the Hebrew names are taken from
Sefaria's index API and matched to the store by the English title the filename carries.
Writes `titles.json` next to each model's stores: filename -> {he, en, he_category}.

    python webapp/build_hebrew_titles.py
"""
import difflib
import json
import pathlib
import re
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from explorer.store import STORE_DIR   # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

API = "https://www.sefaria.org/api/index/"

# Sefaria files these under the minor tractates, where the loose match does not reach.
MANUAL = {
    "Derech Eretz Zuta": "דרך ארץ זוטא",
    "Tractate Derekh Eretz Rabbah": "מסכת דרך ארץ רבה",
}

# The Hebrew name of each top-level corpus, which the filename gives in English only.
CORPUS_HE = {
    "Midrash": "מדרש",
    "Mishnah": "משנה",
    "Talmud": "תלמוד",
    "Tanaitic": "ספרות תנאית",
    "Halakhah": "הלכה",
    "Kabbalah": "קבלה",
    "Liturgy": "סידור ותפילה",
    "Tanakh": "תנ\"ך",
    "Musar": "מוסר",
    "Philosophy": "מחשבה",
    "Responsa": "שו\"ת",
    "Chasidut": "חסידות",
}


def collect(node, out, he_cat=None):
    """Every titled entry in the tree, with the Hebrew category it sits under."""
    if isinstance(node, list):
        for item in node:
            collect(item, out, he_cat)
        return
    if not isinstance(node, dict):
        return
    cat = node.get("heCategory") or he_cat
    title, he = node.get("title"), node.get("heTitle")
    if title and he:
        out.setdefault(title.strip(), (he.strip(), cat))
    for key in ("contents", "nodes"):
        if key in node:
            collect(node[key], out, cat)


def normalise(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def main():
    req = urllib.request.Request(API, headers={"User-Agent": "midrash-explorer/1.0"})
    toc = json.loads(urllib.request.urlopen(req, timeout=120).read().decode("utf-8"))
    titles = {}
    collect(toc, titles)
    by_norm = {normalise(k): v for k, v in titles.items()}
    print("Sefaria entries: %d" % len(titles))

    # The filename is Corpus-SubCorpus-Book-.json; the book is the last non-empty part,
    # and the corpus the first. Both are matched against the index by a loose key.
    for model_dir in sorted(p for p in STORE_DIR.iterdir() if p.is_dir()):
        sizes = [p for p in model_dir.iterdir() if p.is_dir() and p.name.isdigit()]
        if not sizes:
            continue
        index = json.loads((sizes[0] / "index.json").read_text(encoding="utf-8"))
        mapping, missing, fuzzy = {}, [], []
        for filename in index:
            parts = [p for p in filename.replace(".json", "").split("-") if p.strip()]
            book = parts[-1] if parts else filename
            corpus = parts[0] if parts else ""
            hit = by_norm.get(normalise(book))
            if not hit:                      # e.g. "Midrash Tanchuma Buber" -> "Midrash Tanchuma"
                for cut in range(len(book.split()) - 1, 0, -1):
                    hit = by_norm.get(normalise(" ".join(book.split()[:cut])))
                    if hit:
                        break
            if not hit:
                # Transliteration differs between the dump and Sefaria — Bereishit against
                # Bereshit, Eichah against Eikhah. Match on spelling distance, and print
                # every such match so it can be checked by eye.
                close = difflib.get_close_matches(normalise(book), by_norm, n=1, cutoff=0.86)
                if close:
                    hit = by_norm[close[0]]
                    fuzzy.append((book, hit[0]))
            he = hit[0] if hit else MANUAL.get(book, "")
            if not he:
                missing.append(book)
            mapping[filename] = {"he": he, "en": book,
                                 "he_category": (hit[1] if hit else "") or "",
                                 "corpus_en": corpus,
                                 "corpus_he": CORPUS_HE.get(corpus, corpus)}
        (model_dir / "titles.json").write_text(
            json.dumps(mapping, ensure_ascii=False, indent=1), encoding="utf-8")
        found = sum(1 for v in mapping.values() if v["he"])
        print("%-10s %d of %d titles in Hebrew -> %s"
              % (model_dir.name, found, len(mapping), model_dir / "titles.json"))
        if fuzzy:
            print("   matched by spelling distance, check these: %s"
                  % "; ".join("%s = %s" % (en, he) for en, he in sorted(set(fuzzy))))
        if missing:
            print("   still unmatched: %s" % ", ".join(sorted(set(missing))))


if __name__ == "__main__":
    main()
