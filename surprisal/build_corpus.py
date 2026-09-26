"""Extract four midrashic works into the corpus layout the Hebrew pipeline reads.

The question this corpus is built for is whether the statistical/semantic
decoupling measure separates exegetical from homiletical midrash --- the
distinction scholarship draws between Bereishit Rabbah, which proceeds verse by
verse, and Vayikra Rabbah, Pesikta de-Rav Kahana and Tanchuma, which are built
in homiletical units. Both sides are aggadic, amoraic and on the Torah, so
subject matter, register and period are held roughly constant and what differs
is the form of the discourse. A lexical clustering cannot see that difference;
the point of the experiment is to find out whether this measure can.

What is stripped, and why it matters
------------------------------------
The Sefaria dump offers two text fields and neither is usable as it stands.
`sentence` removes the brackets around a scriptural citation but leaves its
content in the running text, so a unit reads `... אחותי כלה שה"ש ה א ר' עזריה`
with the reference embedded mid-sentence. `orig_sentence` keeps the brackets,
and is therefore the field to start from: the bracketed spans can be removed
whole. They are citations and section letters --- of 4,615 distinct spans
sampled from two works, the common ones are `שם`, `תהלים שם`, `בראשית כט, ז`
and bare section letters --- that is, apparatus rather than text.

Two further cleanings are not cosmetic:

    nikkud          present in `orig_sentence` and not in the model's usual
                    input. Stripped here rather than downstream, so the corpus
                    on disk is what was measured.

    piska headers   every unit of Pesikta de-Rav Kahana opens `פסקא <n> אות
                    <n>`, and no other work does. A fixed, highly predictable
                    string carried by one side of the comparison and not the
                    other is exactly the kind of artifact that produces a genre
                    difference out of nothing, so it goes.

`references` is not used. It is empty for all 304 units of Pesikta de-Rav
Kahana although that work cites Scripture more densely than the others --- the
field was never populated for it. The citation count written to the manifest is
the number of bracketed spans removed, which is measured the same way in every
work.

What a unit is
--------------
One record, one file, as the pipeline expects one text per file. Units shorter
than MIN_WORDS after cleaning are dropped: a within-text rank correlation over a
handful of content words is noise.

Usage
-----
    python surprisal/build_corpus.py

Outputs (surprisal/corpus/): one directory per work, one .txt per unit, and
manifest.csv with the work, its form, the unit's location, its length and the
number of citations removed from it.
"""
import csv
import json
import pathlib
import re

SEFARIA = pathlib.Path("G:/My Drive/Haifa/MIDRASH/SefariasData/rabbinic")
OUT = pathlib.Path(__file__).resolve().parent / "corpus"
MIN_WORDS = 60

WORKS = [
    ("bereishit_rabbah", "exegetical",
     "Midrash-Aggadic Midrash-Midrash Rabbah-Bereishit Rabbah-.json"),
    ("vayikra_rabbah", "homiletical",
     "Midrash-Aggadic Midrash-Midrash Rabbah-Vayikra Rabbah-.json"),
    ("pesikta_drk", "homiletical",
     "Midrash-Aggadic Midrash-Pesikta D'Rav Kahanna-.json"),
    ("tanchuma", "homiletical",
     "Midrash-Aggadic Midrash-Midrash Tanchuma-.json"),
]

PAREN = re.compile(r"\([^)]{1,60}\)")
NIKKUD = re.compile(r"[\u0591-\u05C7]")
BIDI = re.compile(r"[\u200e\u200f\u202a-\u202e]")
PISKA = re.compile(r"^\s*פסקא\s+\S+\s+אות\s+\S+\s*")
SAFE = re.compile(r"[^A-Za-z0-9_]+")


def clean(raw):
    """Return the unit's text and the number of citation spans removed."""
    text = str(raw)
    n_citations = len(PAREN.findall(text))
    text = PAREN.sub(" ", text)
    text = NIKKUD.sub("", text)
    text = BIDI.sub("", text)
    text = PISKA.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text, n_citations


def unit_id(location, seen):
    """A filename from the record's location, unique within the work."""
    base = SAFE.sub("_", str(location)).strip("_")
    base = base[-60:].lstrip("_") or "unit"
    n = seen.get(base, 0)
    seen[base] = n + 1
    return base if n == 0 else "%s__%d" % (base, n + 1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    for work, form, filename in WORKS:
        src = SEFARIA / filename
        if not src.exists():
            print("  %-20s MISSING: %s" % (work, filename))
            continue

        target = OUT / work
        target.mkdir(exist_ok=True)
        for old in target.glob("*.txt"):
            old.unlink()

        kept = dropped = 0
        seen = {}
        with open(src, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                text, n_cit = clean(r.get("orig_sentence", ""))
                n = len(text.split())
                if n < MIN_WORDS:
                    dropped += 1
                    continue
                uid = unit_id(r.get("location", ""), seen)
                (target / ("%s.txt" % uid)).write_text(text, encoding="utf-8")
                rows.append({"work": work, "form": form, "unit_id": uid,
                             "location": r.get("location", ""),
                             "n_words": n, "n_citations": n_cit})
                kept += 1

        print("  %-20s %-12s kept %5d   dropped %5d" % (work, form, kept, dropped))

    dst = OUT / "manifest.csv"
    with open(dst, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["work", "form", "unit_id", "location",
                                           "n_words", "n_citations"])
        w.writeheader()
        w.writerows(rows)
    print("\n  %s units -> %s" % (format(len(rows), ","), OUT))


if __name__ == "__main__":
    main()
