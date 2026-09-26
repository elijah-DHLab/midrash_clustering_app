"""Measure every unit of the midrash corpus, one work at a time.

Loads the analyser once --- the model and the lemmatiser are the expensive part,
not the text --- and walks the directories `build_corpus.py` wrote, appending
each unit's word-level rows to one CSV per work.

Written to be interrupted. Rows are flushed after every unit and a unit already
present in the output file is skipped on a rerun, so a run that dies at unit
1,900 of 2,798 resumes rather than restarts.

Start with `--limit 1`. The question that pass answers is not statistical: it is
whether DictaLM predicts rabbinic Hebrew at all sensibly, or whether it is
uniformly surprised and the measure would be reporting the model's training gap
rather than a property of the text. That is visible by eye in the first unit's
predictions and cannot be read off a median.

Usage
-----
    python surprisal/run_corpus.py --limit 1        # one unit per work, to look at
    python surprisal/run_corpus.py                  # everything
    python surprisal/run_corpus.py --work pesikta_drk

Outputs (surprisal/measurements/): <work>_word_level.csv
"""
import argparse
import csv
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
CORPUS = HERE / "corpus"
OUT = HERE / "measurements"

sys.path.insert(0, str(HERE))


def existing_units(path):
    """Unit ids already written, so a rerun resumes instead of repeating."""
    if not path.exists():
        return set()
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return {r["text_id"] for r in csv.DictReader(fh) if r.get("text_id")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="units per work; 0 means all")
    ap.add_argument("--work", action="append",
                    help="restrict to one work; repeatable")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    works = sorted(p.name for p in CORPUS.iterdir() if p.is_dir())
    if args.work:
        works = [w for w in works if w in args.work]
    if not works:
        print("  no works found under %s" % CORPUS)
        return

    from measure_hebrew_midrash import SemanticSurpriseAnalyser
    print("  loading the model ...", flush=True)
    analyser = SemanticSurpriseAnalyser()

    for work in works:
        units = sorted((CORPUS / work).glob("*.txt"))
        if args.limit:
            units = units[:args.limit]
        dst = OUT / ("%s_word_level.csv" % work)
        done = existing_units(dst)
        todo = [u for u in units if u.stem not in done]
        print("\n  === %s: %d units, %d already done, %d to do"
              % (work, len(units), len(units) - len(todo), len(todo)), flush=True)

        writer = None
        fh = open(dst, "a", encoding="utf-8-sig", newline="")
        try:
            for n, unit in enumerate(todo, 1):
                t0 = time.time()
                rows = analyser.analyze(unit.read_text(encoding="utf-8"))
                if not rows:
                    print("    %-40s no rows" % unit.stem, flush=True)
                    continue
                for r in rows:
                    r["text_id"] = unit.stem
                    r["work"] = work
                if writer is None:
                    fields = ["text_id", "work"] + [k for k in rows[0]
                                                    if k not in ("text_id", "work")]
                    writer = csv.DictWriter(fh, fieldnames=fields,
                                            extrasaction="ignore")
                    if not done and fh.tell() == 0:
                        writer.writeheader()
                writer.writerows(rows)
                fh.flush()
                os.fsync(fh.fileno())
                print("    [%d/%d] %-44s %4d words  %5.1fs"
                      % (n, len(todo), unit.stem[-44:], len(rows), time.time() - t0),
                      flush=True)
        finally:
            fh.close()

    print("\n  -> %s" % OUT)


if __name__ == "__main__":
    main()
