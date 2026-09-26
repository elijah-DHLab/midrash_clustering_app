"""Annotate and summarise the midrash measurements.

Separate from the paper's Results_Analysis by design: the paper's pipeline is not
touched. What it shares with the paper is the definition of the filter and of the
two measures; what differs is the corpus, the absence of a Ben-Yehuda footer, and
one correction.

The correction: measure_hebrew writes `word_position = i` from
`for i in range(1, len(words))`, which is a 0-based index into the word list, so
its row k holds the (k+1)-th word. pos_tag_hebrew writes `enumerate(tags,
start=1)`, so its row k holds the k-th word. Joining the two on word_position
therefore gives every word the tag of the word before it. Here the tag file's
position is decremented by one before the join, which aligns them.

Usage
-----
    python surprisal/analyse_midrash.py --work bereishit_rabbah --work pesikta_drk

Outputs (surprisal/analysis/): <work>_annotated.csv, unit_summary.csv
"""
import argparse
import pathlib
import sys

import pandas as pd
from nltk.corpus import stopwords

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = pathlib.Path(__file__).resolve().parent
MEAS = HERE / "measurements"
OUT = HERE / "analysis"

CONTENT_POS = ("NOUN", "VERB", "ADJ", "ADV")
MIN_WORDS = 20
HEB_ONLY = r"[^א-תa-zA-Z]"


def annotate(work, lemma_map, stop):
    meas = pd.read_csv(MEAS / ("%s_word_level.csv" % work),
                       encoding="utf-8-sig", low_memory=False)
    tags = pd.read_csv(MEAS / ("pos_%s.csv" % work), encoding="utf-8-sig")

    # The one-word correction, explained in the module docstring. Read off the
    # file rather than assumed: a tag file written before pos_tag_hebrew.py was
    # fixed starts at 1 and needs the shift, one written after starts at 0 and
    # must be left alone. Applying the shift blindly to a corrected file would
    # reintroduce the same misalignment in the other direction.
    first = int(tags["word_position"].min())
    if first == 1:
        tags["word_position"] = tags["word_position"] - 1
        print("    tags start at 1 (pre-fix tagger): shifted to 0")
    elif first != 0:
        raise SystemExit("tag positions start at %d; expected 0 or 1" % first)

    meas["text_id"] = meas["text_id"].astype(str)
    tags["text_id"] = tags["text_id"].astype(str)
    df = meas.merge(tags[["text_id", "word_position", "pos", "has_prefix"]],
                    on=["text_id", "word_position"], how="left")

    clean = df["actual_word"].fillna("").astype(str).str.replace(HEB_ONLY, "",
                                                                 regex=True)
    df["actual_word_clean"] = clean
    df["lemma"] = clean.map(lemma_map).fillna("")

    df["is_content_pos"] = df["pos"].isin(CONTENT_POS)
    df["is_stopword"] = clean.isin(stop) | df["lemma"].isin(stop)
    df["is_content"] = df["is_content_pos"] & ~df["is_stopword"]
    df["is_alpha"] = clean.ne("") & clean.str.isalpha()
    df["has_context"] = df["word_position"] > 3
    df["in_valid_range"] = (df["surprisal_bits"].between(0, 300)
                            & df["cos_dist_centroid"].between(0, 2))
    df["passes_filter"] = (df["is_content"] & df["is_alpha"]
                           & df["has_context"] & df["in_valid_range"])

    keep = df["passes_filter"]
    g = df[keep].groupby("text_id")
    df.loc[keep, "rank_surprisal"] = g["surprisal_bits"].rank(ascending=False)
    df.loc[keep, "rank_semantic"] = g["cos_dist_centroid"].rank(ascending=False)
    df["rank_gap"] = df["rank_surprisal"] - df["rank_semantic"]
    df["is_hl"] = df["rank_gap"].lt(0).fillna(False)   # surprising, close
    df["is_lh"] = df["rank_gap"].gt(0).fillna(False)   # expected, distant
    return df


def summarise(df, work, form):
    k = df[df["passes_filter"]]
    g = k.groupby("text_id")
    out = pd.DataFrame({
        "work": work,
        "form": form,
        "n_words": df.groupby("text_id").size(),
        "n_kept": g.size(),
        "median_surprisal": g["surprisal_bits"].median(),
        "median_cos": g["cos_dist_centroid"].median(),
        "rho": g.apply(lambda d: d["surprisal_bits"].corr(
            d["cos_dist_centroid"], method="spearman"), include_groups=False),
    })
    return out[out["n_kept"] >= MIN_WORDS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", action="append", required=True)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    stop = set(stopwords.words("hebrew"))
    f2l = pd.read_csv(MEAS / "form_to_lemma.csv")
    lemma_map = dict(zip(f2l["form"].astype(str), f2l["lemma"].astype(str)))

    manifest = pd.read_csv(HERE / "corpus" / "manifest.csv", encoding="utf-8-sig")
    forms = dict(zip(manifest["work"], manifest["form"]))

    summaries = []
    for work in args.work:
        df = annotate(work, lemma_map, stop)
        df.to_csv(OUT / ("%s_annotated.csv" % work), index=False,
                  encoding="utf-8-sig")
        s = summarise(df, work, forms.get(work, "?"))
        summaries.append(s)
        print("  %-18s %4d rows, %4d pass the filter, %d unit(s) over the floor"
              % (work, len(df), int(df["passes_filter"].sum()), len(s)))

    allsum = pd.concat(summaries)
    allsum.to_csv(OUT / "unit_summary.csv", encoding="utf-8-sig")
    print("\n%s" % allsum.to_string())
    print("\n  -> %s" % OUT)


if __name__ == "__main__":
    main()
