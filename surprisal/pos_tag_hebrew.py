"""
POS-tag Hebrew texts with dicta-il/dictabert-morph and align tags back onto
the word_position indexing used by ppl_semantic_beam.py (split_to_words:
whitespace split + hyphen split), so word-level surprisal/cos_dist CSVs can
be filtered to content words (NOUN/VERB/ADJ/ADV), matching the English
report_en_qwen_nobeam.py methodology.

dictabert-morph tags whole sentences, not isolated words, so each text is
split into sentences, tagged, and the per-sentence token lists are
concatenated back in order. Alignment is verified by comparing the
concatenated token count against split_to_words(text); texts that don't
match exactly are logged and skipped rather than silently mis-aligned.

Runs on CPU by default (--device cpu) so it doesn't contend with a GPU job
already running the main pipeline.

Usage:
  python pos_tag_he.py --texts-dir DIR --out-csv OUT.csv [--device cpu|cuda]

Output CSV columns: text_id, word_position, pos, has_prefix (word carries a fused
grammatical proclitic -- o/h/b/l/k/m/sh -- per dictabert-morph's own 'prefixes'
field; used for the no-prefix robustness check)

word_position is a 0-BASED index into split_to_words(text), which is the
convention the measurement stages write: measure_hebrew.py and
measure_english.py both run `for i in range(start, len(words))` and store
`word_position = i` alongside `actual_word = words[i]`. Emitting 1-based
positions here -- as this file did until the alignment was checked against the
source texts -- shifts every tag onto the following word once the two are merged
on word_position, because the measurement skips the opening word or words that
have no left context. The tag for words[0] is written with word_position 0 and
simply finds no measurement row to join.
"""
import argparse
import pathlib
import re
import sys
import unicodedata

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Exactly the range measure_hebrew.py strips. The wider [֑-ׇ]
# also removes the maqaf (U+05BE), which the measurement keeps: a standalone
# maqaf used as a dash is a word there and vanishes here, so the tagger's
# word list gets shorter than the measurement's and every tag after that
# point lands on the wrong word.
# The revision whose remote code matches the published weights. On the current
# main revision the checkpoint's flat head names (pos_cls.weight) no longer match
# the code's nested ones (morph.pos_cls.weight), so every classification head is
# randomly initialised, from_pretrained reports it only as a MISSING-keys notice,
# and predict() returns confident, well-formed, meaningless tags. Pinning is the
# fix; check_heads_loaded below is the alarm if the pin is ever lost.
MORPH_MODEL = "dicta-il/dictabert-morph"
MORPH_REVISION = "292587e0fd1d172fa0f487bae5b8befeab8bf568"

HEBREW_COMBINING_MARKS = re.compile(
    r"[֑-ֽֿׁ-ׂׄ-ׇׅ]"
)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+|\n+")


def strip_niqqud(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = HEBREW_COMBINING_MARKS.sub("", text)
    return unicodedata.normalize("NFKC", text)


def split_to_words(text: str) -> list[str]:
    words = []
    for w in text.split():
        if "-" in w:
            words.extend(part for part in w.split("-") if part)
        else:
            words.append(w)
    return [w for w in words if w.strip()]


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in SENT_SPLIT_RE.split(text)]
    return [p for p in parts if p]


def _norm_chars(s: str) -> str:
    """Alphanumeric-only content, for matching a coarse word against the model's
    own (finer, punctuation-splitting) tokenization of that same word."""
    return re.sub(r"[^\w]", "", s, flags=re.UNICODE)


def tag_text(text: str, model, tokenizer, batch_size: int = 16) -> list[tuple[str, bool]] | None:
    """Returns a list of (pos, has_prefix) aligned to split_to_words(text), or None
    if misaligned. has_prefix = the token dictabert-morph assigned this word's POS
    from also carries a non-empty 'prefixes' list (an attached grammatical proclitic
    like ו-/ה-/ב-/ל-/כ-/מ-/ש- fused onto the word) -- used for the no-prefix
    robustness check, since a fused proclitic inflates summed subword surprisal
    without being a real lexical choice (the Hebrew analogue of the English
    single-token-word robustness restriction).

    dictabert-morph's own tokenizer further splits on internal punctuation
    (e.g. Hebrew acronyms like שד"ל -> ['שד', '"', 'ל']), so token count does not
    equal whitespace-word count. Alignment instead greedily consumes model tokens
    per whitespace word, matching on normalized (alnum-only) character content,
    and takes the first non-PUNCT sub-token's POS/prefixes as that word's tag.
    Any word that can't be matched exactly aborts alignment for the whole text
    (skip it rather than risk silently mis-aligning positions).
    """
    clean = strip_niqqud(text)
    coarse_words = clean.split()  # whitespace-only, whole text, same order sentences preserve

    sentences = split_sentences(clean)
    all_tokens: list[dict] = []
    for i in range(0, len(sentences), batch_size):
        batch = [s for s in sentences[i:i + batch_size] if s.split()]
        if not batch:
            continue
        results = model.predict(batch, tokenizer)
        for r in results:
            all_tokens.extend(r["tokens"])

    coarse_tags: list[tuple[str, bool]] = []
    ti = 0
    for cw in coarse_words:
        target = _norm_chars(cw)
        if not target:
            coarse_tags.append(("X", False))
            continue
        acc = ""
        acc_pos = None
        acc_prefix = False
        while ti < len(all_tokens) and len(acc) < len(target):
            tok = all_tokens[ti]
            tok_chars = _norm_chars(tok["token"])
            acc += tok_chars
            if acc_pos is None and tok.get("pos") != "PUNCT" and tok_chars:
                acc_pos = tok.get("pos")
                acc_prefix = bool(tok.get("prefixes"))
            ti += 1
        if acc != target:
            return None  # misalignment -- bail out for this whole text
        coarse_tags.append((acc_pos or "PUNCT", acc_prefix))

    leftover = all_tokens[ti:]
    if any(_norm_chars(t["token"]) for t in leftover):
        return None  # leftover tokens carry real content -- something split unexpectedly
    # else: leftover is pure trailing punctuation (e.g. a final "."/"\"") with no
    # more words to attach to -- harmless, not a real misalignment

    # Expand hyphenated coarse words to match split_to_words()'s granularity.
    final_tags: list[tuple[str, bool]] = []
    for cw, tag in zip(coarse_words, coarse_tags):
        if "-" in cw:
            parts = [p for p in cw.split("-") if p]
            final_tags.extend([tag] * len(parts))
        else:
            final_tags.append(tag)

    expected_words = split_to_words(clean)
    if len(final_tags) != len(expected_words):
        return None
    return final_tags


def check_heads_loaded(model):
    """Refuse to tag with randomly initialised classifier heads.

    A head that never loaded keeps its initialisation: the bias is exactly zero
    throughout. Real trained biases are not. The tags such a model produces look
    entirely normal -- one valid POS per token -- which is why this is checked
    rather than eyeballed.
    """
    sd = model.state_dict()
    biases = [k for k in sd if k.endswith("pos_cls.bias") or k.endswith("prefix_cls.bias")]
    if not biases:
        raise SystemExit("no POS/prefix head found in the model; wrong architecture?")
    dead = [k for k in biases if float(sd[k].float().std()) == 0.0]
    if dead:
        raise SystemExit(
            "classifier head(s) %s are still at their initialisation -- the "
            "checkpoint did not load into them. The tags would be random. "
            "Check MORPH_REVISION." % ", ".join(dead))
    print("  heads loaded: %s" % ", ".join("%s std=%.4f" % (k, sd[k].float().std())
                                           for k in biases), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ids", default="", help="comma-separated list of text IDs (without .txt) to restrict to")
    args = ap.parse_args()

    texts_dir = pathlib.Path(args.texts_dir)
    files = sorted(texts_dir.glob("*.txt"))
    if args.ids:
        wanted = set(args.ids.split(","))
        files = [f for f in files if f.stem in wanted]

    print(f"Loading {MORPH_MODEL} @ {MORPH_REVISION[:8]} on {args.device}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MORPH_MODEL, revision=MORPH_REVISION)
    model = AutoModel.from_pretrained(MORPH_MODEL, trust_remote_code=True,
                                      revision=MORPH_REVISION)
    model.eval()
    model.to(args.device)
    check_heads_loaded(model)

    rows = []
    n_ok, n_fail = 0, 0
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        with torch.no_grad():
            tags = tag_text(text, model, tokenizer)
        if tags is None:
            n_fail += 1
            print(f"  [SKIP] {f.name}: word-count mismatch (alignment failed)", flush=True)
            continue
        n_ok += 1
        # start=0: word_position indexes split_to_words(text) from zero, the same
        # convention measure_hebrew.py writes. See the module docstring.
        for i, (pos, has_prefix) in enumerate(tags, start=0):
            rows.append({"text_id": f.stem, "word_position": i, "pos": pos, "has_prefix": has_prefix})
        print(f"  [OK] {f.name}: {len(tags)} words tagged", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"\nDone. {n_ok} texts tagged, {n_fail} skipped (alignment failure).", flush=True)
    print(f"Wrote {len(out)} rows -> {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
