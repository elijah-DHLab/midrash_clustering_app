"""
lemmatize_corpus_forms.py
─────────────────────────
Maps every distinct Hebrew surface form in the corpus to its lemma, using the
same lemmatizer the main pipeline already applies to the observed word when
computing semantic distance (ppl_semantic_beam.py, _cosine_distances: the
observed word is lemmatised in beam mode so that it is comparable to the
lemma-deduplicated prediction set).

Motivation: stopword filtering was being applied to surface forms, which misses
inflected and proclitic-bearing variants — `וגם` survives a stoplist containing
`גם`, `שלא` survives one containing `לא`. Filtering at the lemma level makes the
stoplist consistent with the level at which the measure itself operates.

Input : a one-column CSV of distinct forms (header `form`)
Output: CSV with columns form,lemma

Usage:
  python lemmatize_corpus_forms.py --in _unique_forms.csv --out form_to_lemma.csv
"""
import argparse
import pathlib
import sys

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    forms = pd.read_csv(args.inp)["form"].astype(str).tolist()
    print(f"forms to lemmatise: {len(forms)}", flush=True)

    from transformers import AutoModel, AutoTokenizer
    import torch

    print("loading dicta-il/dictabert-lex ...", flush=True)
    tok = AutoTokenizer.from_pretrained("dicta-il/dictabert-lex")
    model = AutoModel.from_pretrained("dicta-il/dictabert-lex", trust_remote_code=True)
    model.eval()

    lemmas = []
    with torch.no_grad():
        for i in range(0, len(forms), args.batch):
            chunk = forms[i:i + args.batch]
            try:
                res = model.predict(chunk, tok)
            except Exception as e:
                print(f"  batch {i} failed ({e}); falling back to per-item", flush=True)
                res = []
                for w in chunk:
                    try:
                        res.append(model.predict([w], tok)[0])
                    except Exception:
                        res.append(None)
            # predict() returns, per input string, a list of (surface, lemma) pairs.
            # A proclitic-bearing form comes back already resolved -- ('וגם', 'גם') --
            # so the last pair carries the content lemma.
            for w, r in zip(chunk, res):
                lem = w
                try:
                    if r:
                        for pair in r:
                            if len(pair) >= 2 and pair[1]:
                                lem = pair[1]
                except Exception:
                    pass
                lemmas.append(lem)
            if (i // args.batch) % 20 == 0:
                print(f"  {i + len(chunk)}/{len(forms)}", flush=True)

    out = pd.DataFrame({"form": forms, "lemma": lemmas})
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    n_changed = (out["form"] != out["lemma"]).sum()
    print(f"\nwrote {args.out}")
    print(f"  {n_changed} of {len(out)} forms ({n_changed/len(out)*100:.1f}%) map to a different lemma")


if __name__ == "__main__":
    main()
