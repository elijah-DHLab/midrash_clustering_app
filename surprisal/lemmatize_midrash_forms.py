"""Map every distinct Hebrew surface form in the midrash measurements to its lemma.

Same model and same output as the paper's lemmatize_hebrew_forms.py
(dicta-il/dictabert-lex, CSV of form,lemma), and the paper's file is left
untouched. This copy exists because the model repo's own `predict` does not run
under the transformers on the Haifa server:

    BertForLexPrediction.predict  ->  lex_parse_logits(inputs, ...)

passes the whole tokenizer output where the function's first parameter is
`input_ids`, so `input_ids[batch_idx]` indexes a BatchEncoding with an integer.
Older transformers allowed that; 5.6.2 raises KeyError: 0. The two lines below
call the same parsing function with the tensor it expects. Nothing else differs,
so the lemmas here are the ones the paper's model produces.

Usage
-----
    python surprisal/lemmatize_midrash_forms.py --in measurements/_unique_forms.csv \
                                                --out measurements/form_to_lemma.csv

Output: CSV with columns form,lemma. A form the model cannot resolve comes back
as [BLANK], exactly as in the paper's map.
"""
import argparse
import sys

import pandas as pd
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    forms = pd.read_csv(args.inp)["form"].astype(str).tolist()
    print("forms to lemmatise: %d" % len(forms), flush=True)

    from transformers import AutoModel, AutoTokenizer
    print("loading dicta-il/dictabert-lex ...", flush=True)
    tok = AutoTokenizer.from_pretrained("dicta-il/dictabert-lex")
    model = AutoModel.from_pretrained("dicta-il/dictabert-lex",
                                      trust_remote_code=True)
    model.eval()

    # The model's own module, reached through the loaded class rather than by
    # path, so this follows whichever snapshot from_pretrained resolved.
    lex_parse_logits = sys.modules[type(model).__module__].lex_parse_logits

    lemmas = []
    with torch.no_grad():
        for i in range(0, len(forms), args.batch):
            chunk = forms[i:i + args.batch]
            enc = tok(chunk, padding="longest", truncation=True,
                      return_tensors="pt")
            enc = {k: v.to(model.device) for k, v in enc.items()}
            logits = model.forward(**enc, return_dict=True).logits
            res = lex_parse_logits(enc["input_ids"], chunk, tok, logits)
            # predict() returns, per input string, a list of (surface, lemma)
            # pairs. A proclitic-bearing form comes back already resolved ---
            # ('וגם', 'גם') --- so the last pair carries the content lemma.
            for w, r in zip(chunk, res):
                lem = w
                for pair in (r or []):
                    if len(pair) >= 2 and pair[1]:
                        lem = pair[1]
                lemmas.append(lem)
            print("  %d/%d" % (min(i + args.batch, len(forms)), len(forms)),
                  flush=True)

    out = pd.DataFrame({"form": forms, "lemma": lemmas})
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    n = (out["form"] != out["lemma"]).sum()
    print("\nwrote %s" % args.out)
    print("  %d of %d forms (%.1f%%) map to a different lemma"
          % (n, len(out), 100.0 * n / max(len(out), 1)))
    print("  %d came back [BLANK]" % (out["lemma"] == "[BLANK]").sum())


if __name__ == "__main__":
    main()
