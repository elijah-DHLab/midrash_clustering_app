"""Train the FastText model on the act_jsonl_v7 corpus, with the recipe of fasttext_rabbinic_2026-09
(see MIDRASH/models/PROVENANCE.md): skipgram, dim 100, minn 3, maxn 6, minCount 5, epoch 10,
on the `sentence` field of every file - the same field the pipeline chunks and embeds.

Rows whose tokens are mostly non-Hebrew are skipped: the cleaning strips Latin letters, so an
English text such as Legends of the Jews arrives as a string of numbers.

    python train_fasttext_v7.py [--threads 4]
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import time

import fasttext

V7_DIR = "corpus_v7"
CORPUS_TXT = "cache/fasttext_v7_corpus.txt"
MODELS = "G:/My Drive/Haifa/MIDRASH/models"
NAME = "fasttext_v7_2026-10"
PARAMS = dict(model="skipgram", dim=100, minn=3, maxn=6, minCount=5, epoch=10)
HEBREW = re.compile(r"[\u05d0-\u05ea]")


def write_corpus():
    os.makedirs(os.path.dirname(CORPUS_TXT), exist_ok=True)
    sha, words, kept, skipped = hashlib.sha256(), 0, 0, 0
    files = sorted(f for f in os.listdir(V7_DIR) if f.endswith(".jsonl"))
    with open(CORPUS_TXT, "w", encoding="utf-8") as out:
        for f in files:
            for line in open(os.path.join(V7_DIR, f), encoding="utf-8"):
                if not line.strip():
                    continue
                tokens = json.loads(line).get("sentence", "").split()
                if not tokens or sum(bool(HEBREW.search(t)) for t in tokens) < len(tokens) / 2:
                    skipped += 1
                    continue
                text = " ".join(tokens) + "\n"
                out.write(text)
                sha.update(text.encode("utf-8"))
                words += len(tokens)
                kept += 1
    return dict(files=len(files), rows=kept, rows_skipped_non_hebrew=skipped,
                words=words, corpus_sha256=sha.hexdigest())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    args = ap.parse_args()

    corpus = write_corpus()
    print("corpus:", corpus, flush=True)
    t0 = time.time()
    model = fasttext.train_unsupervised(CORPUS_TXT, thread=args.threads, **PARAMS)
    minutes = (time.time() - t0) / 60
    path = f"{MODELS}/{NAME}.bin"
    model.save_model(path)
    meta = dict(name=NAME, trained=datetime.date.today().isoformat(), minutes=round(minutes, 1),
                threads=args.threads, params=PARAMS, vocabulary=len(model.words),
                training_corpus=f"all files of act_jsonl_v7 ({V7_DIR}), the `sentence` field", **corpus)
    with open(f"{MODELS}/{NAME}.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
