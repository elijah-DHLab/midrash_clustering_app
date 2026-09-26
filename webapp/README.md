# סייר הקלאסטרים המדרשי — Midrash cluster explorer

A small Django app for exploring the chronological clustering of rabbinic literature:
pick a slice of the corpus, cluster it, look at it, take the tables away.

The heavy part — turning text into vectors — is done once and kept on disk, so the app
never loads FastText and a run costs only one KMeans over the chunks you selected.

## Running it

```
cd Midrash/webapp
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
..\.venv\Scripts\python.exe manage.py runserver
```

Then open <http://127.0.0.1:8000>.

## The embedding store

The app reads `Midrash/embedding_store/`, which is **not in git** — it is 438 MB of
`.npz`. Two ways to get it:

1. **Copy it** from Drive: `MIDRASH/embedding_store/` → `Midrash/embedding_store/`.
2. **Rebuild it**, which needs the FastText model and the Sefaria dump:

```
python webapp/build_store.py --sizes 25 50 75 100 \
    --files-from experiment_files.txt \
    --model "G:/My Drive/Haifa/MIDRASH/models/fasttext_rabbinic_2026-09.bin" \
    --label rabbinic --title "Rabbinic corpus (2026-09)"
python webapp/build_hebrew_titles.py      # Hebrew names, from Sefaria's index API
```

Layout: `embedding_store/<model>/<chunk size>/<file>.npz`, plus `index.json` per size,
`model.json` and `titles.json` per model. Keyed per source file, so any selection of
texts is a concatenation of arrays already on disk and only a new chunk size costs
anything.

## The two models

| store | trained on | vocabulary |
|---|---|---|
| `rabbinic` | the corpus itself, 529 files, 33.5M words, September 2026 | 174,228 |
| `zohar` | the Zohar — the model every result before September 2026 used | 17,627 |

Both are in Drive under `MIDRASH/models/`, with `PROVENANCE.md` beside them. The two
spaces disagree: on one selection only 8.9% of chunks land in a correspondingly
numbered cluster. Any figure should say which model produced it.

## What the app does

- **Texts** — pick by corpus, Hebrew names throughout, with the English filename on hover.
- **Resolution** — chunk size (25 / 50 / 75 / 100 words), k, and the label level of the axis.
- **Smoothing** — averages each chunk with its neighbours before clustering.
- **Change points** — PELT over the block composition. The useful penalty range is about
  0.5 to 2; above 5 it returns nothing.
- **Which k?** — sweeps k from 2 to 20 and reports the elbow and the silhouette, with a
  reading of what the selection supports.
- **Downloads** — `chunks.csv` (chunk → cluster), `figure.html`, `change_points.csv`.

Runs are cached on disk by their full parameter set, including the model, so repeating a
run returns immediately. The cache lives in `webapp/_runs/` and is not in git; delete
`webapp/_runs/index.json` to force recomputation.

## Files

| | |
|---|---|
| `explorer/store.py` | reads the embedding store |
| `explorer/views.py` | the page, the run, the k sweep, the cache |
| `explorer/jobs.py` | background runs, stages, and the progress estimate |
| `build_store.py` | generates a store from a FastText model |
| `build_hebrew_titles.py` | Hebrew names from Sefaria's index API |
| `import_cache.py` | converts an old pipeline cache pickle into a store |

The app imports `pipeline_core` and `cluster_visualization_report` from the project root
rather than duplicating them, so the command-line pipeline and the app stay in step.
