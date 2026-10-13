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

The app reads `Midrash/embedding_store/`, which is **not in git**. Two ways to get it:

1. **Copy it** from Drive: `MIDRASH/embedding_store/` → `Midrash/embedding_store/`.
2. **Rebuild it**, which needs the FastText model and the corpus. Since October 2026 the
   corpus is Sefaria `act_jsonl_v7` in `Midrash/corpus_v7/` (from
   `s3://elasticsearch-back-c/jsons/sefaria-jsons/act_jsonl_v7/`, documented in
   `Inputs/v7_README.html`), and the input table is `Inputs/corpus_catalog.xlsx`
   (`Inputs/build_catalog.py`). Every catalog row with a `hierarchy` is built:

```
python train_fasttext_v7.py               # the model, once (CPU only; ~3 min on 24 cores)
python webapp/build_store.py --sizes 25 50 75 100 --label v7 --title "Sefaria v7 (2026-10)"
```

`build_store.py` writes `titles.json` from the catalog (Hebrew name, Sefaria categories,
scholarly category, composition id, edition). For the older stores, built from the old
dump, the Hebrew names came from Sefaria's API through `build_hebrew_titles.py`.

A v7 store also keeps, per chunk, where it sits in its book: `start_ref` and `end_ref`
(the v7 `location` without the file's own prefix, e.g. `52_10` = parasha 52, section 10),
`ref_spans` (which words of the chunk come from which location, `1–44: 1_1 | 45–50: 1_2`)
and `start_segment` (the v7 segment id, to find the source row). v7 records only the
first Sefaria section of each ~175-word row, so a place is exact to the row, not the section.

The ends of each chunk are also kept word-exact, two ways:
- in the v7 rows: `start_segment` / `start_segment_word` and `end_segment` / `end_segment_word`
  (segment id, and the word within that row's `sentence`, 1-based);
- in Sefaria's text unit: `start_unit` / `start_unit_word` and `end_unit` / `end_unit_word`.
  v7 cuts a long unit into rows whose `from_word` continues the previous row's `to_word`;
  the unit is named by the location of its first row and the word is `from_word` + offset.
  The location alone cannot be trusted here: in 26 of the experiment books (Pesikta Rabbati,
  the Yalkut, Tanhuma Buber, ...) each part of a split unit also gets a part-number suffix,
  so `1_1_1_2_3` is word 501 of unit `1_1`, not a section `1_1_1_2_3`.

Layout: `embedding_store/<model>/<chunk size>/<file>.npz`, plus `index.json` per size,
`model.json` and `titles.json` per model. Keyed per source file, so any selection of
texts is a concatenation of arrays already on disk and only a new chunk size costs
anything.

## The models

| store | trained on | vocabulary |
|---|---|---|
| `v7` | **the default.** All 1,417 files of act_jsonl_v7, 52.6M words, October 2026 | 234,289 |
| `rabbinic` | the corpus itself, 529 files, 33.5M words, September 2026 | 174,228 |
| `zohar` | the Zohar — the model every result before September 2026 used | 17,627 |

All are in Drive under `MIDRASH/models/`, with `PROVENANCE.md` beside them. The two
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
- **Downloads** —
  - `chunks.csv`: every chunk with its book (Hebrew name, Sefaria categories, scholarly
    category, composition id, edition), `hierarchy`, its place in the book, its cluster,
    its similarity to its own and the nearest other cluster, density, segment and text;
  - `clusters.csv`: one row per cluster, with its top texts and n-grams;
  - `segments.csv`: one row per stretch between change points, from where to where in the
    text, which books it holds and which clusters opened it;
  - `change_points.csv`: each boundary with the place just before and after it;
  - `figure.html`.

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
