"""The embedding store: one file per (model, source file, chunk size).

The CLI pipeline caches embeddings under a hash of the *whole file selection*
(`pipeline_core._embedding_cache_key`), so a user who adds one book to the
selection pays for the entire recomputation. The app is interactive and its
users change the selection constantly, so the store here is keyed per source
file instead: any selection is then a concatenation of arrays already on disk,
and only a chunk size that has never been generated costs anything.

The first level is the model, because which FastText produced a vector is not a
detail — the two models here disagree about what the words mean. Keeping both
lets the same selection be read in either space and the readings compared.

Layout, under STORE_DIR:

    <model>/<chunk_size>/<safe filename>.npz   vec (float32 n x dim), chunk, chunk_number
    <model>/<chunk_size>/index.json            per file: n_chunks, words, hierarchy
    <model>/model.json                         label and provenance, shown in the app

Nothing here loads FastText. Generating a store is a separate step
(`webapp/build_store.py`), run where the model lives.
"""
import json
import os
import re
import pathlib

import numpy as np
import pandas as pd

STORE_DIR = pathlib.Path(
    os.environ.get("MIDRASH_STORE_DIR",
                   pathlib.Path(__file__).resolve().parents[2] / "embedding_store")
)
SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_name(filename: str) -> str:
    return SAFE.sub("_", filename)[:120]


def models() -> list:
    """Every model with a store, as {name, label, note, sizes}."""
    out = []
    if not STORE_DIR.exists():
        return out
    for d in sorted(STORE_DIR.iterdir()):
        if not d.is_dir():
            continue
        sizes = sorted(int(p.name) for p in d.iterdir()
                       if p.is_dir() and p.name.isdigit() and (p / "index.json").exists())
        if not sizes:
            continue
        meta = {}
        if (d / "model.json").exists():
            try:
                meta = json.loads((d / "model.json").read_text(encoding="utf-8"))
            except ValueError:
                meta = {}
        out.append({"name": d.name, "label": meta.get("label", d.name),
                    "note": meta.get("note", ""), "sizes": sizes})
    return out


def default_model() -> str:
    found = models()
    return found[0]["name"] if found else None


def available_chunk_sizes(model: str) -> list:
    for m in models():
        if m["name"] == model:
            return m["sizes"]
    return []


def index_for(model: str, chunk_size: int) -> dict:
    path = STORE_DIR / model / str(chunk_size) / "index.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def titles_for(model: str) -> dict:
    """Hebrew names per file, from `build_hebrew_titles.py`; empty if never built."""
    path = STORE_DIR / model / "titles.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}


def catalogue(model: str, chunk_size: int) -> pd.DataFrame:
    """One row per source file: its Hebrew name, its English one, corpus, size.

    The Hebrew name is what a reader of this material recognises — בראשית רבה, not
    "Aggadic Midrash / Midrash Rabbah / Bereishit Rabbah" — so it leads, and the
    English stays alongside for anyone working from the filenames.
    """
    names = titles_for(model)
    rows = []
    for filename, meta in index_for(model, chunk_size).items():
        parts = [p for p in filename.replace(".json", "").split("-") if p]
        name = names.get(filename, {})
        rows.append({
            "filename": filename,
            "corpus": parts[0] if parts else "",
            "corpus_he": name.get("corpus_he") or (parts[0] if parts else ""),
            # The catalogue path — אגדה / מדרש רבה / שיר השירים רבה — with the work
            # itself last, so the eye finds the title at the end of the line.
            "book": name.get("he_path") or name.get("he")
                    or (" / ".join(parts[1:]) if len(parts) > 1 else filename),
            "book_he": name.get("he", ""),
            "book_en": name.get("en") or (parts[-1] if parts else filename),
            "hierarchy": meta.get("hierarchy", ""),
            "n_chunks": meta.get("n_chunks", 0),
            "words": meta.get("words", 0),
        })
    df = pd.DataFrame(rows)
    return df.sort_values(["corpus", "book_en"]).reset_index(drop=True) if len(df) else df


def _hierarchy_tuple(value):
    """`12.1.1-5` sorts after `9.4.2`; compare component by component, numerically."""
    out = []
    for part in str(value).split("."):
        head = part.split("-")[0]
        out.append(int(head) if head.isdigit() else 0)
    return tuple(out)


def load_selection(filenames, chunk_size: int, model: str) -> pd.DataFrame:
    """The chunks of the selected files, in corpus-chronological order.

    Returns the frame the rest of the pipeline expects: filename, hierarchy,
    hierarchy_list, chunk, chunk_number, chronological_index, vec.
    """
    index = index_for(model, chunk_size)
    missing = [f for f in filenames if f not in index]
    if missing:
        raise KeyError("not in the %s store for chunk size %d: %s"
                       % (model, chunk_size, ", ".join(missing[:3])))

    ordered = sorted(filenames, key=lambda f: _hierarchy_tuple(index[f].get("hierarchy", "")))
    frames = []
    for filename in ordered:
        with np.load(STORE_DIR / model / str(chunk_size) / (safe_name(filename) + ".npz"),
                     allow_pickle=True) as z:
            frames.append(pd.DataFrame({
                "filename": filename,
                "hierarchy": index[filename].get("hierarchy", ""),
                "chunk": z["chunk"],
                "chunk_number": z["chunk_number"],
                "vec": list(z["vec"]),
            }))
    df = pd.concat(frames, ignore_index=True)
    df["hierarchy_list"] = df["hierarchy"].apply(_hierarchy_tuple)
    df["chronological_index"] = np.arange(len(df))
    return df
