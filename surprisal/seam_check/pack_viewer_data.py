"""Pack the word-level LM results into compact JS data files for the viewer."""
import json, pathlib
import numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent
TITLES = {"shemot_rabbah": "שמות רבה", "bamidbar_rabbah": "במדבר רבה"}
for work, title in TITLES.items():
    w = pd.read_csv(HERE / ("%s_lm.csv" % work), encoding="utf-8")
    ch = w["chapter"].to_numpy()
    starts = [0] + [int(i) for i in np.flatnonzero(ch[1:] != ch[:-1]) + 1]
    data = {
        "title": title,
        "seamChapter": 15,
        "chapters": [[s, int(ch[s])] for s in starts],
        "W": w["word"].astype(str).tolist(),
        "S": np.clip(np.rint(w["surprisal_bits"] * 10), 0, 999).astype(int).tolist(),
        "E": np.clip(np.rint(w["entropy_bits"] * 10), 0, 999).astype(int).tolist(),
        "P": np.clip(np.rint(w["top1_prob"] * 100), 0, 100).astype(int).tolist(),
        "R": np.clip(w["rank_first_token"], 1, 999).astype(int).tolist(),
        "T": w["n_tokens"].astype(int).tolist(),
        "H": "".join(str(int(x)) for x in w["tanchuma_shared"]),
    }
    js = "window.MIDRASH_DATA=window.MIDRASH_DATA||{};window.MIDRASH_DATA[%s]=%s;" % (
        json.dumps(work), json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    out = HERE / "viewer" / ("data_%s.js" % work)
    out.write_text(js, encoding="utf-8")
    print(work, len(w), "words ->", out.name, "%.1f MB" % (out.stat().st_size / 1e6))
