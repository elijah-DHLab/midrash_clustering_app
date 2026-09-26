"""ממשק Streamlit לתצוגה הכרונולוגית של הקלאסטרים.

מאפשר לבחור פלח מהקורפוס (checklist היררכי לפי קורפוס/ספר, במקום עריכת
עמודות ה-Experiment בקובץ המטא-דאטה), לכוונן את הפרמטרים ה"זולים" (אורך
קטע, k), ולקבל בחזרה את הוויזואליזציה האינטראקטיבית — לצפייה ולהורדה.
זיהוי נקודות שינוי (CPD) הוסר זמנית מהממשק (הלוגיקה עצמה עדיין קיימת
ב-cluster_visualization_report.py) כדי להתמקד קודם בוויזואליזציה הבסיסית.
החלפת מודל ה-embedding עצמו היא מחוץ לתחום כאן — ראו
writing/pipeline_methods.md — זהו שינוי מבני, לא פרמטר.

הרצה מקומית:
    streamlit run app.py
"""

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import pipeline_core
from cluster_visualization_report import (
    add_cluster_ctfidf_ngrams,
    build_figure,
    compute_chunk_cluster_affinities,
    compute_density_original,
    merge_small_groups,
    prepare_dataframe,
)

st.set_page_config(page_title="מדרש — סייר קלאסטרים כרונולוגי", layout="wide")

GROUPING_LEVEL_LABELS = {1: "קורפוס", 2: "תת-קורפוס", 3: "ספר"}

PRESET_LABELS_HE = {
    "None (start empty)": "ללא (התחלה מרשימה ריקה)",
    "Experiment 1": "ניסוי 1: ספרות רבנית",
    "Experiment 2": "ניסוי 2: ספרות מדרשית",
    "Experiment 3": "ניסוי 3: תנחומא (עם ילקוט)",
    "Experiment 4": "ניסוי 4: תנחומא",
    "Experiment 5": "ניסוי 5: ספרות מדרשית ללא אנתולוגיות",
    "Experiment 6": "ניסוי 6: ספרות מוקדמת (הלכתית, אמוראית, תנחומאית)",
}

RTL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Rubik:wght@300;400;500;600;700&display=swap');

html, body, .stApp, .stApp *:not([data-testid="stIconMaterial"]) {
    font-family: 'Rubik', 'Heebo', 'Segoe UI', sans-serif !important;
}

.stApp {
    direction: rtl;
}

section[data-testid="stSidebar"] > div {
    direction: rtl;
    text-align: right;
}
div[data-testid="stAppViewContainer"] .main .block-container {
    direction: rtl;
    text-align: right;
}

/* headings/captions/labels right-aligned; numbers stay LTR-safe automatically */
h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, .stCaption {
    text-align: right;
}

/* keep the chart and the data grid left-to-right internally so axes,
   numbers, and cell content aren't mirrored */
.js-plotly-plot, [data-testid="stDataFrame"] {
    direction: ltr;
    text-align: left;
}

/* checkboxes/expanders: keep the tick box on the right, matching RTL reading order */
.stCheckbox > label {
    flex-direction: row-reverse;
    justify-content: flex-end;
}
[data-testid="stExpander"] summary {
    direction: rtl;
    text-align: right;
}
</style>
"""


@st.cache_data(show_spinner="טוען מטא-דאטה...")
def load_metadata_cached() -> pd.DataFrame:
    return pipeline_core.load_metadata()


@st.cache_resource(show_spinner="טוען מודל FastText (רק בהרצה הראשונה)...")
def get_model():
    return pipeline_core.load_fasttext_model()


def main():
    st.markdown(RTL_CSS, unsafe_allow_html=True)

    st.title("מדרש — סייר קלאסטרים כרונולוגי")
    st.caption(
        "בחרו פלח מהקורפוס, הריצו עליו embedding וקיבוץ לקלאסטרים, ובדקו היכן "
        "ההרכב הסגנוני-תוכני שלו משתנה לאורך הציר הכרונולוגי."
    )

    metadata = load_metadata_cached()

    # ---------------------------------------------------------------
    # Sidebar: בחירת נתונים
    # ---------------------------------------------------------------
    st.sidebar.header("1. נתונים")

    preset_options = ["None (start empty)"] + list(pipeline_core.EXPERIMENT_OPTIONS.keys())
    preset = st.sidebar.selectbox(
        "מערך נתונים מוכן מראש",
        preset_options,
        index=1,
        format_func=lambda p: PRESET_LABELS_HE.get(p, p),
    )

    if preset == "None (start empty)":
        preset_files = set()
    else:
        preset_files = set(pipeline_core.files_for_experiment(metadata, preset))

    st.sidebar.caption("פתחו קורפוס כדי לבחור בתוכו את הספרים הרצויים:")
    selected_files = []
    for corpus in sorted(metadata["level1"].unique()):
        corpus_df = metadata[metadata["level1"] == corpus].sort_values("level3")
        filenames = corpus_df["filename"].tolist()
        n_selected_now = sum(
            st.session_state.get(f"chk_{preset}_{fn}", fn in preset_files) for fn in filenames
        )
        with st.sidebar.expander(f"{corpus} ({n_selected_now}/{len(filenames)} נבחרים)"):
            for _, row in corpus_df.iterrows():
                fn = row["filename"]
                label = row["level3"] or fn
                checked = st.checkbox(label, value=fn in preset_files, key=f"chk_{preset}_{fn}")
                if checked:
                    selected_files.append(fn)

    # ---------------------------------------------------------------
    # Sidebar: פרמטרים
    # ---------------------------------------------------------------
    st.sidebar.header("2. פרמטרים")
    chunk_size = st.sidebar.number_input("אורך קטע (מילים)", min_value=10, max_value=500, value=50, step=10)
    k = st.sidebar.number_input("מספר קלאסטרים (k)", min_value=2, max_value=50, value=10, step=1)
    apply_smoothing = st.sidebar.checkbox("החלת החלקה (smoothing)", value=False)
    grouping_level = st.sidebar.selectbox(
        "רמת קיבוץ (תוויות ציר והרכב)",
        [1, 2, 3],
        index=1,
        format_func=lambda level: GROUPING_LEVEL_LABELS[level],
    )

    if selected_files:
        st.sidebar.caption(f"נבחרו {len(selected_files)} קבצים.")
        if len(selected_files) > 50:
            st.sidebar.warning(
                "בחירה גדולה — הבנייה וה-embedding עשויים לקחת זמן רב (עד כמה דקות ואף יותר) בהרצה הראשונה."
            )

    with st.sidebar.expander("הצעת k (סריקת elbow / silhouette)"):
        st.caption(
            "מריץ KMeans שוב ושוב על טווח k, על הקטעים שכבר קודדו בהרצה האחרונה "
            "למטה. איטי יותר — אופציונלי."
        )
        k_min = st.number_input("k מינימלי", min_value=2, max_value=49, value=2, key="k_min")
        k_max = st.number_input("k מקסימלי", min_value=k_min + 1, max_value=50, value=15, key="k_max")
        if st.button("הרץ סריקת k"):
            if "df_embedded" not in st.session_state:
                st.warning("הריצו את הפייפליין פעם אחת קודם (למטה) כדי שיהיו קטעים מקודדים לסריקה.")
            else:
                _render_k_sweep(st.session_state["df_embedded"], k_min, k_max)

    run_clicked = st.sidebar.button("הרצת הפייפליין", type="primary")

    # ---------------------------------------------------------------
    # Pipeline
    # ---------------------------------------------------------------
    if run_clicked:
        if not selected_files:
            st.sidebar.error("בחרו לפחות קובץ אחד.")
        else:
            _run_pipeline(
                metadata=metadata,
                selected_files=selected_files,
                chunk_size=chunk_size,
                k=k,
                apply_smoothing=apply_smoothing,
                grouping_level=grouping_level,
            )

    if "result" in st.session_state:
        _render_result(st.session_state["result"])


def _run_pipeline(
    metadata,
    selected_files,
    chunk_size,
    k,
    apply_smoothing,
    grouping_level,
):
    embed_key = (tuple(sorted(selected_files)), chunk_size)
    if st.session_state.get("embed_key") != embed_key:
        df_embedded = pipeline_core.load_cached_embeddings(selected_files, chunk_size)
        if df_embedded is not None:
            st.toast("נטענו embeddings קיימים מהמטמון על הדיסק — אין צורך לקדד מחדש.")
        else:
            with st.spinner(f"בונה ומקודד קטעים עבור {len(selected_files)} קבצים..."):
                file_rows = metadata[metadata["filename"].isin(selected_files)]
                df_chunks = pipeline_core.build_chunks(file_rows, chunk_size=chunk_size)
                if df_chunks.empty:
                    st.error("לא נוצרו קטעים — ייתכן שהקבצים שנבחרו חסרים בתיקיית הנתונים.")
                    return
                model = get_model()
                df_embedded = pipeline_core.embed_chunks(df_chunks, model)
                pipeline_core.save_cached_embeddings(selected_files, chunk_size, df_embedded)
        st.session_state["embed_key"] = embed_key
        st.session_state["df_embedded"] = df_embedded
    else:
        df_embedded = st.session_state["df_embedded"]

    cluster_key = embed_key + (k, apply_smoothing)
    if st.session_state.get("cluster_key") != cluster_key:
        with st.spinner(f"מקבץ {len(df_embedded)} קטעים ל-{k} קלאסטרים..."):
            df_clustered = pipeline_core.cluster_chunks(df_embedded, k=k, apply_smoothing=apply_smoothing)
            df_clustered = pipeline_core.stringify_vectors(df_clustered)
        st.session_state["cluster_key"] = cluster_key
        st.session_state["df_clustered"] = df_clustered
    else:
        df_clustered = st.session_state["df_clustered"]

    with st.spinner("בונה ויזואליזציה..."):
        df_viz = prepare_dataframe(df_clustered, grouping_level=grouping_level)
        df_viz = merge_small_groups(df_viz)
        df_viz = compute_density_original(df_viz)
        df_viz = compute_chunk_cluster_affinities(df_viz)
        df_viz = add_cluster_ctfidf_ngrams(df_viz, ngram_n=3, top_k=20)

        fig = build_figure(df_viz, show_text_boundaries=True, change_points=None)

        cluster_table = _build_cluster_summary_table(df_viz)

    st.session_state["result"] = {
        "fig": fig,
        "df_viz": df_viz,
        "cluster_table": cluster_table,
        "n_chunks": len(df_viz),
        "k": k,
    }


def _build_cluster_summary_table(df_viz: pd.DataFrame, top_n_ngrams: int = 8) -> pd.DataFrame:
    """One row per cluster: size, share of the corpus, and its top
    characteristic n-grams (already computed by add_cluster_ctfidf_ngrams) —
    the "what is each cluster, and how big is it" info, independent of CPD."""
    sizes = df_viz.groupby("scluster").size()
    total = int(sizes.sum())
    rows = []
    for cid in sorted(df_viz["scluster"].unique()):
        ngrams_raw = df_viz.loc[df_viz["scluster"] == cid, "cluster_top_ngrams"].iloc[0]
        top_terms = str(ngrams_raw).split("<br>")[:top_n_ngrams]
        rows.append(
            {
                "קלאסטר": int(cid),
                "מספר קטעים": int(sizes[cid]),
                "% מהסה\"כ": round(100.0 * sizes[cid] / total, 1),
                "מילות מפתח מובילות": ", ".join(top_terms),
            }
        )
    return pd.DataFrame(rows)


def _render_result(result):
    st.subheader(f"תצוגה כרונולוגית של קלאסטרים — {result['n_chunks']:,} קטעים, k={result['k']}")
    st.caption("גלגלת עכבר להתקרבות/התרחקות, גרירה להזזה, לחיצה כפולה לאיפוס התצוגה.")

    # Embed the exact same standalone HTML the download button produces, in
    # an iframe — this renders identically to opening that file directly in
    # a browser (same Plotly toolbar, zoom-box, and reset-axes behavior),
    # instead of Streamlit's own plotly wrapper, whose container stretching
    # interacts poorly with the page's RTL layout and made the chart look
    # small and out of proportion to its axes.
    html_str = result["fig"].to_html(
        include_plotlyjs="cdn",
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "doubleClick": "reset+autosize",
        },
    )
    fig_height = result["fig"].layout.height or 720
    components.html(html_str, height=fig_height + 50, scrolling=True)

    st.download_button(
        "הורדת הוויזואליזציה (HTML)",
        data=html_str.encode("utf-8"),
        file_name="chronological_cluster_view.html",
        mime="text/html",
    )

    st.subheader("סיכום קלאסטרים")
    st.dataframe(result["cluster_table"], use_container_width=True, hide_index=True)
    st.download_button(
        "הורדת סיכום הקלאסטרים (CSV)",
        data=result["cluster_table"].to_csv(index=False).encode("utf-8-sig"),
        file_name="cluster_summary.csv",
        mime="text/csv",
    )

    with st.expander("הורדת נתוני קטעים מלאים"):
        include_vec = st.checkbox("כלול וקטורי embedding (קובץ גדול)", value=False)
        df_export = result["df_viz"].copy()
        if "vec" in df_export.columns and not include_vec:
            df_export = df_export.drop(columns=["vec"])
        st.download_button(
            "הורדת טבלת הקטעים (CSV)",
            data=df_export.to_csv(index=False).encode("utf-8-sig"),
            file_name="chunks.csv",
            mime="text/csv",
        )


def _render_k_sweep(df_embedded, k_min, k_max):
    import numpy as np
    import plotly.graph_objects as go
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    features = np.stack(df_embedded["vec"].values)
    k_range = list(range(k_min, k_max + 1))
    inertias, silhouettes = [], []

    progress = st.progress(0.0)
    for i, kk in enumerate(k_range):
        km = KMeans(n_clusters=kk, random_state=42, n_init="auto")
        labels = km.fit_predict(features)
        inertias.append(km.inertia_)
        sample_size = min(2000, len(features))
        silhouettes.append(silhouette_score(features, labels, sample_size=sample_size, random_state=42))
        progress.progress((i + 1) / len(k_range))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k_range, y=inertias, mode="lines+markers", name="אינרציה", yaxis="y1"))
    fig.add_trace(go.Scatter(x=k_range, y=silhouettes, mode="lines+markers", name="סילואט", yaxis="y2"))
    fig.update_layout(
        xaxis=dict(title="k"),
        yaxis=dict(title="אינרציה"),
        yaxis2=dict(title="סילואט", overlaying="y", side="right"),
        legend=dict(orientation="h"),
        height=350,
        margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
    best_k = k_range[int(np.argmax(silhouettes))]
    st.caption(f"ה-k הטוב ביותר לפי מדד הסילואט: {best_k} (ציון={max(silhouettes):.4f})")


if __name__ == "__main__":
    main()
