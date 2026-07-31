from pathlib import Path
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="FindMyLab",
    page_icon="🔎",
    layout="wide",
)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
# Using the app.py location prevents path errors when Streamlit is launched
# from a different working directory.
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "Abstracts" / "df_paper.txt"
EMBEDDING_PATH = BASE_DIR / "Embeddings" / "MPnet" / "MPnet_embeddings.npy"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


# ------------------------------------------------------------
# Cached loaders
# ------------------------------------------------------------
@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error(f"df_paper.txt file not found: {DATA_PATH}")
        st.stop()

    if not EMBEDDING_PATH.exists():
        st.error(f"Embedding file not found: {EMBEDDING_PATH}")
        st.stop()

    df_paper = pd.read_csv(DATA_PATH, sep="\t")
    embeddings = np.load(EMBEDDING_PATH)

    return df_paper, embeddings


@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    return model


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------
def sanitize_filename(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9가-힣_-]+", "_", text.strip())
    text = text.strip("_")
    return text[:50] if text else "query"


def make_professor_boxplot(
    keyword: str,
    top_n: int = 15,
    year_filter: int = 2015,
    min_papers: int = 10,
    aggregation: str = "median",
):

    print(f">> Searching for keyword: {keyword}")
    df_paper, embeddings_general = load_data()
    model_general = load_model()

    query_embedding = model_general.encode(
        [keyword],
        show_progress_bar=False,
    )

    similarities = cosine_similarity(query_embedding, embeddings_general).flatten()

    df = df_paper.copy()
    df["similarity"] = similarities

    df_year = df.loc[df["Year"] >= year_filter, :].copy()

    if aggregation == "mean":
        grouped = (
            df_year.groupby("Professor")
            .agg(
                similarity=("similarity", "mean"),
                n_papers=("Professor", "count"),
            )
            .sort_values("similarity", ascending=False)
        )
    else:
        grouped = (
            df_year.groupby("Professor")
            .agg(
                similarity=("similarity", "median"),
                n_papers=("Professor", "count"),
            )
            .sort_values("similarity", ascending=False)
        )

    grouped = grouped[grouped["n_papers"] >= min_papers]
    top_professors = grouped.head(top_n).index.to_list()

    if not top_professors:
        return None, grouped, df.iloc[np.argsort(similarities)[::-1][:top_n]].copy()

    df_plot = df_year[df_year["Professor"].isin(top_professors)].copy()

    hover_columns = [
        col
        for col in ["Professor", "title", "Year", "journal", "PMID"]
        if col in df_plot.columns
    ]

    fig = px.box(
        df_plot,
        x="similarity",
        y="Professor",
        category_orders={"Professor": top_professors},
        points="all",
        color="Professor",
        notched=False,
        hover_data=hover_columns,
        template="plotly_white",
    )

    fig.update_traces(
        jitter=0.5,
        pointpos=0,
        marker=dict(size=5),
    )

    # Force a light Plotly theme so the plot remains readable even when
    # Windows, the browser, or Streamlit is using dark mode.
    fig.update_layout(
        height=650,
        width=1100,
        showlegend=False,
        title=f"Top {top_n} Professors by Similarity to: {keyword}",
        xaxis_title="Cosine Similarity",
        yaxis_title="Professor",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        title_font=dict(color="black"),
        yaxis={
            "categoryorder": "array",
            "categoryarray": top_professors,
            "autorange": "reversed",
        },
        hovermode="closest",
        hoverdistance=-1,
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 0.95)",
            font_color="black",
            align="left",
            font_size=12,
            font_family="Arial",
            bordercolor="black",
        ),
    )

    fig.update_xaxes(
        range=[0, 1],
        showgrid=True,
        gridcolor="lightgray",
        zeroline=False,
        linecolor="black",
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
    )

    fig.update_yaxes(
        showgrid=False,
        linecolor="black",
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
    )

    # top_papers = df.iloc[np.argsort(similarities)[::-1][:top_n]].copy()
    # top_papers = df_plot.iloc[np.argsort(similarities)[::-1][:top_n]].copy()
    top_papers = df_plot.copy()
    top_papers.sort_values("similarity", ascending=False, inplace=True)
    top_papers.reset_index(inplace=True)

    return fig, grouped.head(top_n), top_papers


def make_downloadable_html(fig) -> str:
    """Create a self-contained HTML file with an explicitly white background."""
    plot_html = fig.to_html(
        full_html=False,
        include_plotlyjs=True,
    )

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>FindMyLab Plot</title>
    <style>
        html, body {{
            background-color: white !important;
            color: black !important;
            margin: 0;
            padding: 20px;
        }}
    </style>
</head>
<body>
    {plot_html}
</body>
</html>
"""
    return html


# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.title("🔎 FindMyLab")
st.write(
    "Enter research keywords or a short research-interest paragraph. "
    "FindMyLab will rank professors based on semantic similarity to their publications."
)

with st.sidebar:
    st.header("Settings")

    keyword = st.text_area(
        "Research interest: keywords or a short paragraph (<300 words)",
        #        value="cancer, genomics, diagnosis",
        # value="cancer, machine learning, genomics, transcriptomics, computational biology",
        value="cancer, machine learning, genome, transcriptome, computational biology",
        height=220,
    )

    top_n = st.slider("Top N professors", 5, 50, 10)
    year_filter = st.number_input(
        "Use papers from year ≥",
        min_value=1900,
        max_value=2100,
        value=2015,
    )
    min_papers = st.number_input(
        "Minimum papers per professor",
        min_value=1,
        max_value=100,
        value=10,
    )
    aggregation = st.selectbox("Professor-level summary metric", ["median", "mean"])

    run_button = st.button("Generate plot", type="primary")


if run_button:
    if not keyword.strip():
        st.warning("Please enter a keyword or research-interest text.")
    else:
        with st.spinner("Computing similarities and generating plot..."):
            fig, professor_table, top_papers = make_professor_boxplot(
                keyword=keyword,
                top_n=top_n,
                year_filter=year_filter,
                min_papers=min_papers,
                aggregation=aggregation,
            )

        if fig is None:
            st.warning(
                "No professors matched the current filters. "
                "Try lowering the minimum paper count or changing the year filter."
            )
        else:
            st.subheader("Professor similarity plot")

            # Important: theme=None prevents Streamlit's dark theme from overriding
            # the explicitly white Plotly layout.
            st.plotly_chart(fig, use_container_width=True, theme=None)

            html = make_downloadable_html(fig)

            filename_keyword = sanitize_filename(keyword)
            today = datetime.now().strftime("%Y%m%d")
            output_filename = f"FindMyLab_{filename_keyword}_{today}.html"

            st.download_button(
                label="Download Plotly HTML",
                data=html.encode("utf-8"),
                file_name=output_filename,
                mime="text/html",
            )

        st.subheader("Top professors")
        # remove index
        st.dataframe(
            professor_table,
            width=500,
            # hide_index=True,
        )

        # use_container_width=True

        st.subheader("Top matching papers")
        columns_to_show = [
            col
            for col in [
                "Date",
                "Year",
                "PMID",
                "Professor",
                "title",
                "journal",
                "abstract",
                "similarity",
            ]
            if col in top_papers.columns
        ]
        st.dataframe(
            top_papers[columns_to_show],
            use_container_width=True,
            height=600,
            # hide_index=True,
        )
