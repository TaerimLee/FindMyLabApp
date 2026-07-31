"""
FindMyLab core analysis library (v2).

This module is a refactor of `5.App_Preparation_v1.py` into reusable,
importable, side-effect-free functions so that a multipage Streamlit app
(and the `validate_v1_v2.py` script) can call into the exact same analysis
logic.

IMPORTANT: This file does not modify or depend on `5.App_Preparation_v1.py`.
Every function here is a faithful, line-by-line port of the corresponding
block in v1 so that outputs (tables) are identical, and figures carry the
same underlying data/meaning (see `validate_v1_v2.py` for the checks).
"""

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------
# Paths (resolved relative to this file so the app works regardless
# of the working directory it is launched from)
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Abstracts" / "df_paper.txt"
EMBEDDING_PATH = BASE_DIR / "Embeddings" / "abstract_embeddings.npy"
FACULTY_LIST_PATH = BASE_DIR / "ILS_Faculty_List.txt"
FIGURES_DIR = BASE_DIR / "Figures"

MODEL_NAME = "all-mpnet-base-v2"

# Max number of professors that can be selected for the publication-trend
# facet plot (Figure 3), to keep the layout readable.
MAX_TREND_PROFESSORS = 8

# Defaults, matching the "Input_*" values hardcoded in v1.
DEFAULT_TREND_PROFESSORS = [
    "Andrew Ellington",
    "Edward M. Marcotte",
    "Jason McLellan",
    "Jennifer Maynard",
]
DEFAULT_NETWORK_PROFESSOR = "Claus Wilke"
DEFAULT_RESEARCH_INTEREST = (
    "protein language models, deep learning, structural biology, drug discovery"
)
DEFAULT_YEAR_FILTER = 2020
DEFAULT_TOPN_PAPERS = 20
DEFAULT_TOPN_PIS = 10
DEFAULT_MIN_PAPERS = 5
DEFAULT_TOPN_PAPERS_AGGREGATE = 10
DEFAULT_SUMMARY_SCORE = "median"


# ==============================================================
# Data loading
# ==============================================================
def load_paper_data(data_path=DATA_PATH):
    """Load the abstracts table. Mirrors v1's `df_paper = pd.read_csv(...)`."""
    df_paper = pd.read_csv(data_path, sep="\t")
    return df_paper


def load_embeddings(embedding_path=EMBEDDING_PATH):
    """Load precomputed abstract embeddings. Mirrors v1's `np.load(...)`."""
    return np.load(embedding_path)


def load_faculty_list(path=FACULTY_LIST_PATH):
    """Load the ILS faculty roster, one name per line."""
    with open(path) as f:
        names = [line.strip() for line in f if line.strip()]
    return names


def load_sentence_model(model_name=MODEL_NAME):
    """Load the SentenceTransformer model used to embed research-interest text."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


# ==============================================================
# Section A: Basic Stats
# ==============================================================
def get_professor_paper_counts(df_paper):
    """Series of paper counts per professor, sorted descending (v1 Figure 1 data)."""
    return df_paper["Professor"].value_counts()


def make_fig1_bar_papers_per_professor(df_paper):
    """Static bar plot: number of papers per professor. Identical to v1 Figure 1."""
    counts = get_professor_paper_counts(df_paper)

    fig, ax = plt.subplots(figsize=(20, 5))
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    ax.set_xticks(range(len(counts.index)))
    ax.set_xticklabels(counts.index, rotation=90, size=8, ha="center")
    ax.set_title("Number of Papers per Professor")
    fig.tight_layout()

    return fig, counts


def get_paper_year_professor_heatmap_data(df_paper):
    """Crosstab of paper counts: rows=Year, cols=Professor (v1 Figure 2 data)."""
    counts = get_professor_paper_counts(df_paper)
    df_plot_heatmap = pd.crosstab(df_paper["Year"], df_paper["Professor"])
    df_plot_heatmap = df_plot_heatmap.loc[:, counts.index]
    return df_plot_heatmap


def make_fig2_heatmap_interactive(df_plot_heatmap):
    """
    Interactive Plotly heatmap (same data as v1 Figure 2, but interactive so
    exact paper counts are visible on hover, per user preference).
    """
    z = df_plot_heatmap.replace(0, np.nan).values

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=df_plot_heatmap.columns.tolist(),
            y=[str(y) for y in df_plot_heatmap.index.tolist()],
            colorscale="Reds",
            hovertemplate="Professor: %{x}<br>Year: %{y}<br>Papers: %{z}<extra></extra>",
            colorbar=dict(title="Papers"),
        )
    )
    fig.update_layout(
        title="Number of Papers per Professor by Year",
        xaxis_title="Professor",
        yaxis_title="Year",
        height=550,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
    )
    fig.update_xaxes(tickfont=dict(size=8))
    return fig


def get_publication_trend_data(df_paper, target_professors):
    """Per-professor, per-year paper counts, filtered to `target_professors` (v1 Figure 3 data)."""
    df_plot_line = (
        df_paper.groupby(["Professor", "Year"])
        .size()
        .reset_index(name="count")
        .sort_values(["Professor", "Year"])
    )
    df_plot_line = df_plot_line.loc[df_plot_line["Professor"].isin(target_professors)]
    return df_plot_line


def make_fig3_publication_trend(df_paper, target_professors):
    """
    Static FacetGrid line plot of publication trend per professor.
    Logic identical to v1 Figure 3. `target_professors` is user-selectable,
    capped at MAX_TREND_PROFESSORS.
    """
    if len(target_professors) == 0:
        raise ValueError("Select at least one professor for the trend plot.")
    if len(target_professors) > MAX_TREND_PROFESSORS:
        raise ValueError(
            f"Select at most {MAX_TREND_PROFESSORS} professors for the trend plot."
        )

    df_plot_line = get_publication_trend_data(df_paper, target_professors)

    def plot_publication_trend(data, **kwargs):
        historical = data.loc[data["Year"] < 2026]

        sns.lineplot(
            data=historical,
            x="Year",
            y="count",
            linewidth=1.5,
        )
        sns.scatterplot(
            data=historical,
            x="Year",
            y="count",
            s=25,
        )

    col_wrap = min(4, len(target_professors))

    g = sns.FacetGrid(
        df_plot_line,
        col="Professor",
        col_wrap=col_wrap,
        sharey=True,
        sharex=True,
        col_order=target_professors,
    )

    g.map_dataframe(plot_publication_trend)
    g.set_titles("{col_name}")
    g.set_axis_labels("Year", "Number of Papers")

    plt.subplots_adjust(top=0.9)

    for ax in g.axes.flatten():
        ax.tick_params(labelbottom=True)

        for y in [10, 20, 30]:
            ax.axhline(
                y=y, color="gray", linestyle="--", alpha=0.3, linewidth=1, zorder=0
            )

        for x in [2000, 2010, 2020]:
            ax.axvline(
                x=x, color="gray", linestyle="--", alpha=0.3, linewidth=1, zorder=0
            )

    g.figure.tight_layout()

    return g.figure, df_plot_line


# ==============================================================
# Section B: Coauthorship Network
# ==============================================================
def build_coauthorship_data(df_paper):
    """
    Build the co-authorship graph edge weights and per-professor paper counts.
    Logic identical to the data-prep portion of v1's `co_authorship_network`.
    """
    df = df_paper.copy()
    df["Year"] = df["Year"].astype(int)

    prof_paper_counts = df.groupby("Professor")["PMID"].nunique().to_dict()

    pmid_groups = df.groupby("PMID")["Professor"].apply(lambda x: list(set(x)))

    edge_weights = {}
    for professors in pmid_groups:
        if len(professors) >= 2:
            unique_profs = sorted(professors)
            for p1, p2 in combinations(unique_profs, 2):
                pair = (p1, p2)
                edge_weights[pair] = edge_weights.get(pair, 0) + 1

    G = nx.Graph()
    for (p1, p2), w in edge_weights.items():
        G.add_edge(p1, p2, weight=w)

    return G, prof_paper_counts, edge_weights


def make_coauthorship_figure(
    df_paper, target_pi="All", background=True, fig_size=(13, 13)
):
    """
    Draws the co-authorship network for a given target PI. Logic and visual
    styling identical to v1's `co_authorship_network` function; only
    difference is this returns a Figure object instead of relying on the
    implicit pyplot state, and it returns the graph/edge data for validation.

    Parameters
    ----------
    target_pi : str
        The name of the target PI. Use "All" to include all PIs.
    background : bool
        Whether to include background nodes/edges in the visualization.
    fig_size : tuple
        The size of the figure for the network plot.
    """
    G, prof_paper_counts, edge_weights = build_coauthorship_data(df_paper)

    if target_pi == "All":
        target_neighbors = set()
        highlighted_nodes = set(G.nodes)
    else:
        if target_pi in G:
            target_neighbors = set(G.neighbors(target_pi))
            highlighted_nodes = target_neighbors | {target_pi}
        else:
            target_neighbors = set()
            highlighted_nodes = set()

    fig = plt.figure(figsize=fig_size, dpi=100)

    pos = nx.spring_layout(G, k=0.8, iterations=100, seed=42)

    nodes_to_draw = (
        list(G.nodes) if background or target_pi == "All" else list(highlighted_nodes)
    )

    node_sizes = [15 + (prof_paper_counts.get(node, 0) * 1) for node in nodes_to_draw]

    node_colors = []
    node_alphas = []
    for node in nodes_to_draw:
        if target_pi == "All":
            node_colors.append("skyblue")
            node_alphas.append(0.8)
        elif node == target_pi:
            node_colors.append("orangered")
            node_alphas.append(1.0)
        elif node in target_neighbors:
            node_colors.append("skyblue")
            node_alphas.append(0.9)
        else:
            node_colors.append("lightgray")
            node_alphas.append(0.2)

    edges_to_draw = []
    edge_widths = []
    edge_colors = []
    edge_alphas = []

    for u, v in G.edges:
        w = G[u][v]["weight"]
        is_highlighted_edge = target_pi == "All" or u == target_pi or v == target_pi

        if is_highlighted_edge:
            edges_to_draw.append((u, v))
            edge_widths.append(1.0 + (w * 1.2))
            edge_colors.append("gray")
            edge_alphas.append(0.6 if target_pi == "All" else 0.8)
        elif background:
            edges_to_draw.append((u, v))
            edge_widths.append(0.5)
            edge_colors.append("gray")
            edge_alphas.append(0.2)

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=nodes_to_draw,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=node_alphas,
    )

    for edge, w, c, a in zip(edges_to_draw, edge_widths, edge_colors, edge_alphas):
        nx.draw_networkx_edges(G, pos, edgelist=[edge], width=w, edge_color=c, alpha=a)

    pos_labels = {
        node: (coords[0] + 0.005, coords[1] + 0.005) for node, coords in pos.items()
    }

    for node in nodes_to_draw:
        x, y = pos_labels[node]
        is_highlighted = node in highlighted_nodes

        if target_pi == "All":
            text_color = "black"
            text_size = 8
            is_bold = False
            rotation_val = 0
        else:
            text_color = (
                "orangered"
                if node == target_pi
                else ("black" if is_highlighted else "gray")
            )
            text_size = 10 if node == target_pi else (8 if is_highlighted else 6)
            is_bold = node == target_pi
            rotation_val = 15 if is_highlighted else 0

        plt.text(
            x,
            y,
            s=node,
            fontsize=text_size,
            fontfamily="sans-serif",
            color=text_color,
            alpha=1.0 if is_highlighted else 0.25,
            weight="bold" if is_bold else "normal",
            rotation=rotation_val,
            rotation_mode="anchor",
        )

    title_suffix = "Overall Map" if target_pi == "All" else f"{target_pi}"
    plt.title(
        f"Co-authorship Network ({title_suffix})",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )

    plt.axis("off")
    plt.tight_layout()

    return fig, G, prof_paper_counts, edge_weights


# ==============================================================
# Section C: Professor Prioritization
# ==============================================================
def compute_similarity(df_paper, embeddings, interest_text, model):
    """Embed `interest_text` and compute cosine similarity to every paper (v1 logic)."""
    query_embedding = model.encode([interest_text], show_progress_bar=False)
    similarities = cosine_similarity(query_embedding, embeddings).flatten()

    df = df_paper.copy()
    df["similarity"] = similarities
    return df


def get_table1_top_papers(
    df, year_filter=DEFAULT_YEAR_FILTER, top_n=DEFAULT_TOPN_PAPERS
):
    """Table 1: top-N most similar papers overall, regardless of professor."""
    top_papers = (
        df.loc[df.Year >= year_filter]
        .sort_values(by="similarity", ascending=False)
        .head(top_n)
        .copy()
    )
    top_papers.reset_index(drop=True, inplace=True)

    table1 = top_papers[
        ["Date", "PMID", "Professor", "similarity", "title", "journal"]
    ].copy()
    return table1


def get_table2_grouped_by_professor(
    df,
    year_filter=DEFAULT_YEAR_FILTER,
    top_n_pis=DEFAULT_TOPN_PIS,
    min_papers=DEFAULT_MIN_PAPERS,
    top_n_aggregate=DEFAULT_TOPN_PAPERS_AGGREGATE,
    summary_score=DEFAULT_SUMMARY_SCORE,
):
    """Table 2: similarity aggregated per professor, plus intermediate frames needed by Tables 3/4 & Figure 6."""
    df_paper_year = df.loc[df["Year"] >= year_filter].copy()

    professor_paper_counts = (
        df_paper_year.groupby("Professor").size().rename("n_papers")
    )

    eligible_professors = professor_paper_counts[
        professor_paper_counts >= min_papers
    ].index

    df_paper_eligible = df_paper_year.loc[
        df_paper_year["Professor"].isin(eligible_professors)
    ].copy()

    df_top_papers_by_professor = (
        df_paper_eligible.sort_values(
            ["Professor", "similarity"],
            ascending=[True, False],
        )
        .groupby("Professor", group_keys=False)
        .head(top_n_aggregate)
    )

    df_paper_grouped_by_professor = df_top_papers_by_professor.groupby("Professor").agg(
        similarity=("similarity", summary_score),
        top_n_papers_aggregated=("similarity", "size"),
    )

    df_paper_grouped_by_professor["n_papers"] = professor_paper_counts

    df_paper_grouped_by_professor.sort_values(
        "similarity",
        ascending=False,
        inplace=True,
    )

    df_paper_grouped_by_professor_topn = df_paper_grouped_by_professor.head(
        top_n_pis
    ).copy()
    df_paper_grouped_by_professor_topn.reset_index(drop=False, inplace=True)
    df_paper_grouped_by_professor_topn = df_paper_grouped_by_professor_topn.loc[
        :, ["Professor", "n_papers", "top_n_papers_aggregated", "similarity"]
    ].copy()

    table2 = df_paper_grouped_by_professor_topn.copy()

    return table2, df_top_papers_by_professor, df_paper_year


def get_table3_top_professor_topn_papers(df_top_papers_by_professor, table2):
    """Table 3: top-N professors' top-N papers, merged with each professor's median score."""
    df_top_professor_topn_papers = df_top_papers_by_professor.loc[
        df_top_papers_by_professor.Professor.isin(table2.Professor)
    ].copy()

    df_top_professor_topn_papers = df_top_professor_topn_papers.merge(
        table2[["Professor", "similarity"]],
        on="Professor",
        how="left",
        suffixes=("", "_median"),
    )

    df_top_professor_topn_papers.sort_values(
        ["similarity_median", "similarity"],
        ascending=[False, False],
        inplace=True,
    )
    df_top_professor_topn_papers.reset_index(drop=True, inplace=True)
    df_top_professor_topn_papers = df_top_professor_topn_papers.loc[
        :,
        [
            "Year",
            "PMID",
            "Professor",
            "similarity_median",
            "similarity",
            "title",
            "journal",
        ],
    ]

    table3 = df_top_professor_topn_papers.copy()
    return table3


def make_fig6_boxplot(table3, table2, top_n_pis, keyword):
    """Figure 6: interactive Plotly boxplot of similarity per top professor."""
    top_professors = table2.Professor.values

    fig = px.box(
        table3,
        x="similarity",
        y="Professor",
        category_orders={"Professor": list(top_professors)},
        points="all",
        color="Professor",
        notched=False,
        hover_data=["Professor", "title", "Year"],
    )

    fig.update_traces(
        jitter=0.5,
        pointpos=0,
        marker=dict(size=5),
    )

    fig.update_layout(
        height=650,
        width=1100,
        showlegend=False,
        yaxis={
            "categoryorder": "array",
            "categoryarray": list(top_professors),
            "autorange": "reversed",
        },
        title=f"Top {top_n_pis} Professors by Similarity to: {keyword}",
        xaxis_title="Cosine Similarity",
        yaxis_title="Professor",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        title_font=dict(color="black"),
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

    return fig


def get_table4_full_list(df_paper_year, table2):
    """Table 4: full list of papers (since year_filter) for the top-N professors."""
    top_professors = table2.Professor.values

    df_paper_year_topn_professors = df_paper_year.loc[
        df_paper_year.Professor.isin(top_professors), :
    ].copy()
    df_paper_year_topn_professors = df_paper_year_topn_professors.merge(
        table2[["Professor", "similarity"]],
        on="Professor",
        how="left",
        suffixes=("", "_median"),
    )

    df_paper_year_topn_professors = df_paper_year_topn_professors.loc[
        :,
        [
            "Year",
            "PMID",
            "Professor",
            "similarity_median",
            "similarity",
            "title",
            "journal",
        ],
    ].copy()

    df_paper_year_topn_professors = df_paper_year_topn_professors.sort_values(
        ["similarity_median", "similarity"],
        ascending=[False, False],
        inplace=False,
    )
    df_paper_year_topn_professors.reset_index(drop=True, inplace=True)

    table4 = df_paper_year_topn_professors.copy()
    return table4


# ==============================================================
# Faculty coverage utility (ILS_Faculty_List.txt)
# ==============================================================
def get_faculty_coverage(df_paper, faculty_list):
    """
    For every professor in the ILS faculty roster, report whether they
    appear in the abstracts dataset and how many papers they have.
    """
    counts = df_paper["Professor"].value_counts()

    coverage = pd.DataFrame({"Professor": faculty_list})
    coverage["n_papers"] = coverage["Professor"].map(counts).fillna(0).astype(int)
    coverage["In_Dataset"] = coverage["n_papers"] > 0

    coverage.sort_values(
        ["In_Dataset", "n_papers"], ascending=[False, False], inplace=True
    )
    coverage.reset_index(drop=True, inplace=True)
    return coverage


def search_faculty(query, faculty_list, df_paper):
    """
    Case-insensitive substring search for a professor name, across both the
    ILS faculty roster and the professors actually present in the dataset.
    """
    columns = ["Professor", "n_papers", "In_ILS_List", "In_Dataset"]
    query_clean = (query or "").strip().lower()
    if not query_clean:
        return pd.DataFrame(columns=columns)

    counts = df_paper["Professor"].value_counts()
    all_names = sorted(set(faculty_list) | set(df_paper["Professor"].unique()))
    matches = [n for n in all_names if query_clean in n.lower()]

    result = pd.DataFrame({"Professor": matches})
    result["n_papers"] = result["Professor"].map(counts).fillna(0).astype(int)
    result["In_ILS_List"] = result["Professor"].isin(faculty_list)
    result["In_Dataset"] = result["n_papers"] > 0
    result = result[columns]
    return result
