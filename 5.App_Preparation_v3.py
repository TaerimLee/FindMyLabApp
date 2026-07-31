"""
FindMyLab core analysis library (v3).

Presentation-layer enhancements on top of `5.App_Preparation_v2.py`:
- Higher-resolution static figures (Figure 1, Figure 4/5 network plots).
- An interactive (Plotly) version of the publication-trend plot (Figure 3),
  with hover tooltips showing exact paper counts.

This module does NOT modify `5.App_Preparation_v2.py` or
`5.App_Preparation_v1.py`. It loads v2 dynamically (same technique as
`core_loader.py`) and re-exports everything from it, then adds/overrides
only the functions listed above. All table-construction logic (Tables 1-4)
and data loading are reused unchanged from v2.
"""

import importlib.util
import sys
from math import ceil
from pathlib import Path
from types import ModuleType

import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------
# Dynamically load v2 (unmodified) and re-export its public names
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
_V2_PATH = BASE_DIR / "5.App_Preparation_v2.py"
_V2_MODULE_NAME = "findmylab_core_v2_for_v3"

_spec = importlib.util.spec_from_file_location(_V2_MODULE_NAME, _V2_PATH)
v2 = importlib.util.module_from_spec(_spec)
sys.modules[_V2_MODULE_NAME] = v2
_spec.loader.exec_module(v2)


def _reexport(module: ModuleType, target: dict) -> None:
    for name in dir(module):
        if not name.startswith("_"):
            target[name] = getattr(module, name)


_reexport(v2, globals())

# High-resolution DPI used for the static matplotlib figures in v3.
HD_DPI = 300


# ==============================================================
# Figure 1 (hi-res) — Number of papers per professor
# ==============================================================
def make_fig1_bar_papers_per_professor_hd(df_paper, dpi=450):
    """
    High-resolution version of v2's Figure 1 (papers per professor).
    Figure width scales with the number of professors so labels stay
    legible even at high DPI.
    """
    counts = v2.get_professor_paper_counts(df_paper)

    width = max(20, len(counts) * 0.22)
    fig, ax = plt.subplots(figsize=(width, 6), dpi=dpi)
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    ax.set_xticks(range(len(counts.index)))
    ax.set_xticklabels(counts.index, rotation=90, size=9, ha="center")
    ax.set_title("Number of Papers per Professor", fontsize=14)
    fig.tight_layout()

    return fig, counts


def make_fig1_bar_papers_per_professor_interactive(df_paper):
    """
    Interactive Plotly version of Figure 1 (papers per professor). Browsers
    render matplotlib PNGs at a fixed pixel size regardless of DPI, so on
    screen they can still look blurry when stretched; Plotly renders as
    vector/SVG in the browser so it stays crisp at any size, and also
    exposes a built-in camera icon to export a high-resolution PNG
    (no server-side kaleido/Chrome dependency needed).
    """
    counts = v2.get_professor_paper_counts(df_paper)

    fig = px.bar(
        x=counts.index,
        y=counts.values,
        labels={"x": "Professor", "y": "Number of papers"},
        template="plotly_white",
    )
    fig.update_traces(
        hovertemplate="Professor: %{x}<br>Papers: %{y}<extra></extra>",
        marker_color="steelblue",
    )
    fig.update_layout(
        title="Number of Papers per Professor",
        height=550,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        margin=dict(b=160),
    )
    fig.update_xaxes(tickangle=90, tickfont=dict(size=11, color="black"))
    fig.update_yaxes(tickfont=dict(color="black"))

    return fig, counts


def make_fig2_heatmap_interactive_hd(df_plot_heatmap):
    """
    Same data/figure as v2's `make_fig2_heatmap_interactive`, with a larger
    x-axis (professor name) tick font and enough bottom margin so the
    (rotated) professor names are not cut off.
    """
    fig = v2.make_fig2_heatmap_interactive(df_plot_heatmap)
    fig.update_xaxes(tickangle=90, tickfont=dict(size=11, color="black"))
    fig.update_layout(height=700, margin=dict(b=160))
    return fig


# ==============================================================
# Figure 3 (interactive) — Publication trend for selected professors
# ==============================================================
def make_fig3_publication_trend_interactive(df_paper, target_professors):
    """
    Interactive Plotly version of v2's Figure 3 (publication trend).
    Same underlying data as v2's `make_fig3_publication_trend`, but with
    hover tooltips showing the exact paper count per year.
    """
    if len(target_professors) == 0:
        raise ValueError("Select at least one professor for the trend plot.")
    if len(target_professors) > v2.MAX_TREND_PROFESSORS:
        raise ValueError(
            f"Select at most {v2.MAX_TREND_PROFESSORS} professors for the trend plot."
        )

    df_plot_line = v2.get_publication_trend_data(df_paper, target_professors)
    historical = df_plot_line.loc[df_plot_line["Year"] < 2026].copy()

    facet_col_wrap = min(4, len(target_professors))

    fig = px.line(
        historical,
        x="Year",
        y="count",
        facet_col="Professor",
        facet_col_wrap=facet_col_wrap,
        category_orders={"Professor": target_professors},
        markers=True,
        template="plotly_white",
    )

    fig.update_traces(
        hovertemplate="Year: %{x}<br>Papers: %{y}<extra></extra>",
        line=dict(width=1.5),
        marker=dict(size=6),
    )

    # Plotly facet titles default to "Professor=Name"; strip the prefix.
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=", 1)[-1]))

    for y in (10, 20, 30):
        fig.add_hline(
            y=y, line_dash="dash", line_color="gray", opacity=0.3, row="all", col="all"
        )
    for x in (2000, 2010, 2020):
        fig.add_vline(
            x=x, line_dash="dash", line_color="gray", opacity=0.3, row="all", col="all"
        )

    n_rows = ceil(len(target_professors) / facet_col_wrap)
    fig.update_layout(
        height=280 * n_rows + 60,
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        margin=dict(t=60),
    )
    fig.update_xaxes(showticklabels=True, tickfont=dict(color="black"), matches=None)
    fig.update_yaxes(tickfont=dict(color="black"))

    return fig, df_plot_line


# ==============================================================
# Figure 4 / 5 (hi-res) — Coauthorship network
# ==============================================================
def make_coauthorship_figure_hd(
    df_paper, target_pi="All", background=True, fig_size=(13, 13), dpi=HD_DPI
):
    """
    High-resolution version of v2's `make_coauthorship_figure`. Identical
    drawing logic; only the figure DPI is configurable (v2 hardcodes 100).
    """
    G, prof_paper_counts, edge_weights = v2.build_coauthorship_data(df_paper)

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

    fig = plt.figure(figsize=fig_size, dpi=dpi)

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

    for u, v_ in G.edges:
        w = G[u][v_]["weight"]
        is_highlighted_edge = target_pi == "All" or u == target_pi or v_ == target_pi

        if is_highlighted_edge:
            edges_to_draw.append((u, v_))
            edge_widths.append(1.0 + (w * 1.2))
            edge_colors.append("gray")
            edge_alphas.append(0.6 if target_pi == "All" else 0.8)
        elif background:
            edges_to_draw.append((u, v_))
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
# Simple "is this professor in the dataset?" search (no ILS roster needed)
# ==============================================================
def search_professor_in_dataset(query, df_paper):
    """
    Case-insensitive substring search for a professor name among professors
    that actually have papers in the dataset. Returns a DataFrame with
    columns ["Professor", "n_papers"], or an empty DataFrame if no match.
    """
    columns = ["Professor", "n_papers"]
    query_clean = (query or "").strip().lower()
    if not query_clean:
        return v2.pd.DataFrame(columns=columns)

    counts = df_paper["Professor"].value_counts()
    matches = [p for p in counts.index if query_clean in p.lower()]

    result = v2.pd.DataFrame({"Professor": matches})
    result["n_papers"] = result["Professor"].map(counts).astype(int)
    result.sort_values("n_papers", ascending=False, inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result
