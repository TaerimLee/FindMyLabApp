"""FindMyLab v3 — Page 1: Basic Stats."""

import streamlit as st

from core_loader_v3 import core
from ui_helpers_v3 import (
    disable_clear_cache_shortcut,
    plotly_download_button,
    render_plotly_with_png_export,
)

st.set_page_config(page_title="FindMyLab — Basic Stats", page_icon="📊", layout="wide")

disable_clear_cache_shortcut()

st.title("📊 Basic Stats")
st.write(
    "Overview of how many papers each professor has published in the dataset, "
    "and how that activity has evolved over time."
)


@st.cache_data
def _load_data():
    return core.load_paper_data()


df_paper = _load_data()

# ------------------------------------------------------------
# Figure 1 — Number of papers per professor (interactive, crisp at any zoom)
# ------------------------------------------------------------
st.subheader("Number of papers per professor")
st.caption(
    "Use the camera icon in the top-right toolbar to download a high-resolution PNG."
)

fig1, counts = core.make_fig1_bar_papers_per_professor_interactive(df_paper)
render_plotly_with_png_export(
    fig1,
    "Fig1_Number_of_Papers_per_Professor.png",
    use_container_width=True,
    theme=None,
)
plotly_download_button(
    fig1,
    "Download plot (HTML)",
    "Fig1_Number_of_Papers_per_Professor.html",
    key="dl_fig1",
)

st.divider()

# ------------------------------------------------------------
# Figure 2 — Number of papers per professor by year (interactive heatmap)
# ------------------------------------------------------------
st.subheader("Number of papers per professor, by year")
st.caption(
    "Hover over a cell to see the exact paper count. "
    "Use the camera icon in the top-right toolbar to download a high-resolution PNG."
)

df_plot_heatmap = core.get_paper_year_professor_heatmap_data(df_paper)
fig2 = core.make_fig2_heatmap_interactive_hd(df_plot_heatmap)
render_plotly_with_png_export(
    fig2,
    "Fig2_Number_of_Papers_per_Professor_by_Year.png",
    use_container_width=True,
    theme=None,
)
plotly_download_button(
    fig2,
    "Download plot (HTML)",
    "Fig2_Number_of_Papers_per_Professor_by_Year.html",
    key="dl_fig2",
)

st.divider()

# ------------------------------------------------------------
# Figure 3 — Publication trend for selected professors (interactive)
# ------------------------------------------------------------
st.subheader("Publication trend for selected professors")
st.caption(
    "Hover over a point to see the exact paper count for that year. "
    "Use the camera icon in the top-right toolbar to download a high-resolution PNG."
)

all_professors = sorted(df_paper["Professor"].unique().tolist())
default_professors = [p for p in core.DEFAULT_TREND_PROFESSORS if p in all_professors]

target_professors = st.multiselect(
    f"Select up to {core.MAX_TREND_PROFESSORS} professors",
    options=all_professors,
    default=default_professors,
    max_selections=core.MAX_TREND_PROFESSORS,
)

if not target_professors:
    st.info("Select at least one professor to see their publication trend.")
else:
    fig3, df_plot_line = core.make_fig3_publication_trend_interactive(
        df_paper, target_professors
    )
    render_plotly_with_png_export(
        fig3,
        "Fig3_Publication_Trend.png",
        use_container_width=True,
        theme=None,
    )
    plotly_download_button(
        fig3,
        "Download plot (HTML)",
        "Fig3_Publication_Trend.html",
        key="dl_fig3",
    )
