"""FindMyLab v2 — Page 1: Basic Stats."""

import streamlit as st

from core_loader import core
from ui_helpers import (
    matplotlib_download_button,
    plotly_download_button,
    dataframe_download_button,
)

st.set_page_config(page_title="FindMyLab — Basic Stats", page_icon="📊", layout="wide")

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
# Figure 1 — Number of papers per professor (static)
# ------------------------------------------------------------
st.subheader("Number of papers per professor")

fig1, counts = core.make_fig1_bar_papers_per_professor(df_paper)
st.pyplot(fig1)
matplotlib_download_button(
    fig1,
    "Download plot (PNG)",
    "Fig1_Number_of_Papers_per_Professor.png",
    key="dl_fig1",
)

st.divider()

# ------------------------------------------------------------
# Figure 2 — Number of papers per professor by year (interactive heatmap)
# ------------------------------------------------------------
st.subheader("Number of papers per professor, by year")
st.caption("Hover over a cell to see the exact paper count.")

df_plot_heatmap = core.get_paper_year_professor_heatmap_data(df_paper)
fig2 = core.make_fig2_heatmap_interactive(df_plot_heatmap)
st.plotly_chart(fig2, use_container_width=True, theme=None)
plotly_download_button(
    fig2,
    "Download plot (HTML)",
    "Fig2_Number_of_Papers_per_Professor_by_Year.html",
    key="dl_fig2",
)

st.divider()

# ------------------------------------------------------------
# Figure 3 — Publication trend for selected professors (static, interactive selection)
# ------------------------------------------------------------
st.subheader("Publication trend for selected professors")

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
    fig3, df_plot_line = core.make_fig3_publication_trend(df_paper, target_professors)
    st.pyplot(fig3)
    matplotlib_download_button(
        fig3,
        "Download plot (PNG)",
        "Fig3_Publication_Trend.png",
        key="dl_fig3",
    )
    with st.expander("Show underlying data"):
        st.dataframe(df_plot_line, use_container_width=True, hide_index=True)
        dataframe_download_button(
            df_plot_line, "Download data (CSV)", "Fig3_data.csv", key="dl_fig3_data"
        )
