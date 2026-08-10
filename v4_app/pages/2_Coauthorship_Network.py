"""FindMyLab v4 — Page 2: Coauthorship Network."""

import streamlit as st

from core_loader_v4 import core
from ui_helpers_v4 import (
    dataframe_download_button,
    disable_clear_cache_shortcut,
    matplotlib_download_button,
    with_row_number,
)

st.set_page_config(
    page_title="FindMyLab — Coauthorship Network", page_icon="🕸️", layout="wide"
)

disable_clear_cache_shortcut()

st.title("🕸️ Coauthorship Network")
st.write(
    "Explore which professors have co-authored papers together, based on shared PMIDs."
)


@st.cache_data
def _load_data():
    return core.load_paper_data()


df_paper = _load_data()

all_professors = sorted(df_paper["Professor"].unique().tolist())

default_index = (
    all_professors.index(core.DEFAULT_NETWORK_PROFESSOR)
    if core.DEFAULT_NETWORK_PROFESSOR in all_professors
    else 0
)

# ------------------------------------------------------------
# Controls for the single-professor focus, placed above the plots so that
# Figure 4, Figure 5, and the co-authors table all start at the same height.
# ------------------------------------------------------------
ctrl_col1, ctrl_col2, _ = st.columns([2, 1, 5])
with ctrl_col1:
    target_professor = st.selectbox(
        "Focus on a single professor",
        options=all_professors,
        index=default_index,
    )
with ctrl_col2:
    background = st.checkbox("Background nodes", value=False)

col_left, col_mid, col_right = st.columns([5, 3, 2])

# ------------------------------------------------------------
# Figure 4 — Overall map (static, high-resolution)
# ------------------------------------------------------------
with col_left:
    st.subheader("Overall co-authorship map (all PIs)")

    fig4, G, prof_paper_counts, edge_weights = core.make_coauthorship_figure_hd(
        df_paper, target_pi="All", background=True, fig_size=(15, 9)
    )
    st.pyplot(fig4, use_container_width=True)
    matplotlib_download_button(
        fig4,
        "Download plot (PNG)",
        "Fig4_Co_authorship_Network_Overall_Map.png",
        key="dl_fig4",
    )

# ------------------------------------------------------------
# Figure 5 — Single PI focus (static, high-resolution, interactive selection)
# ------------------------------------------------------------
with col_mid:
    st.subheader(f"Focus on {target_professor}")

    if target_professor not in G:
        st.warning(
            f"{target_professor} has no recorded co-authorship links in this dataset."
        )
    else:
        fig5, _, _, _ = core.make_coauthorship_figure_hd(
            df_paper, target_pi=target_professor, background=background, fig_size=(6, 6)
        )
        st.pyplot(fig5, use_container_width=True)
        matplotlib_download_button(
            fig5,
            "Download plot (PNG)",
            f"Fig5_Co_authorship_Network_{target_professor.replace(' ', '_')}.png",
            key="dl_fig5",
        )

# ------------------------------------------------------------
# Professor list for the selected sub-network (co-authors of target_professor)
# ------------------------------------------------------------
with col_right:
    st.subheader("Co-authors in this network")

    if target_professor in G:
        neighbor_rows = [
            {
                "Professor": neighbor,
                "Shared papers": G[target_professor][neighbor]["weight"],
                "Total papers": prof_paper_counts.get(neighbor, 0),
            }
            for neighbor in G.neighbors(target_professor)
        ]
        neighbors_df = (
            core.pd.DataFrame(neighbor_rows)
            .sort_values(["Shared papers", "Total papers"], ascending=[False, False])
            .reset_index(drop=True)
        )
        st.caption(f"{len(neighbors_df)} co-author(s) of **{target_professor}**")
        st.dataframe(with_row_number(neighbors_df), use_container_width=True)
        dataframe_download_button(
            neighbors_df,
            "Download (CSV)",
            f"Coauthors_{target_professor.replace(' ', '_')}.csv",
            key="dl_neighbors",
        )
    else:
        st.info("No co-authors to list for this professor.")
