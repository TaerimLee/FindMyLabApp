"""FindMyLab v2 — Page 2: Coauthorship Network."""

import streamlit as st

from core_loader import core
from ui_helpers import matplotlib_download_button

st.set_page_config(
    page_title="FindMyLab — Coauthorship Network", page_icon="🕸️", layout="wide"
)

st.title("🕸️ Coauthorship Network")
st.write(
    "Explore which professors have co-authored papers together, based on shared PMIDs."
)


@st.cache_data
def _load_data():
    return core.load_paper_data()


df_paper = _load_data()

# ------------------------------------------------------------
# Figure 4 — Overall map (static)
# ------------------------------------------------------------
st.subheader("Overall co-authorship map (all PIs)")

fig4, G, prof_paper_counts, edge_weights = core.make_coauthorship_figure(
    df_paper, target_pi="All", background=True, fig_size=(20, 13)
)
st.pyplot(fig4)
matplotlib_download_button(
    fig4,
    "Download plot (PNG)",
    "Fig4_Co_authorship_Network_Overall_Map.png",
    key="dl_fig4",
)

st.divider()

# ------------------------------------------------------------
# Figure 5 — Single PI focus (static, interactive selection)
# ------------------------------------------------------------
st.subheader("Focus on a single professor")

all_professors = sorted(df_paper["Professor"].unique().tolist())
default_index = (
    all_professors.index(core.DEFAULT_NETWORK_PROFESSOR)
    if core.DEFAULT_NETWORK_PROFESSOR in all_professors
    else 0
)

col1, col2 = st.columns([3, 1])
with col1:
    target_professor = st.selectbox(
        "Select a professor",
        options=all_professors,
        index=default_index,
    )
with col2:
    background = st.checkbox("Show background (unconnected) nodes", value=False)

if target_professor not in G:
    st.warning(
        f"{target_professor} has no recorded co-authorship links in this dataset."
    )
else:
    fig5, _, _, _ = core.make_coauthorship_figure(
        df_paper, target_pi=target_professor, background=background, fig_size=(8, 8)
    )
    st.pyplot(fig5)
    matplotlib_download_button(
        fig5,
        "Download plot (PNG)",
        f"Fig5_Co_authorship_Network_{target_professor.replace(' ', '_')}.png",
        key="dl_fig5",
    )
