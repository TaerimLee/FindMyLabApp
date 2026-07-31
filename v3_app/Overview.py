"""
FindMyLab — Overview page (v3 multipage app).

Run with:
    streamlit run v3_app/Overview.py

v3 changes vs. v2 (see conversation for full list):
- Sidebar label is "Overview" (this file's name), not "app v2".
- Faculty check simplified to: search a professor, see if they're in the
  dataset (no mention of the ILS_Faculty_List.txt roster).
- Disables the Ctrl+C -> "Clear caches" popup annoyance.
- Redesigned layout: header with BioML Society branding, dataset stats as
  metric cards, navigation cards for the three pages, and the professor
  search/download tool in its own section.
"""

import base64
from pathlib import Path

import streamlit as st

from core_loader_v3 import core
from ui_helpers_v3 import dataframe_download_button, disable_clear_cache_shortcut

FINDMYLAB_LOGO_PATH = Path(__file__).parent / "FindMyLab.png"

st.set_page_config(
    page_title="FindMyLab",
    page_icon=str(FINDMYLAB_LOGO_PATH) if FINDMYLAB_LOGO_PATH.exists() else "🔎",
    layout="wide",
)

disable_clear_cache_shortcut()

BIOML_LOGO_PATH = Path(__file__).parent / "BioML.png"
BIOML_URL = "https://www.biomlsociety.org/"
ILS_URL = "https://ils.utexas.edu/"


@st.cache_data
def _load_data():
    return core.load_paper_data()


df_paper = _load_data()

# ------------------------------------------------------------
# Header — text on the left, FindMyLab + BioML Society logos on the right
# (rendered at the same fixed height so they appear consistently sized)
# ------------------------------------------------------------
text_col, logos_col = st.columns([3, 2])
with text_col:
    st.title("🔎 FindMyLab")
    st.markdown(
        "##### Explore faculty publications: activity, collaborations, and fit to your research interests."
    )
with logos_col:
    logo_html_parts = []
    if FINDMYLAB_LOGO_PATH.exists():
        findmylab_logo_b64 = base64.b64encode(FINDMYLAB_LOGO_PATH.read_bytes()).decode()
        logo_html_parts.append(
            f'<img src="data:image/png;base64,{findmylab_logo_b64}" '
            'style="height:160px; width:auto; object-fit:contain;">'
        )
    if BIOML_LOGO_PATH.exists():
        logo_b64 = base64.b64encode(BIOML_LOGO_PATH.read_bytes()).decode()
        logo_html_parts.append(
            f'<a href="{BIOML_URL}" target="_blank">'
            f'<img src="data:image/png;base64,{logo_b64}" '
            'style="height:160px; width:auto; object-fit:contain;"></a>'
        )
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; justify-content:flex-end;
                    gap:24px; height:100%; margin-top:16px;">
            {''.join(logo_html_parts)}
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption(
    "Built for the "
    f"[Interdisciplinary Life Sciences (ILS) Graduate Programs]({ILS_URL}) at "
    "UT Austin, to help prospective and current students explore faculty research."
)
st.caption(
    "Built by the BioML Society, a community for those passionate about the "
    f"intersection of biology and machine learning. Visit [HERE]({BIOML_URL}) "
    "to learn more about us and our work."
)

st.divider()

# ------------------------------------------------------------
# Dataset snapshot
# ------------------------------------------------------------
st.subheader("📈 Dataset snapshot")

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Professors in dataset", df_paper["Professor"].nunique())
metric_col2.metric("Total papers in dataset", len(df_paper))
metric_col3.metric(
    "Years covered",
    f"{int(df_paper['Year'].min())}–{int(df_paper['Year'].max())}",
)

st.divider()

# ------------------------------------------------------------
# Navigation cards
# ------------------------------------------------------------
st.subheader("🧭 Explore the app")

nav_col1, nav_col2, nav_col3 = st.columns(3)
with nav_col1:
    with st.container(border=True):
        st.markdown("**📊 Basic Stats**")
        st.write("How many papers each professor has published, and when.")
        st.page_link("pages/1_Basic_Stats.py", label="Open Basic Stats", icon="📊")
with nav_col2:
    with st.container(border=True):
        st.markdown("**🕸️ Coauthorship Network**")
        st.write("Who collaborates with whom.")
        st.page_link(
            "pages/2_Coauthorship_Network.py",
            label="Open Coauthorship Network",
            icon="🕸️",
        )
with nav_col3:
    with st.container(border=True):
        st.markdown("**🎯 Professor Prioritization**")
        st.write("Rank professors by semantic similarity to your research interests.")
        st.page_link(
            "pages/3_Professor_Prioritization.py",
            label="Open Professor Prioritization",
            icon="🎯",
        )

st.divider()

# ------------------------------------------------------------
# Quick "is this professor in the dataset?" check
# ------------------------------------------------------------
st.subheader("🔍 Check if a professor is in the dataset")
st.write("Search for a professor by name to see if they have papers in the dataset.")

query = st.text_input(
    "Professor name",
    placeholder="e.g. Wilke, Cenik, Marcotte...",
)

if query.strip():
    matches = core.search_professor_in_dataset(query, df_paper)
    if matches.empty:
        st.warning(f"No professor matching '{query}' found in the dataset.")
    else:
        for _, row in matches.iterrows():
            st.success(
                f"✅ **{row['Professor']}** — found in dataset ({row['n_papers']} papers)"
            )

st.divider()

# ------------------------------------------------------------
# Full dataset download
# ------------------------------------------------------------
st.subheader("⬇️ Download the full dataset")

_columns_to_exclude = [
    "corresponding_authors",
    "corresponding_affiliations",
    "corresponding_countries",
]
df_paper_public = df_paper.drop(
    columns=[c for c in _columns_to_exclude if c in df_paper.columns]
)

dataframe_download_button(
    df_paper_public,
    "Download full paper table (CSV)",
    "FindMyLab_all_papers.csv",
    key="dl_full_paper_table",
)
