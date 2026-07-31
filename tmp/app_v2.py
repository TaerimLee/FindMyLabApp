"""
FindMyLab — Home / landing page (v2 multipage app).

Run with:
    streamlit run app_v2.py

This is a refactor/expansion of the pilot `app.py`, split into three
dedicated pages (see `pages/`), all backed by the shared analysis library
in `5.App_Preparation_v2.py` (loaded via `core_loader.py`).
"""

import streamlit as st

from core_loader import core

st.set_page_config(
    page_title="FindMyLab",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 FindMyLab")

st.markdown("""
FindMyLab helps you explore a dataset of faculty publications across three
lenses. Use the sidebar (or the links below) to navigate:

- **📊 Basic Stats** — how many papers each professor has published, and when.
- **🕸️ Coauthorship Network** — who collaborates with whom.
- **🎯 Professor Prioritization** — rank professors by semantic similarity to
  your own research interests.
""")

st.divider()

# ------------------------------------------------------------
# Faculty coverage lookup
# ------------------------------------------------------------
st.subheader("🔍 Faculty coverage check")
st.write(
    "Quickly check whether a professor is included in the underlying dataset "
    "(`ILS_Faculty_List.txt` roster vs. professors that actually have papers)."
)


@st.cache_data
def _load_faculty_and_papers():
    df_paper = core.load_paper_data()
    faculty_list = core.load_faculty_list()
    return df_paper, faculty_list


df_paper, faculty_list = _load_faculty_and_papers()
coverage = core.get_faculty_coverage(df_paper, faculty_list)

n_total = len(coverage)
n_found = int(coverage["In_Dataset"].sum())
n_missing = n_total - n_found

col1, col2, col3 = st.columns(3)
col1.metric("ILS faculty listed", n_total)
col2.metric("Found in dataset", n_found)
col3.metric("Missing from dataset", n_missing)

query = st.text_input(
    "Search for a professor by name (partial match is fine)",
    placeholder="e.g. Wilke, Cenik, Marcotte...",
)

if query.strip():
    matches = core.search_faculty(query, faculty_list, df_paper)
    if matches.empty:
        st.warning("No matching professor found.")
    else:
        st.dataframe(matches, use_container_width=True, hide_index=True)
else:
    with st.expander("Show full ILS faculty coverage table"):
        st.dataframe(coverage, use_container_width=True, hide_index=True)
        st.download_button(
            "Download coverage table (CSV)",
            data=coverage.to_csv(index=False).encode("utf-8"),
            file_name="ILS_faculty_coverage.csv",
            mime="text/csv",
        )
