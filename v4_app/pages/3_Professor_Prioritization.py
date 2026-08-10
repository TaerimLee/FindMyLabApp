"""FindMyLab v4 — Page 3: Professor Prioritization."""

import streamlit as st

from core_loader_v4 import core
from ui_helpers_v4 import (
    disable_clear_cache_shortcut,
    dataframe_download_button,
    plotly_download_button,
    with_row_number,
)

st.set_page_config(
    page_title="FindMyLab — Professor Prioritization", page_icon="🎯", layout="wide"
)

disable_clear_cache_shortcut()

st.title("🎯 Professor Prioritization")
st.write(
    "Enter research keywords or a short research-interest paragraph. "
    "FindMyLab will rank professors and papers based on semantic similarity "
    "to their publications."
)


@st.cache_data
def _load_data():
    df_paper = core.load_paper_data()
    embeddings = core.load_embeddings()
    return df_paper, embeddings


@st.cache_resource
def _load_model():
    return core.load_sentence_model()


df_paper, embeddings = _load_data()

with st.sidebar:
    st.header("Settings")

    keyword = st.text_area(
        "Research interest: keywords or a short paragraph (<300 words)",
        value=core.DEFAULT_RESEARCH_INTEREST,
        height=180,
    )

    year_filter = st.slider(
        "Use papers from year ≥",
        min_value=int(df_paper["Year"].min()),
        max_value=int(df_paper["Year"].max()),
        value=core.DEFAULT_YEAR_FILTER,
    )
    top_n_papers = st.slider(
        "Top N papers (overall ranking, regardless of professor)",
        5,
        100,
        core.DEFAULT_TOPN_PAPERS,
    )
    top_n_pis = st.slider("Top N professors", 3, 50, core.DEFAULT_TOPN_PIS)
    min_papers = st.number_input(
        "Minimum papers per professor (since year filter)",
        min_value=1,
        max_value=100,
        value=core.DEFAULT_MIN_PAPERS,
    )
    top_n_aggregate = st.number_input(
        "Top N papers aggregated per professor",
        min_value=1,
        max_value=50,
        value=core.DEFAULT_TOPN_PAPERS_AGGREGATE,
    )

    include_preprints = st.checkbox(
        "Include pre-prints",
        value=True,
        help=(
            "Uncheck to exclude papers published in medRxiv, ArXiv, or "
            "bioRxiv (pre-print servers) from this entire analysis "
            "(all tables and figures below)."
        ),
        key="include_preprints",
    )

    run_button = st.button("Run analysis", type="primary")

if not keyword.strip():
    st.info("Enter a research interest in the sidebar, then click **Run analysis**.")
    st.stop()

if run_button or "pp_results" in st.session_state:
    if run_button:
        with st.spinner(
            "Embedding your research interest and computing similarities..."
        ):
            model = _load_model()
            # Compute similarity on the full (unfiltered) dataset so that the
            # `embeddings` array (aligned with df_paper's row order) stays in
            # sync. The pre-print filter is applied afterwards, so toggling
            # it doesn't require recomputing embeddings/similarity.
            df_similarity_full = core.compute_similarity(
                df_paper, embeddings, keyword, model
            )

        st.session_state["pp_results"] = dict(
            keyword=keyword,
            df_similarity_full=df_similarity_full,
            year_filter=year_filter,
            top_n_papers=top_n_papers,
            top_n_pis=top_n_pis,
            min_papers=min_papers,
            top_n_aggregate=top_n_aggregate,
        )

    results = st.session_state["pp_results"]
    df_similarity_full = results["df_similarity_full"]
    keyword_used = results["keyword"]
    year_filter_used = results["year_filter"]
    top_n_papers_used = results["top_n_papers"]
    top_n_pis_used = results["top_n_pis"]
    min_papers_used = results["min_papers"]
    top_n_aggregate_used = results["top_n_aggregate"]

    df_similarity = core.filter_preprints(df_similarity_full, include_preprints)

    table1 = core.get_table1_top_papers(df_similarity, year_filter_used, top_n_papers_used)
    table2, df_top_papers_by_professor, df_paper_year = core.get_table2_grouped_by_professor(
        df_similarity,
        year_filter=year_filter_used,
        top_n_pis=top_n_pis_used,
        min_papers=min_papers_used,
        top_n_aggregate=top_n_aggregate_used,
    )

    if table2.empty:
        st.warning(
            "No professors matched the current filters. "
            "Try lowering the minimum paper count or changing the year filter."
        )
    else:
        table3 = core.get_table3_top_professor_topn_papers(
            df_top_papers_by_professor, table2
        )
        table4 = core.get_table4_full_list(df_paper_year, table2)

        st.subheader("Top matching papers overall")
        st.dataframe(with_row_number(table1), use_container_width=True)
        dataframe_download_button(
            table1, "Download (CSV)", "Table1_top_papers.csv", key="dl_t1"
        )

        st.divider()

        # ------------------------------------------------------------
        # Table 2 (30%) + Figure 6 (70%) side by side
        # ------------------------------------------------------------
        col_table2, col_fig6 = st.columns([3, 7])

        with col_table2:
            st.subheader("Top professors")
            table2_display = table2.rename(
                columns={
                    "n_papers": f"Total papers published (>{year_filter_used})",
                    "top_n_papers_aggregated": "TopN papers aggregated",
                    "similarity": "Aggregated similarity score",
                }
            )
            st.dataframe(with_row_number(table2_display), use_container_width=True)
            dataframe_download_button(
                table2_display,
                "Download (CSV)",
                "Table2_top_professors.csv",
                key="dl_t2",
            )

        with col_fig6:
            st.subheader("Similarity distribution per top professor")
            fig6 = core.make_fig6_boxplot(table3, table2, top_n_pis_used, keyword_used)
            fig6.update_yaxes(automargin=True)
            fig6.update_layout(margin=dict(l=220))
            st.plotly_chart(fig6, use_container_width=True, theme=None)
            plotly_download_button(
                fig6, "Download plot (HTML)", "Fig6_boxplot.html", key="dl_fig6"
            )

        st.divider()

        st.subheader(f"Top professors' top papers (>{year_filter_used})")
        st.dataframe(with_row_number(table3), use_container_width=True)
        dataframe_download_button(
            table3, "Download (CSV)", "Table3_top_professor_top_papers.csv", key="dl_t3"
        )

        st.divider()

        st.subheader(f"Full paper list for top professors (>{year_filter_used})")
        st.dataframe(with_row_number(table4), use_container_width=True, height=500)
        dataframe_download_button(
            table4, "Download (CSV)", "Table4_full_list.csv", key="dl_t4"
        )
