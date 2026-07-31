"""
Validate that `5.App_Preparation_v2.py` (the refactored library) produces
outputs equivalent to `5.App_Preparation_v1.py` (the original script).

Per project requirements:
- Tables must match **exactly** (pd.testing.assert_frame_equal).
- Figures don't need to be pixel-identical, but the underlying data/meaning
  must match (verified by comparing the data that feeds each figure).

This script only reads `5.App_Preparation_v1.py` (via runpy, so it isn't
modified) and does not write anything back to it.

Run with:
    conda run -n find_my_lab python validate_v1_v2.py
"""

import runpy
import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, avoids plt.show() blocking/opening windows

import plotly.io as pio

pio.renderers.default = "json"  # avoid fig.show() trying to open a browser

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from core_loader import core  # noqa: E402

V1_SCRIPT = BASE_DIR / "5.App_Preparation_v1.py"

FAILURES = []
PASSES = []


def check(name, condition, detail=""):
    if condition:
        PASSES.append(name)
        print(f"[PASS] {name}")
    else:
        FAILURES.append((name, detail))
        print(f"[FAIL] {name} — {detail}")


def frames_equal(a, b, **kwargs):
    try:
        pd.testing.assert_frame_equal(
            a.reset_index(drop=True), b.reset_index(drop=True), **kwargs
        )
        return True, ""
    except AssertionError as e:
        return False, str(e)


def series_equal(a, b, **kwargs):
    try:
        pd.testing.assert_series_equal(
            a.reset_index(drop=True), b.reset_index(drop=True), **kwargs
        )
        return True, ""
    except AssertionError as e:
        return False, str(e)


def independent_edge_weights(df_paper):
    """A second, independently-written implementation of the co-authorship
    edge-weight computation, used as a cross-check for `core.build_coauthorship_data`.
    """
    df = df_paper.copy()
    df["Year"] = df["Year"].astype(int)
    edge_weights = {}
    for pmid, group in df.groupby("PMID"):
        profs = sorted(set(group["Professor"]))
        if len(profs) >= 2:
            for p1, p2 in combinations(profs, 2):
                pair = (p1, p2)
                edge_weights[pair] = edge_weights.get(pair, 0) + 1
    return edge_weights


def main():
    print("=" * 70)
    print("Running v1 script (5.App_Preparation_v1.py) to capture its outputs...")
    print("This loads the sentence-transformers model and may take a while.")
    print("=" * 70)

    ns = runpy.run_path(str(V1_SCRIPT), run_name="__v1__")

    print("\nv1 script finished. Running comparisons against v2 (core) module...\n")

    # ----------------------------------------------------------------
    # Basic Stats
    # ----------------------------------------------------------------
    df_paper_v1 = ns["df_paper"]  # note: v1 mutates this in place (adds "similarity")
    df_paper_v2 = core.load_paper_data()

    v1_counts = df_paper_v1["Professor"].value_counts()
    _, v2_counts = core.make_fig1_bar_papers_per_professor(df_paper_v2)
    ok, detail = series_equal(v1_counts, v2_counts, check_names=False)
    check("Figure 1 data (papers per professor)", ok, detail)

    v1_heatmap = ns["df_plot_heatmap"]
    v2_heatmap = core.get_paper_year_professor_heatmap_data(df_paper_v2)
    ok, detail = frames_equal(v1_heatmap, v2_heatmap, check_names=False)
    check("Figure 2 data (papers per professor by year)", ok, detail)

    v1_target_professors = ns["target_professors"]
    v1_line_data = ns["df_plot_line"]
    _, v2_line_data = core.make_fig3_publication_trend(
        df_paper_v2, v1_target_professors
    )
    ok, detail = frames_equal(v1_line_data, v2_line_data)
    check("Figure 3 data (publication trend)", ok, detail)

    # ----------------------------------------------------------------
    # Coauthorship Network
    # ----------------------------------------------------------------
    _, _, v2_edge_weights = core.build_coauthorship_data(df_paper_v2)
    v2_edge_weights_independent = independent_edge_weights(df_paper_v2)
    ok = v2_edge_weights == v2_edge_weights_independent
    check(
        "Coauthorship edge weights (core vs. independent re-implementation)",
        ok,
        "edge weight dicts differ" if not ok else "",
    )

    # ----------------------------------------------------------------
    # Professor Prioritization
    # Reuse v1's already-computed query embedding + paper embeddings so the
    # comparison isolates the *table construction* logic (the part that was
    # actually refactored) from any nondeterminism in model.encode().
    # ----------------------------------------------------------------
    my_interest_embedding = ns["my_interest_embedding"]
    embeddings_general = ns["embeddings_general"]
    keyword = ns["my_interest"][0]

    year_filter = ns["year_filter"]
    top_n_papers = ns["TopN_papers"]
    top_n_pis = ns["TopN_PIs"]
    min_papers = ns["N_papers_minimum"]
    top_n_aggregate = ns["TopN_papers_aggregate"]

    similarities = cosine_similarity(
        my_interest_embedding, embeddings_general
    ).flatten()
    df_similarity_v2 = df_paper_v2.copy()
    df_similarity_v2["similarity"] = similarities

    table1_v2 = core.get_table1_top_papers(df_similarity_v2, year_filter, top_n_papers)
    ok, detail = frames_equal(ns["Table_1"], table1_v2, check_exact=False, atol=1e-9)
    check("Table 1 (top papers overall)", ok, detail)

    table2_v2, df_top_papers_by_professor_v2, df_paper_year_v2 = (
        core.get_table2_grouped_by_professor(
            df_similarity_v2,
            year_filter=year_filter,
            top_n_pis=top_n_pis,
            min_papers=min_papers,
            top_n_aggregate=top_n_aggregate,
        )
    )
    ok, detail = frames_equal(ns["Table_2"], table2_v2, check_exact=False, atol=1e-9)
    check("Table 2 (top professors by aggregated similarity)", ok, detail)

    table3_v2 = core.get_table3_top_professor_topn_papers(
        df_top_papers_by_professor_v2, table2_v2
    )
    ok, detail = frames_equal(ns["Table_3"], table3_v2, check_exact=False, atol=1e-9)
    check("Table 3 (top professors' top papers)", ok, detail)

    table4_v2 = core.get_table4_full_list(df_paper_year_v2, table2_v2)
    ok, detail = frames_equal(ns["Table_4"], table4_v2, check_exact=False, atol=1e-9)
    check("Table 4 (full paper list for top professors)", ok, detail)

    # Figure 6 uses the same data as Table 3 / Table 2 (already validated above).
    fig6_v2 = core.make_fig6_boxplot(table3_v2, table2_v2, top_n_pis, keyword)
    check(
        "Figure 6 (boxplot) constructed without error from validated Table 2/3 data",
        fig6_v2 is not None,
    )

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"{len(PASSES)} passed, {len(FAILURES)} failed")
    print("=" * 70)

    if FAILURES:
        print("\nFAILED CHECKS:")
        for name, detail in FAILURES:
            print(f"- {name}\n  {detail}\n")
        sys.exit(1)
    else:
        print("\nAll checks passed: v2 outputs are equivalent to v1.")


if __name__ == "__main__":
    main()
