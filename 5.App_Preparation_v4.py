"""
FindMyLab core analysis library (v4).

Presentation-layer additions on top of `5.App_Preparation_v3.py`:
- `PREPRINT_JOURNALS` / `filter_preprints()`: lets the "full paper list"
  table (Table 4) optionally exclude pre-print entries (medRxiv/ArXiv/bioRxiv).
- `make_fig6_boxplot()`: overrides v2's version to fix a color bug (see
  docstring below) that only manifests once Streamlit is imported.

This module does NOT modify `5.App_Preparation_v2.py` or
`5.App_Preparation_v3.py`. It loads v3 dynamically (same technique v3 uses
to load v2) and re-exports everything from it, then adds/overrides only the
functions listed above. All other table/figure-construction logic is reused
unchanged from v2/v3.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import plotly.express as px

# ------------------------------------------------------------
# Dynamically load v3 (unmodified) and re-export its public names
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
_V3_PATH = BASE_DIR / "5.App_Preparation_v3.py"
_V3_MODULE_NAME = "findmylab_core_v3_for_v4"

_spec = importlib.util.spec_from_file_location(_V3_MODULE_NAME, _V3_PATH)
v3 = importlib.util.module_from_spec(_spec)
sys.modules[_V3_MODULE_NAME] = v3
_spec.loader.exec_module(v3)


def _reexport(module: ModuleType, target: dict) -> None:
    for name in dir(module):
        if not name.startswith("_"):
            target[name] = getattr(module, name)


_reexport(v3, globals())


# ==============================================================
# Pre-print filtering (used by Table 4 / "full paper list")
# ==============================================================
PREPRINT_JOURNALS = [
    "medRxiv : the preprint server for health sciences",
    "ArXiv",
    "bioRxiv : the preprint server for biology",
]


def filter_preprints(df, include_preprints=True):
    """
    Optionally drop rows whose `journal` is one of the known pre-print
    servers (medRxiv, ArXiv, bioRxiv). No-op when `include_preprints=True`.
    """
    if include_preprints:
        return df
    return df.loc[~df["journal"].isin(PREPRINT_JOURNALS)].copy()


# ==============================================================
# Fig6 boxplot color-bug fix
# ==============================================================
# Plotly's own default qualitative color sequence, hardcoded here as a
# literal list. This must NOT be read from `px.colors.qualitative.Plotly`
# (or from the default template's `layout.colorway`) at runtime, because
# once `streamlit` has been imported, Streamlit monkey-patches that default
# sequence to a set of near-black placeholder colors (`#000001`, `#000002`,
# ...). Streamlit's own `st.plotly_chart()` front-end JS knows how to swap
# those placeholders for real theme colors when rendering inside the app,
# but a plain `fig.to_html()` export (used by the "Download plot (HTML)"
# button) has no such JS, so the placeholders would be baked into the
# downloaded file as-is, rendering as solid near-black/gray boxes instead
# of one distinct color per professor. Passing this explicit
# `color_discrete_sequence` bypasses Streamlit's patched default entirely.
_PLOTLY_DEFAULT_COLORS = [
    "#636efa",
    "#EF553B",
    "#00cc96",
    "#ab63fa",
    "#FFA15A",
    "#19d3f3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]


def make_fig6_boxplot(table3, table2, top_n_pis, keyword):
    """
    Figure 6: interactive Plotly boxplot of similarity per top professor.

    Same as v2's `make_fig6_boxplot`, except it passes an explicit
    `color_discrete_sequence` to `px.box()` (see `_PLOTLY_DEFAULT_COLORS`
    docstring above for why this is required once Streamlit is imported).
    """
    top_professors = table2.Professor.values

    fig = px.box(
        table3,
        x="similarity",
        y="Professor",
        category_orders={"Professor": list(top_professors)},
        points="all",
        color="Professor",
        color_discrete_sequence=_PLOTLY_DEFAULT_COLORS,
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
