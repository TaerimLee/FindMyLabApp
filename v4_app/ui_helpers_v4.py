"""Shared UI helpers for the v4 Streamlit pages.

Built on top of `v3_app/ui_helpers_v3.py`'s design, with one change for v4:
`plotly_download_button()` now forces a light `color-scheme` in the exported
HTML as defense-in-depth against browser/OS auto-dark-mode inverting a
standalone HTML file's colors (see its docstring for details). Plotly figures
only offer an HTML download (no PNG export button) in v4.
"""

import io

import streamlit as st
import streamlit.components.v1 as components


def disable_clear_cache_shortcut():
    """
    Prevent Streamlit's built-in 'c' (clear cache) keyboard shortcut from
    firing when the user presses Ctrl+C / Cmd+C to copy text.

    Streamlit's hotkey listener is bound on `document`. The DOM capturing
    phase always visits `window` before it visits `document`, regardless of
    when each listener was registered. So attaching our interceptor to
    `window` (capture phase) guarantees it runs before Streamlit's own
    `document`-level listener, and `stopImmediatePropagation` there prevents
    the event from ever reaching it.
    """
    components.html(
        """
        <script>
        (function() {
            const win = window.parent;
            if (win.__findmylab_copy_fix_installed) { return; }
            win.__findmylab_copy_fix_installed = true;
            win.addEventListener('keydown', function(e) {
                if ((e.ctrlKey || e.metaKey) && (e.key === 'c' || e.key === 'C')) {
                    e.stopImmediatePropagation();
                    e.stopPropagation();
                }
            }, true);
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def with_row_number(df):
    """Return a copy of `df` with a 1-based row-number index."""
    d = df.reset_index(drop=True).copy()
    d.index = d.index + 1
    return d


def matplotlib_download_button(fig, label, file_name, key=None):
    """Render a download button that exports a matplotlib Figure as PNG at its own DPI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=fig.dpi, bbox_inches="tight")
    st.download_button(
        label=label,
        data=buf.getvalue(),
        file_name=file_name,
        mime="image/png",
        key=key,
    )


def plotly_download_button(fig, label, file_name, key=None):
    """
    Render a download button that exports a Plotly Figure as a self-contained
    HTML file.

    Forces `color-scheme: light` (and an explicit white background) in the
    exported page as defense-in-depth against browsers/OSes with an auto
    "dark mode" for web content, which would otherwise invert a standalone
    HTML file's colors on open (the on-page chart is unaffected because it
    renders inside Streamlit's own light-themed app).

    Note: this is unrelated to the (separate, now-fixed) bug where Fig6's
    box colors were baked in as near-black placeholders -- that was caused
    by Streamlit monkey-patching Plotly Express's default qualitative color
    sequence once `streamlit` is imported, and is fixed at the source in
    `5.App_Preparation_v4.py`'s `make_fig6_boxplot()` by passing an explicit
    `color_discrete_sequence` instead of relying on Plotly's (patched)
    default.
    """
    html = fig.to_html(full_html=True, include_plotlyjs=True)
    html = html.replace(
        "<head>",
        '<head><meta name="color-scheme" content="light only">'
        "<style>html,body{background-color:#ffffff;}</style>",
        1,
    )
    st.download_button(
        label=label,
        data=html.encode("utf-8"),
        file_name=file_name,
        mime="text/html",
        key=key,
    )


def dataframe_download_button(df, label, file_name, key=None):
    """Render a download button that exports a DataFrame as CSV."""
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
        key=key,
    )
