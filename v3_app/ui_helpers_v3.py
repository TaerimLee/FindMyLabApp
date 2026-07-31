"""Shared UI helpers for the v3 Streamlit pages."""

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
    """Render a download button that exports a Plotly Figure as a self-contained HTML file."""
    html = fig.to_html(full_html=True, include_plotlyjs=True)
    st.download_button(
        label=label,
        data=html.encode("utf-8"),
        file_name=file_name,
        mime="text/html",
        key=key,
    )


def render_plotly_with_png_export(fig, file_name, scale=3, **kwargs):
    """
    Display a Plotly figure with its built-in camera/download icon (in the
    modebar) configured to export a high-resolution PNG client-side in the
    browser, with a sensible filename. This avoids needing a server-side
    kaleido/Chrome install for PNG export.
    """
    stem = file_name[:-4] if file_name.lower().endswith(".png") else file_name
    config = {
        "toImageButtonOptions": {
            "format": "png",
            "filename": stem,
            "scale": scale,
        },
        "displaylogo": False,
    }
    st.plotly_chart(fig, config=config, **kwargs)


def dataframe_download_button(df, label, file_name, key=None):
    """Render a download button that exports a DataFrame as CSV."""
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
        key=key,
    )
