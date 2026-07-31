"""Small shared UI helpers for the Streamlit pages (download buttons, etc.)."""

import io

import streamlit as st


def matplotlib_download_button(fig, label, file_name, key=None):
    """Render a download button that exports a matplotlib Figure as PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
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


def dataframe_download_button(df, label, file_name, key=None):
    """Render a download button that exports a DataFrame as CSV."""
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
        key=key,
    )
