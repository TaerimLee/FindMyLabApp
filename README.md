# FindMyLab

A tool to help ILS (Interdisciplinary Life Sciences, UT Austin) students explore
faculty publications: activity over time, coauthorship networks, and semantic
similarity to a student's own research interests.

Built by the [BioML Society](https://www.biomlsociety.org/).

## Running locally

```bash
pip install -r requirements.txt
streamlit run v3_app/Overview.py
```

## Project layout

- `5.App_Preparation_v2.py` — core analysis library (data loading, tables, figures).
- `5.App_Preparation_v3.py` — presentation-layer additions on top of v2 (interactive
  Plotly figures, hi-res exports); dynamically loads v2 and re-exports it.
- `v3_app/` — the Streamlit multipage app (entry point: `v3_app/Overview.py`).
- `Abstracts/`, `Embeddings/` — dataset and precomputed sentence embeddings.
- `ILS_Faculty_List.txt` — ILS faculty roster.
- `tmp/` — earlier (v1/v2) app versions and the v1↔v2 validation script, kept for
  reference; not used by the running app.

