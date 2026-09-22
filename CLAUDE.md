# CLAUDE.md

## Project
Standalone Streamlit dashboard. Not part of the Scratch research repo.

## Layout
- `app/Overview.py` — entry point (home page)
- `app/pages/` — one file per page; Streamlit auto-discovers them
- `app/data/` — data loaders; all file paths come from `config.py` or env vars
- `tests/` — pytest

## Run
    streamlit run app/Overview.py

## Rules
- No hardcoded file paths; use `config.py` / env vars
- Type hints on all function signatures
- Cache data loads with `@st.cache_data`
- Keep pages thin: loaders in `app/data/`, plotting helpers in `app/plots.py`
