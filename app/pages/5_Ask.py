import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.data import chat as C
from app.state import sidebar

st.set_page_config(page_title="Ask", layout="wide")
sidebar()
st.title("Ask the desk")

c1, c2 = st.columns([3, 2])
with c1:
    st.caption("Questions are answered by writing SQL against your warehouse. Every query is shown. "
               "The provenance badge under each answer is computed by the app from the queries that "
               "actually ran — the model cannot set it.")
with c2:
    data_only = st.toggle("Data-only mode", value=True,
                          help="On: the model may only state what a query returned, and must decline "
                               "anything the warehouse cannot answer. Off: general knowledge is allowed "
                               "but every such sentence is flagged.")
    if not C.has_api_key():
        st.warning("No `ANTHROPIC_API_KEY` found — running in demo mode with canned queries.", icon="🔑")

if "chat" not in st.session_state:
    st.session_state.chat = []


def render_queries(queries: list[dict]) -> None:
    for i, q in enumerate(queries, 1):
        ok = q.get("ok")
        head = f"{'✅' if ok else '❌'} Query {i} — {q.get('purpose') or ('failed' if not ok else '')}"
        with st.expander(head, expanded=False):
            st.code(q["sql"], language="sql")
            if ok:
                st.caption(f"{q['n']} row(s)" + (f" — showing first {C.ROW_CAP}" if q.get("truncated") else ""))
                st.dataframe(q["rows"], width="stretch", height=min(320, 60 + 28 * min(len(q["rows"]), 8)))
            else:
                st.error(q["error"])


def render_answer(text: str) -> None:
    """Render the answer, surfacing any [GK] general-knowledge lines as visible warnings."""
    if not text:
        return
    buf: list[str] = []
    for para in text.split("\n"):
        if para.strip().startswith("[GK]"):
            if buf:
                st.markdown("\n".join(buf))
                buf = []
            st.warning(re.sub(r"^\s*\[GK\]\s*", "", para) + "\n\n*— general knowledge, not from your data*",
                       icon="⚠️")
        else:
            buf.append(para)
    if buf:
        st.markdown("\n".join(buf))


def render_badge(a: C.Answer) -> None:
    if a.demo:
        st.info("**Demo mode** — canned query, no model call.", icon="🧪")
        return
    if a.data_backed:
        tables = ", ".join(a.tables_used) or "warehouse"
        n_ok = sum(1 for q in a.queries if q.get("ok"))
        st.success(f"**From your data** — {n_ok} quer{'y' if n_ok == 1 else 'ies'} against `{tables}`.",
                   icon="✅")
    else:
        st.warning("**Not from your data** — no query ran for this answer, so nothing here is backed "
                   "by your warehouse. Treat it as general knowledge.", icon="⚠️")


for turn in st.session_state.chat:
    with st.chat_message(turn["role"]):
        if turn["role"] == "user":
            st.markdown(turn["content"])
        else:
            a: C.Answer = turn["answer"]
            if a.error:
                st.error(a.error)
            render_answer(a.text)
            if a.queries:
                render_queries(a.queries)
            render_badge(a)

if prompt := st.chat_input("e.g. when was white maize last more than 1.5 sigma rich, and what happened next?"):
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    history = [{"role": t["role"], "content": t["content"] if t["role"] == "user" else t["answer"].text}
               for t in st.session_state.chat if t["role"] == "user" or t["answer"].text]
    with st.chat_message("assistant"):
        with st.spinner("Querying the warehouse…"):
            a = C.ask(history, data_only=data_only)
        if a.error:
            st.error(a.error)
        render_answer(a.text)
        if a.queries:
            render_queries(a.queries)
        render_badge(a)
    st.session_state.chat.append({"role": "assistant", "content": a.text, "answer": a})

with st.sidebar:
    if st.button("Clear conversation", width="stretch"):
        st.session_state.chat = []
        st.rerun()

with st.expander("What this can and cannot do"):
    st.markdown(f"""
**Can**
- Query every table in the warehouse: `prices`, `balance_sheet` (point-in-time SAGIS),
  `macro`, `macro_snap` (10:00 UTC parity snapshots), `signals` (fair value, residuals, z by date),
  `ingest_log`.
- Explain the methodology behind the models, because it is described in its system prompt.

**Cannot**
- Write anything. The connection is read-only, and only a single `SELECT`/`WITH` per call is accepted —
  writes, DDL and multi-statement input are rejected before execution.
- Fetch new data, or call the Python model functions directly. If a number is not in the warehouse,
  rebuild it with `python ingest/build_signals.py` first.

**How provenance is enforced**
1. The badge under every answer is computed by this page from the queries that actually executed.
   Zero successful queries always renders *Not from your data*, whatever the answer claims.
2. Data-only mode (default on) instructs the model to decline rather than guess.
3. With data-only off, any sentence not derived from a query must be marked `[GK]`, and this page
   renders those as amber warnings.

Model: `{C.MODEL}`. Results are capped at {C.ROW_CAP} rows per query and {C.MAX_ROUNDS} query rounds per question.
""")
