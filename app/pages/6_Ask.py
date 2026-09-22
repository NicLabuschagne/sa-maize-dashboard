import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.data import chat as C
from app.data import router as R
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
    _ok, _why = C.key_looks_valid(C.resolve_api_key())
    backend = st.radio(
        "Backend", ["Rules-based (free)", "Claude (needs API credit)"],
        index=0 if not _ok else 1, horizontal=True,
        help="Rules-based maps your question to a fixed set of SQL templates — deterministic, "
             "offline and free. Claude writes SQL from scratch and handles any phrasing.")
    use_llm = backend.startswith("Claude")
    if use_llm and not _ok:
        st.warning(f"No usable key — {_why}.", icon="🔑")

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
    if a.local:
        if a.data_backed:
            tables = ", ".join(a.tables_used) or "warehouse"
            st.success(f"**From your data** — rules-based backend, 1 query against `{tables}`. "
                       f"No model was called, so nothing here is generated text.", icon="✅")
        else:
            st.info("**Rules-based backend** — the question did not match a known pattern, so no "
                    "query ran and no data is shown.", icon="🧭")
        return
    if a.demo:
        st.info("**Demo mode** — canned query, no model call.", icon="🧪")
        return
    cost = f"  ·  ${a.cost_usd:.3f}" if a.usage else ""
    if a.data_backed:
        tables = ", ".join(a.tables_used) or "warehouse"
        n_ok = sum(1 for q in a.queries if q.get("ok"))
        st.success(f"**From your data** — {n_ok} quer{'y' if n_ok == 1 else 'ies'} against `{tables}`.{cost}",
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
            a = C.ask(history, data_only=data_only) if use_llm else C.local_answer(history)
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
**Two backends, same interface and the same provenance rules.**

*Rules-based (free, default)* — your question is matched against a fixed set of intents, entities are
extracted (class, model, threshold, horizon, year, top-N) and a parameterised SQL template is filled.
It is deterministic: it cannot invent a number or a column. If nothing matches it says so rather
than guessing. No network, no account, no cost.

*Claude* — writes SQL from scratch against the schema, handles any phrasing, and can chain several
queries. Needs API credit on the Anthropic account.

**Both can** query `prices`, `balance_sheet` (point-in-time SAGIS), `macro`, `macro_snap`
(10:00 UTC parity snapshots), `signals` (fair value, residual, z by date) and `ingest_log`.

**Neither can** write anything: the connection is read-only and only a single `SELECT`/`WITH` per
call is accepted — writes, DDL and multi-statement input are rejected before execution.

**Provenance** — the badge under each answer is computed by this page from the queries that actually
executed. Zero successful queries always renders as not-from-your-data, whatever the answer says.

Questions the rules-based backend understands:

{chr(10).join("- " + e for e in R.EXAMPLES)}

Model when using Claude: `{C.MODEL}`. Caps: {C.ROW_CAP} rows per query, {C.MAX_ROUNDS} query rounds per question.
""")
