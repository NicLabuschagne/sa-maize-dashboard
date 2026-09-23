import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from app.data import fairvalue as FV
from app.data import trend as TR
from app.data.blotter import MARKETS, ROW_SYMBOL, build_rows, next_release, signal
from app.data.warehouse import load_balance_sheet, load_signals
from app.state import derived
from config import APP_TITLE

st.set_page_config(page_title=APP_TITLE, layout="wide")

NAVY, BAND, LINE = "#00204D", "#00193C", "rgba(255,255,255,0.13)"
INK, DIM = "#FFFFFF", "#8FA6C4"

st.markdown(f"""
<style>
/* full-bleed navy: the monitor owns the whole viewport */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background:{NAVY}; }}
[data-testid="stHeader"] {{ background:transparent; }}
[data-testid="stMain"] .block-container {{ padding:2.2rem 2.6rem 3rem; max-width:none; }}
[data-testid="stMain"] h1 {{ color:{INK}; font-size:2.1rem; letter-spacing:-.01em; margin-bottom:.15rem; }}
[data-testid="stMain"] [data-testid="stCaptionContainer"],
[data-testid="stMain"] [data-testid="stCaptionContainer"] p {{ color:{DIM}; }}
[data-testid="stMain"] a {{ color:#8FB8FF; }}

table.blot {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums;
              margin-top:1.1rem; }}
table.blot th {{ color:{DIM}; font-size:10.5px; font-weight:600; letter-spacing:.12em;
                 text-transform:uppercase; padding:0 14px 10px; text-align:right;
                 white-space:nowrap; border-bottom:1px solid {LINE}; }}
table.blot th.l, table.blot td.l {{ text-align:left; }}
table.blot td {{ color:{INK}; font-size:15.5px; padding:14px; text-align:right;
                 white-space:nowrap; border-bottom:1px solid {LINE}; }}
table.blot tr:hover td {{ background:rgba(255,255,255,.05); }}

table.blot td.grp {{ background:{BAND}; color:{DIM}; text-align:left; font-size:10.5px; font-weight:700;
          letter-spacing:.16em; text-transform:uppercase; padding:9px 14px;
          border-top:1px solid {LINE}; border-bottom:1px solid {LINE}; }}
.sym {{ font-weight:700; }}
.desc {{ color:{DIM}; font-size:11.5px; display:block; margin-top:3px; font-weight:400; }}
.unit {{ color:{DIM}; font-size:11px; margin-left:5px; }}
.pill {{ display:inline-block; min-width:96px; padding:5px 10px; border-radius:5px;
         font-weight:700; font-size:14px; }}
.act {{ font-size:10px; letter-spacing:.1em; opacity:.85; margin-right:7px; }}
.legend {{ color:{DIM}; font-size:11.5px; margin-top:.9rem; }}
.chip {{ display:inline-block; width:26px; height:11px; border-radius:3px;
         vertical-align:middle; margin:0 5px 0 12px; }}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def trend_by_symbol() -> dict:
    """Aggregate trend-model position per outright market, daily."""
    return {s_: TR.aggregate(TR.trend_panel(FV.roll_adjusted_index(derived()["cont"], s_)))
            for s_ in ("WMAZ", "YMAZ")}


sig_df = load_signals()
bs = load_balance_sheet()
releases = sorted(pd.to_datetime(sig_df.vintage_date).unique())

head, pick = st.columns([3, 1])
with head:
    st.title("Maize Monitor")
with pick:
    as_of = st.selectbox("As of release", releases[::-1], index=0,
                         format_func=lambda d: pd.Timestamp(d).strftime("%d %b %Y"),
                         help="Replay the board as it stood at any past release. Every figure is "
                              "point-in-time, so nothing published later leaks in.")

as_of = pd.Timestamp(as_of)
rows = build_rows(sig_df, as_of)
nxt, days = next_release(bs[pd.to_datetime(bs.vintage_date) <= as_of].vintage_date.unique(), today=as_of)
nxt_txt = f"{nxt:%d %b %Y}" if nxt is not None else "—"
live = as_of == pd.Timestamp(max(releases))
# Live board: flow to the latest session. Replay: flow as it stood on the replayed release date,
# so a 2016 board shows 2016 flow rather than today's.
flows = {s_: st_ for s_, a_ in trend_by_symbol().items()
         if (st_ := TR.flow_state(a_ if live else a_.loc[:as_of]))}
st.caption(f"South Africa · point-in-time supply & demand · "
           f"{'latest release' if live else 'replayed at release'} **{as_of:%d %b %Y}** · "
           f"next release **{nxt_txt}**{f' (in {days} days)' if days is not None else ''} · "
           f"{len(rows)} markets")

BUY, SELL = "#2FCF87", "#FF5C5C"


def flow_cell(symbol: str | None) -> str:
    """One-week implied trend flow: what systematic momentum has been buying or selling."""
    st_ = flows.get(symbol or "")
    if not st_:
        return "<span class='flat'>—</span>"
    f = st_["flow"] * 100
    if abs(f) < 2:
        return f"<span class='flat'>{f:+.0f}pp</span>"
    colour = BUY if f > 0 else SELL
    return (f"<span style='color:{colour};font-weight:600'>{f:+.0f}pp</span>"
            f"<span class='unit'>{'buy' if f > 0 else 'sell'}</span>")


html = ["<table class='blot'><tr>"
        "<th class='l'>Symbol</th><th>Market</th><th>Fair value</th><th>Deviation</th>"
        "<th>Signal (&sigma;)</th><th>Flow 1w</th><th>Next release</th><th>Cover</th></tr>"]

current_group = None
for r in rows:
    if r.group != current_group:
        current_group = r.group
        html.append(f"<tr><td class='grp' colspan='8'>{r.group}</td></tr>")

    money = r.unit == "R/t"
    market = ("—" if r.market is None else f"{r.market:,.0f}" if money else f"{r.market:+.1f}")
    fair = ("—" if r.fair is None else f"{r.fair:,.0f}" if money else f"{r.fair:+.1f}")
    dev = "—" if r.dev is None else f"{r.dev:+,.1f}{'%' if r.dev_pct else ' pp'}"
    cover = "—" if r.cover is None else f"{r.cover:.1f}"

    action, colour, alpha = signal(r.z)
    zt = "—" if r.z is None else f"{r.z:+.2f}"
    if alpha:
        rgb = tuple(int(colour[i:i + 2], 16) for i in (1, 3, 5))
        pill = (f"<span class='pill' style='background:rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha});"
                f"color:{colour};'><span class='act'>{action}</span>{zt}</span>")
        dev_html = f"<span style='color:{colour};font-weight:600'>{dev}</span>"
    else:
        pill = f"<span class='pill' style='background:rgba(255,255,255,.05);color:{DIM};'>{zt}</span>"
        dev_html = f"<span style='color:{DIM}'>{dev}</span>"

    html.append(
        f"<tr><td class='l'><span class='sym'>{r.symbol}</span>"
        f"<span class='desc'>{r.description}</span></td>"
        f"<td>{market}<span class='unit'>{r.unit}</span></td>"
        f"<td>{fair}<span class='unit'>{r.unit}</span></td>"
        f"<td>{dev_html}</td><td>{pill}</td>"
        f"<td>{flow_cell(ROW_SYMBOL.get(r.group))}</td>"
        f"<td>{nxt:%d %b}</td>"
        f"<td>{cover}<span class='unit'>mo</span></td></tr>")

html.append("</table>")
st.markdown("".join(html), unsafe_allow_html=True)

st.markdown(
    "<div class='legend'>"
    "<span class='chip' style='background:rgba(255,92,92,.55)'></span>rich vs fair value &rarr; sell"
    "<span class='chip' style='background:rgba(47,207,135,.55)'></span>cheap &rarr; buy"
    "<span class='chip' style='background:rgba(255,255,255,.05)'></span>within &plusmn;0.5&sigma; &rarr; no signal"
    "&nbsp;&nbsp;·&nbsp;&nbsp;shading intensity scales with severity, saturating at 2.5&sigma;."
    "</div>", unsafe_allow_html=True)

st.caption(
    "Fair value is the expanding-window model fit, estimated only on releases published before each "
    "observation, so it never saw the point it prices. Deviation is market less fair value in each "
    "market's own units; σ standardises it by its own rolling deviation. Cover is months of "
    "consumption held as stock at the latest SAGIS release."
)

c1, c2, c3, c4 = st.columns(4)
c1.page_link("pages/0_Overview.py", label="Overview →")
c2.page_link("pages/3_Fair_Value.py", label="Fair Value models →")
c3.page_link("pages/5_Positioning.py", label="Positioning →")
c4.page_link("pages/6_Ask.py", label="Ask the desk →")
