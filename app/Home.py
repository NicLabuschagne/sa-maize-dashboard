import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st

from app.data.blotter import build_rows, next_release
from app.data.warehouse import load_balance_sheet, load_signals
from config import APP_TITLE

st.set_page_config(page_title=APP_TITLE, layout="wide")

NAVY, NAVY_HEAD, LINE = "#00204D", "#0A3266", "rgba(255,255,255,0.10)"
INK, INK_DIM = "#FFFFFF", "#9FB3CE"
RICH, CHEAP, FLAT = "#FF6B6B", "#3DD68C", "#9FB3CE"

st.markdown(f"""
<style>
.blotter {{ background:{NAVY}; border-radius:10px; padding:0; overflow:hidden;
            font-variant-numeric:tabular-nums; margin-bottom:10px; }}
.blotter table {{ width:100%; border-collapse:collapse; }}
.blotter th {{ background:{NAVY_HEAD}; color:{INK_DIM}; font-size:11px; font-weight:600;
               letter-spacing:.09em; text-transform:uppercase; padding:11px 16px; text-align:right;
               white-space:nowrap; }}
.blotter th.l, .blotter td.l {{ text-align:left; }}
.blotter td {{ color:{INK}; font-size:15px; padding:13px 16px; text-align:right;
               border-top:1px solid {LINE}; white-space:nowrap; }}
.blotter tr:hover td {{ background:rgba(255,255,255,0.045); }}
.sym {{ font-weight:700; letter-spacing:.02em; }}
.desc {{ color:{INK_DIM}; font-size:12px; display:block; margin-top:2px; font-weight:400; }}
.unit {{ color:{INK_DIM}; font-size:11px; margin-left:4px; }}
.rich {{ color:{RICH}; font-weight:600; }}
.cheap {{ color:{CHEAP}; font-weight:600; }}
.flat {{ color:{FLAT}; }}
.bar {{ display:inline-block; height:4px; border-radius:2px; vertical-align:middle; margin-left:8px; }}
.asof {{ color:{INK_DIM}; font-size:12px; }}
</style>
""", unsafe_allow_html=True)

sig = load_signals()
bs = load_balance_sheet()
rows = build_rows(sig)
nxt, days = next_release(bs.vintage_date.unique())
as_of = sig.vintage_date.max()

st.title("Maize Monitor")
c1, c2, c3 = st.columns([2, 1, 1])
c1.markdown(f"<span class='asof'>South Africa · point-in-time S&amp;D · "
            f"latest release <b>{pd.Timestamp(as_of):%d %b %Y}</b></span>", unsafe_allow_html=True)
c2.markdown(f"<span class='asof'>Next release <b>{nxt:%d %b %Y}</b>"
            f"{f' · in {days}d' if days is not None else ''}</span>" if nxt is not None
            else "<span class='asof'>Next release —</span>", unsafe_allow_html=True)
c3.markdown(f"<span class='asof'>{len(rows)} markets</span>", unsafe_allow_html=True)


def dev_cell(dev: float | None, z: float | None, pct: bool) -> str:
    if dev is None:
        return "<span class='flat'>—</span>"
    cls = "rich" if (z or 0) > 0.5 else "cheap" if (z or 0) < -0.5 else "flat"
    width = min(abs(z or 0) / 3 * 46, 46)
    colour = RICH if cls == "rich" else CHEAP if cls == "cheap" else FLAT
    suffix = "%" if pct else " pp"
    bar = f"<span class='bar' style='width:{width:.0f}px;background:{colour};opacity:.55'></span>"
    return f"<span class='{cls}'>{dev:+,.1f}{suffix}</span>{bar}"


body = []
for r in rows:
    price = f"{r.market:,.0f}" if r.unit == "R/t" and r.market is not None else (
        f"{r.market:+.1f}" if r.market is not None else "—")
    fair = f"{r.fair:,.0f}" if r.unit == "R/t" and r.fair is not None else (
        f"{r.fair:+.1f}" if r.fair is not None else "—")
    z = f"{r.z:+.2f}" if r.z is not None else "—"
    zc = "rich" if (r.z or 0) > 0.5 else "cheap" if (r.z or 0) < -0.5 else "flat"
    cover = f"{r.cover:.1f}" if r.cover is not None else "—"
    body.append(
        f"<tr><td class='l'><span class='sym'>{r.symbol}</span>"
        f"<span class='desc'>{r.description}</span></td>"
        f"<td>{price}<span class='unit'>{r.unit}</span></td>"
        f"<td>{fair}<span class='unit'>{r.unit}</span></td>"
        f"<td>{dev_cell(r.dev, r.z, r.dev_pct)}</td>"
        f"<td class='{zc}'>{z}</td>"
        f"<td>{nxt:%d %b}</td>"
        f"<td>{cover}<span class='unit'>mo</span></td></tr>")

st.markdown(
    "<div class='blotter'><table>"
    "<tr><th class='l'>Symbol</th><th>Market</th><th>Fair value</th><th>Deviation</th>"
    "<th>σ</th><th>Next release</th><th>Cover</th></tr>"
    + "".join(body) + "</table></div>", unsafe_allow_html=True)

st.caption(
    "Fair value is the expanding-window model fit, estimated only on data published before each "
    "release, so it never saw the point it prices. Deviation is market less fair value; σ is that "
    "deviation divided by its own rolling standard deviation — **positive means rich**. "
    "Cover is months of consumption held as stock, from the latest SAGIS release. "
    "Rows 3–7 are relative markets, quoted in percentage points rather than rand."
)

st.page_link("pages/0_Overview.py", label="Overview →")
st.page_link("pages/3_Fair_Value.py", label="Fair Value models →")
st.page_link("pages/5_Ask.py", label="Ask the desk →")
