"""Plotly helpers. One palette, fixed hue-per-entity, recessive grid, no dual axes."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Categorical slots (light, dark). Entities keep their slot regardless of what else is plotted.
_SLOTS = {
    "white":  ("#2a78d6", "#3987e5"),
    "yellow": ("#eb6834", "#d95926"),
    "total":  ("#1baf7a", "#199e70"),
    "aux":    ("#eda100", "#c98500"),
    "aux2":   ("#4a3aa7", "#9085e9"),
}
_SYMBOL_CLASS = {"WMAZ": "white", "YMAZ": "yellow"}
_MUTED = ("#c3c2b7", "#52514e")
_GRID = ("#e8e7e3", "#2a2a28")
_TEXT = ("#0b0b0b", "#ffffff")
_TEXT2 = ("#52514e", "#c3c2b7")


def _dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:  # noqa: BLE001 - older Streamlit
        return False


def color(entity: str) -> str:
    key = _SYMBOL_CLASS.get(entity, entity)
    return _SLOTS.get(key, _SLOTS["aux"])[_dark()]


def muted() -> str:
    return _MUTED[_dark()]


def layout(fig: go.Figure, title: str = "", height: int = 380, ytitle: str = "", xtitle: str = "") -> go.Figure:
    d = _dark()
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=_TEXT[d]), x=0, xanchor="left",
                   y=1.0, yanchor="top", pad=dict(t=6)),
        height=height, margin=dict(l=16, r=16, t=72 if title else 36, b=16),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=_TEXT2[d], size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11)),
        hovermode="x unified",
    )
    grid = dict(gridcolor=_GRID[d], gridwidth=1, zeroline=False, linecolor=_GRID[d], showline=False)
    fig.update_xaxes(title=xtitle, showgrid=False, **{k: v for k, v in grid.items() if k != "gridcolor"})
    fig.update_yaxes(title=ytitle, tickformat=",", **grid)
    return fig


def line(fig: go.Figure, x: pd.Series, y: pd.Series, name: str, entity: str | None = None,
         width: float = 2, dash: str | None = None, opacity: float = 1.0, hover: str | None = None) -> go.Figure:
    fig.add_trace(go.Scatter(
        x=x, y=y, name=name, mode="lines", opacity=opacity,
        line=dict(color=color(entity or name), width=width, dash=dash),
        hovertemplate=hover or "%{y:,.0f}<extra>" + name + "</extra>"))
    return fig


def season_overlay(piv: pd.DataFrame, title: str, entity: str, ytitle: str = "t",
                   highlight: int = 2) -> go.Figure:
    """Columns = seasons, index = marketing-year month. Latest `highlight` seasons in colour, rest muted."""
    fig = go.Figure()
    cols = list(piv.columns)
    months = ["May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr"]
    for c in cols[:-highlight]:
        fig.add_trace(go.Scatter(x=piv.index, y=piv[c], name=c, mode="lines", showlegend=False,
                                 line=dict(color=muted(), width=1), opacity=0.5,
                                 hovertemplate="%{y:,.0f}<extra>" + c + "</extra>"))
    for c, ent in zip(cols[-highlight:], ["aux2", entity][-highlight:]):
        fig.add_trace(go.Scatter(x=piv.index, y=piv[c], name=c, mode="lines+markers",
                                 line=dict(color=color(ent), width=2.5), marker=dict(size=7),
                                 hovertemplate="%{y:,.0f}<extra>" + c + "</extra>"))
    layout(fig, title, ytitle=ytitle)
    fig.update_xaxes(tickmode="array", tickvals=list(range(1, 13)), ticktext=months)
    return fig


def add_vlines(fig: go.Figure, dates: pd.Series, label: str = "", xmax: pd.Timestamp | None = None) -> go.Figure:
    if xmax is not None:
        dates = dates[dates <= xmax]
    for d in dates:
        fig.add_vline(x=d, line=dict(color=muted(), width=1, dash="dot"), opacity=0.6)
    if label and len(dates):
        fig.add_annotation(x=dates.iloc[-1], y=1, yref="paper", text=label, showarrow=False,
                           font=dict(size=10, color=muted()), xanchor="left")
    return fig
