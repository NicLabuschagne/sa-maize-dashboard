"""Plotly helpers. One palette, fixed hue-per-entity, recessive grid, no dual axes."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Palette derived from bamfunds.com design tokens, validated for CVD separation on a white surface.
# Entities keep their slot regardless of what else is plotted. (light, dark) pairs.
_SLOTS = {
    "white":  ("#0054cc", "#5b99f1"),   # BAM primary blue
    "yellow": ("#eb6834", "#d95926"),
    "total":  ("#4a3aa7", "#9085e9"),
    "aux":    ("#118554", "#2fb377"),   # BAM green
    "aux2":   ("#8c8c85", "#bdbdbd"),   # BAM gray-600
}
_SYMBOL_CLASS = {"WMAZ": "white", "YMAZ": "yellow"}
_MUTED = ("#bdbdbd", "#52514e")
_GRID = ("#dfdfdb", "#2a2a28")
_TEXT = ("#101010", "#ffffff")
_TEXT2 = ("#8c8c85", "#c3c2b7")
STATUS = {"good": "#118554", "bad": "#dc2626", "navy": "#00204d"}


def _dark() -> bool:
    # The theme is fixed to light in .streamlit/config.toml; dark slots are kept for a future toggle.
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
                   highlight: int = 2, weekly: bool = False) -> go.Figure:
    """Columns = seasons, index = marketing-year month (or week if `weekly`).
    Latest `highlight` seasons in colour, rest muted."""
    fig = go.Figure()
    cols = list(piv.columns)
    months = ["May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr"]
    for c in cols[:-highlight]:
        fig.add_trace(go.Scatter(x=piv.index, y=piv[c], name=c, mode="lines", showlegend=False,
                                 line=dict(color=muted(), width=1), opacity=0.5,
                                 hovertemplate="%{y:,.0f}<extra>" + c + "</extra>"))
    for c, ent in zip(cols[-highlight:], ["aux2", entity][-highlight:]):
        fig.add_trace(go.Scatter(x=piv.index, y=piv[c], name=c, mode="lines+markers",
                                 line=dict(color=color(ent), width=2.5), marker=dict(size=4 if weekly else 7),
                                 hovertemplate="%{y:,.0f}<extra>" + c + "</extra>"))
    layout(fig, title, ytitle=ytitle)
    if weekly:      # week 1 ends on the first Friday of May; ~4.35 weeks per month
        fig.update_xaxes(tickmode="array", tickvals=[1 + 4.35 * i for i in range(12)], ticktext=months)
    else:
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


def price_position_flow(px: pd.Series, position: pd.Series, flow_: pd.Series, entity: str,
                        title: str, price_label: str = "price", price_unit: str = "",
                        height: int = 560) -> go.Figure:
    """Three stacked panels on a shared time axis: price, modelled position, implied flow.

    The flow panel is the one that matters - position is the stock a trend follower holds,
    flow is what they had to buy or sell to get there, and only the flow moves a market.
    """
    from plotly.subplots import make_subplots

    d = _dark()
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.44, 0.28, 0.28],
                        subplot_titles=(price_label, "modelled position (−100% short … +100% long)",
                                        "implied flow — buying (green) / selling (red)"))
    fig.add_trace(go.Scatter(x=px.index, y=px, name=price_label, mode="lines",
                             line=dict(color=color(entity), width=2),
                             hovertemplate="%{y:,.0f} " + price_unit + "<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=position.index, y=position * 100, name="position", mode="lines",
                             line=dict(color=STATUS["navy"], width=1.5), fill="tozeroy",
                             fillcolor="rgba(0,32,77,0.16)",
                             hovertemplate="%{y:+.0f}%<extra>position</extra>"), row=2, col=1)
    cols = [STATUS["good"] if v >= 0 else STATUS["bad"] for v in flow_]
    fig.add_trace(go.Bar(x=flow_.index, y=flow_ * 100, name="flow", marker=dict(color=cols),
                         hovertemplate="%{y:+.1f}pp<extra>flow</extra>"), row=3, col=1)
    for r in (2, 3):
        fig.add_hline(y=0, line=dict(color=muted(), width=1), row=r, col=1)

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=_TEXT[d]), x=0, xanchor="left",
                   y=1.0, yanchor="top", pad=dict(t=6)),
        height=height, margin=dict(l=16, r=16, t=86, b=16), showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=_TEXT2[d], size=12), hovermode="x unified", bargap=0.1)
    for ann in fig.layout.annotations:
        ann.font = dict(size=11, color=_TEXT2[d])
        ann.x, ann.xanchor = 0, "left"
    grid = dict(gridcolor=_GRID[d], gridwidth=1, zeroline=False)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(**grid)
    fig.update_yaxes(tickformat=",", row=1, col=1)
    fig.update_yaxes(tickformat="+.0f", ticksuffix="%", range=[-105, 105], row=2, col=1)
    fig.update_yaxes(tickformat="+.0f", ticksuffix="pp", row=3, col=1)
    return fig


def equity_drawdown(curves: pd.DataFrame, legs: list[str], title: str = "", height: int = 520) -> go.Figure:
    """Growth of 1 for the portfolio (bold) and each leg (thin), with the portfolio drawdown below."""
    from plotly.subplots import make_subplots

    d = _dark()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.72, 0.28],
                        subplot_titles=("growth of 1", "portfolio drawdown"))
    leg_colours = [_SLOTS["yellow"][d], _SLOTS["aux"][d], _SLOTS["white"][d], _SLOTS["total"][d], _SLOTS["aux2"][d]]
    dashes = ["dash", "dot", "dashdot", "longdash", "longdashdot"]
    for i, name in enumerate(legs):
        fig.add_trace(go.Scatter(x=curves.index, y=curves[name], name=name, mode="lines",
                                 line=dict(color=leg_colours[i % 5], width=1.3, dash=dashes[i % 5]),
                                 hovertemplate="%{y:.3f}x<extra>" + name + "</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=curves.index, y=curves["Portfolio"], name="Portfolio", mode="lines",
                             line=dict(color=STATUS["navy"], width=2.5),
                             hovertemplate="%{y:.3f}x<extra>Portfolio</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=curves.index, y=curves["drawdown"] * 100, name="drawdown", mode="lines",
                             line=dict(color=STATUS["bad"], width=1.2), fill="tozeroy",
                             fillcolor="rgba(220,38,38,0.14)", showlegend=False,
                             hovertemplate="%{y:.1f}%<extra>drawdown</extra>"), row=2, col=1)
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=_TEXT[d]), x=0, xanchor="left", y=1.0, yanchor="top",
                   pad=dict(t=6)),
        height=height, margin=dict(l=16, r=16, t=86 if title else 56, b=16),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=_TEXT2[d], size=12),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0, font=dict(size=11)))
    for ann in fig.layout.annotations:
        ann.font = dict(size=11, color=_TEXT2[d])
        ann.x, ann.xanchor = 0, "left"
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=_GRID[d], gridwidth=1, zeroline=False)
    fig.update_yaxes(tickformat=".2f", ticksuffix="x", row=1, col=1)
    fig.update_yaxes(ticksuffix="%", tickformat=".0f", row=2, col=1)
    return fig
