"""Derived series: point-in-time S&D state, continuous front-month prices, spreads.

Pure pandas, no Streamlit, so it is unit-testable. Every function returns a
DataFrame the pages can plot directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAIN_MONTHS = (3, 5, 7, 9, 12)  # liquid SAFEX maize delivery months
ROLL_DAYS = 7                   # roll to next contract this many days before expiry
CM_TENOR_DAYS = 90              # constant-maturity tenor: the longest gap between main months
STATE_ATTRS = ("opening_stock", "deliveries", "imports", "utilisation", "exports",
               "closing_stock", "human_consumption", "animal_feed")


def sd_monthly(bs: pd.DataFrame) -> pd.DataFrame:
    """Latest-month figures per vintage: one row per (vintage_date, grain_class).

    Uses only the `latest_month` block of each regular release, i.e. what was
    known on vintage_date. Finals are excluded (they restate a closed season).
    """
    d = bs[(bs.period_type == "latest_month") & (~bs.is_final) & bs.attribute.isin(STATE_ATTRS)]
    w = d.pivot_table(index=["vintage_date", "latest_month", "marketing_year", "grain_class"],
                      columns="attribute", values="value_t").reset_index()
    w = w.sort_values(["grain_class", "vintage_date"]).reset_index(drop=True)
    # trailing 12-month utilisation and exports (sum of the monthly preliminaries as published)
    g = w.groupby("grain_class")
    for col in ("utilisation", "exports", "deliveries"):
        w[f"{col}_12m"] = g[col].transform(lambda s: s.rolling(12, min_periods=9).sum())
    w["disappearance_12m"] = w["utilisation_12m"] + w["exports_12m"]
    w["months_cover"] = w["closing_stock"] / (w["disappearance_12m"] / 12)
    w["stocks_to_use"] = w["closing_stock"] / w["disappearance_12m"]
    w["my_month"] = ((w["latest_month"].dt.month - 5) % 12) + 1  # 1 = May .. 12 = Apr
    return w


def sd_revisions(bs: pd.DataFrame, attribute: str = "closing_stock", grain_class: str = "total") -> pd.DataFrame:
    """Preliminary vs revised value for each month: latest_month at vintage v vs prev_month at v+1."""
    d = bs[(bs.attribute == attribute) & (bs.grain_class == grain_class) & (~bs.is_final)]
    prelim = d[d.period_type == "latest_month"][["latest_month", "vintage_date", "value_t"]]
    prelim = prelim.rename(columns={"value_t": "preliminary", "vintage_date": "prelim_vintage"})
    # prev_month block refers to the month before latest_month
    rev = d[d.period_type == "prev_month"].copy()
    rev["month"] = rev["latest_month"] - pd.offsets.MonthBegin(1)
    rev = rev[["month", "vintage_date", "value_t"]].rename(
        columns={"value_t": "revised", "vintage_date": "revised_vintage"})
    out = prelim.rename(columns={"latest_month": "month"}).merge(rev, on="month", how="left")
    out["revision"] = out["revised"] - out["preliminary"]
    out["revision_pct"] = out["revision"] / out["preliminary"].abs().replace(0, np.nan) * 100
    return out.sort_values("month").reset_index(drop=True)


def ytd_by_season(bs: pd.DataFrame, attribute: str, grain_class: str = "total") -> pd.DataFrame:
    """Season-to-date path of an attribute by marketing-year month, one column per season."""
    d = bs[(bs.attribute == attribute) & (bs.grain_class == grain_class)
           & (bs.period_type == "ytd") & (~bs.is_final)].copy()
    d["my_month"] = ((d["latest_month"].dt.month - 5) % 12) + 1
    return d.pivot_table(index="my_month", columns="marketing_year", values="value_t")


def _contract_rank(p: pd.DataFrame) -> pd.DataFrame:
    """Rank live main-month contracts by expiry on each trade date after applying the roll rule."""
    live = p[(p.expiry_date.dt.month.isin(MAIN_MONTHS)) & (p.days_to_expiry >= ROLL_DAYS) & (p.close > 0)].copy()
    live["rank"] = live.groupby(["symbol", "trade_date"])["expiry_date"].rank(method="first").astype(int)
    return live


def constant_maturity(p: pd.DataFrame, tenor_days: int = CM_TENOR_DAYS) -> pd.DataFrame:
    """Price at a fixed `tenor_days` to expiry, per symbol and trade date, with no jump at rolls.

    The front month switches from the old-crop March contract to new-crop May in one day, and the
    front close can move 30-40% on that day with no trade taking place. Interpolating log price
    linearly in days to expiry between the main-month contracts either side of the tenor turns the
    switch into a gradual blend. Used for price *levels* in the fair-value models; tradeable returns
    stay on the roll-adjusted front month. If every live contract is beyond the tenor the nearest is
    used; if none reaches it the day has no value.
    """
    live = p[(p.expiry_date.dt.month.isin(MAIN_MONTHS)) & (p.days_to_expiry >= ROLL_DAYS) & (p.close > 0)]
    live = live[["symbol", "trade_date", "days_to_expiry", "close"]].sort_values(["symbol", "trade_date", "days_to_expiry"])
    rows = []
    for (symbol, date), g in live.groupby(["symbol", "trade_date"], sort=True):
        days, close = g["days_to_expiry"].to_numpy(dtype=float), g["close"].to_numpy(dtype=float)
        above = np.flatnonzero(days >= tenor_days)
        if len(above) == 0:
            continue
        upper = above[0]
        if upper == 0:
            price = close[0]
        else:
            w = (tenor_days - days[upper - 1]) / (days[upper] - days[upper - 1])
            price = float(np.exp((1 - w) * np.log(close[upper - 1]) + w * np.log(close[upper])))
        rows.append((symbol, date, price))
    return pd.DataFrame(rows, columns=["symbol", "trade_date", "close_cm"])


def continuous(p: pd.DataFrame) -> pd.DataFrame:
    """Front (rank 1) and second (rank 2) main-month contracts per symbol/date, plus the calendar spread."""
    live = _contract_rank(p)
    cols = ["symbol", "trade_date", "expiry", "close", "volume", "open_interest", "days_to_expiry"]
    f1 = live[live["rank"] == 1][cols].rename(columns={c: f"{c}_1" for c in cols[2:]})
    f2 = live[live["rank"] == 2][cols].rename(columns={c: f"{c}_2" for c in cols[2:]})
    c = f1.merge(f2, on=["symbol", "trade_date"], how="left")
    c["spread_2_1"] = c["close_2"] - c["close_1"]              # positive = carry
    c["spread_2_1_pct_ann"] = c["spread_2_1"] / c["close_1"] / (c["days_to_expiry_2"] - c["days_to_expiry_1"]) * 365 * 100
    # roll-adjusted log return series (returns within the front contract; no jump on roll day)
    c = c.sort_values(["symbol", "trade_date"])
    same = c.groupby("symbol")["expiry_1"].shift() == c["expiry_1"]
    c["log_ret_1"] = np.where(same, np.log(c["close_1"] / c.groupby("symbol")["close_1"].shift()), np.nan)
    c = c.merge(constant_maturity(p), on=["symbol", "trade_date"], how="left")
    return c.reset_index(drop=True)


def white_yellow_spread(cont: pd.DataFrame) -> pd.DataFrame:
    """White minus yellow: front-month spread on matching expiries, and the premium on the 90-day
    constant-maturity prices (`wy_spread_pct`), which does not jump when both legs roll to new crop."""
    w = cont[cont.symbol == "WMAZ"][["trade_date", "expiry_1", "close_1", "close_cm"]]
    y = cont[cont.symbol == "YMAZ"][["trade_date", "expiry_1", "close_1", "close_cm"]]
    m = w.merge(y, on=["trade_date", "expiry_1"], suffixes=("_w", "_y"))
    m["wy_spread"] = m["close_1_w"] - m["close_1_y"]
    m["wy_spread_pct"] = (m["close_cm_w"] / m["close_cm_y"] - 1) * 100
    return m


def align_price_to_vintage(cont: pd.DataFrame, sd: pd.DataFrame, symbol: str, grain_class: str) -> pd.DataFrame:
    """Join each S&D vintage to the front-month close on the first trade date >= vintage_date."""
    px = cont[cont.symbol == symbol][["trade_date", "close_1", "expiry_1"]].sort_values("trade_date")
    px["trade_date"] = px["trade_date"].astype("datetime64[ns]")
    s = sd[sd.grain_class == grain_class].sort_values("vintage_date").copy()
    s["vintage_date"] = s["vintage_date"].astype("datetime64[ns]")
    out = pd.merge_asof(s, px, left_on="vintage_date", right_on="trade_date", direction="forward")
    return out
