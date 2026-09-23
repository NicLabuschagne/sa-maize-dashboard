"""Release-fade portfolio engine: trade mechanics on synthetic prices, and a pin to the research result."""
import duckdb
import numpy as np
import pandas as pd
import pytest

from app.data import release_fade as RF
from app.data.backtest import CostModel
from config import DB_PATH

ZERO = CostModel(0.0, 0.0, 0.0)


def _idx(rets: list[float], start: str = "2024-01-01") -> pd.Series:
    d = pd.bdate_range(start, periods=len(rets) + 1)
    return pd.Series(np.exp(np.r_[0.0, np.cumsum(rets)]), index=d.as_unit("ns"))


def test_fades_rich_and_holds_fixed_sessions() -> None:
    idx = _idx([-0.01] * 30)                                 # price falls 1% a session
    ent = pd.DataFrame({"vintage_date": [idx.index[0]], "z": [1.5]})   # rich -> short
    r = RF.run_leg(idx, ent, hold=10, costs=ZERO)
    t = r["trades"].iloc[0]
    assert t.side == "short" and t.sessions == 10 and t.exit_on == "time"
    assert t.net_pct == pytest.approx((np.exp(0.10) - 1) * 100)
    assert r["daily"].sum() == pytest.approx(0.10)


def test_open_entry_adds_first_session_move() -> None:
    idx = _idx([0.0] * 20)
    ent = pd.DataFrame({"vintage_date": [idx.index[0]], "z": [-1.2]})  # cheap -> long
    oc = pd.Series(0.02, index=idx.index)                   # mark 2% above the open every day
    r = RF.run_leg(idx, ent, hold=5, costs=ZERO, oc=oc)
    assert r["trades"].iloc[0].net_pct == pytest.approx((np.exp(0.02) - 1) * 100)


def test_hard_stop_exits_early() -> None:
    idx = _idx([0.03] * 20)                                 # rallies against a short
    ent = pd.DataFrame({"vintage_date": [idx.index[0]], "z": [2.0]})
    t = RF.run_leg(idx, ent, hold=10, costs=ZERO, stop_pct=5)["trades"].iloc[0]
    assert t.exit_on == "stop" and t.sessions == 2           # -3%, then -6% breaches -5%


def test_one_position_per_leg() -> None:
    idx = _idx([0.0] * 40)
    ent = pd.DataFrame({"vintage_date": [idx.index[0], idx.index[3], idx.index[20]], "z": [1.1, 1.3, 1.2]})
    t = RF.run_leg(idx, ent, hold=10, costs=ZERO)["trades"]
    assert len(t) == 2                                       # the second release lands mid-trade


def test_parity_cost_includes_hedge() -> None:
    base = CostModel()
    assert RF.leg_costs(RF.LEGS["YMAZ parity"]).round_trip_r_t == pytest.approx(base.round_trip_r_t + 2.0)
    assert RF.leg_costs(RF.LEGS["YMAZ spread"]).round_trip_r_t == pytest.approx(2 * base.round_trip_r_t)
    assert RF.leg_costs(RF.LEGS["YMAZ outright"], 2.0).round_trip_r_t == pytest.approx(2 * base.round_trip_r_t)


def test_metrics_definitions() -> None:
    r = pd.Series([0.01, -0.02, 0.0, 0.02, -0.005] * 60)       # positive drift
    m = RF.metrics(r)
    assert m["Calmar"] == pytest.approx(m["CAGR %"] / abs(m["Max DD %"]))
    assert m["Sortino"] > m["Sharpe"]                         # downside deviation < total sd here


def test_reproduces_research_portfolio() -> None:
    """Default page configuration must match research/portfolio_5y.py (Sharpe 1.05, max DD -7.64%)."""
    if not DB_PATH.exists():
        pytest.skip("warehouse not built")
    from app.data import features as F
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        sig, px, snap = (con.execute(f"SELECT * FROM {t}").df() for t in ("signals", "prices", "macro_snap"))
    finally:
        con.close()
    cont = F.continuous(px)
    res = RF.run_portfolio(sig, RF.instruments(cont, px, snap), {k: 0.5 for k in RF.DEFAULT_LEGS},
                           "2021-09-01", "2026-12-31")
    m = RF.metrics(res["portfolio"], res["trades"])
    assert m["Trades"] == 49
    assert m["Sharpe"] == pytest.approx(1.05, abs=0.01)
    assert m["Max DD %"] == pytest.approx(-7.64, abs=0.05)
