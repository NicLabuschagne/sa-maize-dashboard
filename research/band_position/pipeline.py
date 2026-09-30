"""Assemble the daily study frame for one class, and replicate the existing Model A as the baseline.

`build_daily` is the single entry point used by both the full run and the truncation test, so the
test exercises exactly the code that produced the results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.data import fairvalue as FV
from app.data import features as F
from research.band_position import band, models, stocks
from research.band_position.inputs import Inputs


def build_daily(inputs: Inputs, grain_class: str, settings: dict) -> pd.DataFrame:
    """Daily rows for one class: SAFEX, world price, bands, positions and the three STU ratios."""
    symbol = settings["symbols"][grain_class]
    daily = band.daily_band_frame(inputs.prices, inputs.snapshots, inputs.parity, symbol, settings)
    monthly = stocks.monthly_stocks(inputs.balance_sheet)
    events = stocks.stocks_to_use_events(monthly, inputs.weekly, grain_class,
                                         settings["stocks"]["days_per_month"])
    ratios = stocks.daily_stocks_to_use(daily["date"], events)
    daily = daily.merge(ratios, on="date", how="left")
    daily["season_stage"] = daily["date"].dt.month.map(models.SEASON_STAGES)
    daily["grain_class"] = grain_class
    return daily


def fair_position_at(daily: pd.DataFrame, date: pd.Timestamp, settings: dict) -> float:
    """Primary model's fair position for the weekly snapshot on `date`, fitted on earlier weeks only.
    Used by the truncation test to compare a rebuilt-from-cut value with the full-run value."""
    stu_column, name = settings["stocks"]["primary"], settings["models"]["primary"]
    snapshots = models.weekly_snapshots(daily).dropna(subset=["position", stu_column])
    snapshots = snapshots[snapshots[stu_column] > 0].reset_index(drop=True)
    target = snapshots.index[snapshots["date"] == date]
    if len(target) == 0 or target[0] == 0:
        return np.nan
    model = models.fit_model(name, snapshots.iloc[:target[0]], stu_column, settings["models"]["fourier_harmonics"])
    return float(model.predict(snapshots.iloc[[target[0]]])[0])


def baseline_model_a(inputs: Inputs, grain_class: str, settings: dict) -> dict:
    """The existing levels regression (fairvalue.py Model A): log real front price on log months of
    cover plus season, fitted monthly on an expanding window. Reported as-is, plus the in-sample R^2 of
    the same regression in nominal log price, which shows how much a levels fit borrows from trend."""
    sd = F.sd_monthly(inputs.balance_sheet)
    continuous = F.continuous(inputs.prices)
    panel = FV.panel_price(sd, continuous, inputs.cpi, grain_class, settings["symbols"][grain_class])
    fit = FV.fit_expanding(panel)
    scored = fit.panel.dropna(subset=["fv"])
    naive = fit.panel["y"].expanding().mean().shift(1).loc[scored.index]
    oos = models.out_of_sample_r2(scored["y"], scored["fv"], naive)
    nominal = fit.panel.dropna(subset=["x"])
    design = np.column_stack([FV.fourier(nominal["my_month"]), nominal["x"]])
    y_nominal = np.log(nominal["close_1"].to_numpy())
    coefficients, *_ = np.linalg.lstsq(design, y_nominal, rcond=None)
    residual = y_nominal - design @ coefficients
    r2_nominal = 1 - residual @ residual / ((y_nominal - y_nominal.mean()) ** 2).sum()
    return {"grain_class": grain_class, "releases": int(len(fit.panel)), "r2_in_sample_real": fit.r2_full,
            "r2_in_sample_nominal": float(r2_nominal), "oos_r2_vs_expanding_mean": oos,
            "oos_releases": int(len(scored)), "slope_log_cover": float(fit.coef_full["x"]),
            "t_log_cover": float(fit.tstat_full["x"])}
