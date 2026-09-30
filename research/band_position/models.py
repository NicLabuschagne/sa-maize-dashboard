"""Fair-position models: band position as a function of stocks-to-use and season.

    linear    position = a + season + b * STU
    log       position = a + season + b * log STU
    logistic  position = 1 / (1 + exp(-(a + season + b * log STU)))      non-linear least squares
    isotonic  position = non-increasing step function of STU (no season)
    naive     expanding mean of position (the benchmark)

Fitted on weekly snapshots. The out-of-sample prediction at week t comes from a fit on weeks before t.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import optimize, stats

MARKETING_YEAR_START_MONTH = 5
SEASON_STAGES = {5: "harvest (May–Jul)", 6: "harvest (May–Jul)", 7: "harvest (May–Jul)",
                 8: "post-harvest (Aug–Oct)", 9: "post-harvest (Aug–Oct)", 10: "post-harvest (Aug–Oct)",
                 11: "mid-season (Nov–Jan)", 12: "mid-season (Nov–Jan)", 1: "mid-season (Nov–Jan)",
                 2: "pre-harvest (Feb–Apr)", 3: "pre-harvest (Feb–Apr)", 4: "pre-harvest (Feb–Apr)"}


# ----------------------------------------------------------------------------- sample construction
def marketing_year_start(dates: pd.Series) -> pd.Series:
    """1 May of the marketing year each date belongs to (May–April)."""
    dates = pd.to_datetime(dates)
    year = dates.dt.year - (dates.dt.month < MARKETING_YEAR_START_MONTH).astype(int)
    return pd.to_datetime(year.astype(str) + "-05-01")


def season_terms(dates: pd.Series, harmonics: int) -> np.ndarray:
    """Fourier terms on the fraction of the marketing year elapsed. Same idea as fairvalue.fourier, but
    on days rather than months because the sample is weekly."""
    dates = pd.to_datetime(dates)
    fraction = ((dates - marketing_year_start(dates)).dt.days / 365.25).to_numpy()
    columns = []
    for k in range(1, harmonics + 1):
        columns += [np.sin(2 * np.pi * k * fraction), np.cos(2 * np.pi * k * fraction)]
    return np.column_stack(columns) if columns else np.empty((len(dates), 0))


def weekly_snapshots(daily: pd.DataFrame) -> pd.DataFrame:
    """Last trading day of each calendar week (weeks end Friday). Fitting on weekly rather than daily
    rows avoids counting the same slow-moving stocks information five times."""
    week = daily["date"].dt.to_period("W-FRI")
    return daily.groupby(week, sort=True).tail(1).reset_index(drop=True)


def first_out_of_sample_date(band_start: pd.Timestamp, min_seasons: int) -> pd.Timestamp:
    """The first 1 May preceded by `min_seasons` complete marketing years of band history."""
    band_start = pd.Timestamp(band_start)
    first_full_year = marketing_year_start(pd.Series([band_start])).iloc[0]
    if first_full_year < band_start:
        first_full_year = first_full_year + pd.DateOffset(years=1)
    return first_full_year + pd.DateOffset(years=min_seasons)


# ----------------------------------------------------------------------------- models
@dataclass
class FittedModel:
    """A fitted model: its name, the STU slope (NaN for isotonic) and a predict function."""

    name: str
    slope: float
    predict: Callable[[pd.DataFrame], np.ndarray]


def _design(frame: pd.DataFrame, stu_column: str, transform: str, harmonics: int) -> np.ndarray:
    """Constant, season terms and the (possibly logged) stocks-to-use column."""
    stu = frame[stu_column].to_numpy(dtype=float)
    x = np.log(stu) if transform == "log" else stu
    return np.column_stack([np.ones(len(frame)), season_terms(frame["date"], harmonics), x])


def _ols(design: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares coefficients."""
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coefficients


def _logistic(z: np.ndarray) -> np.ndarray:
    """Numerically safe logistic function."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def fit_model(name: str, train: pd.DataFrame, stu_column: str, harmonics: int) -> FittedModel:
    """Fit one of the four candidate shapes on `train` (columns: date, position, stu_column)."""
    y = train["position"].to_numpy(dtype=float)
    if name in ("linear", "log"):
        transform = "log" if name == "log" else "level"
        coefficients = _ols(_design(train, stu_column, transform, harmonics), y)
        return FittedModel(name, float(coefficients[-1]),
                           lambda f: _design(f, stu_column, transform, harmonics) @ coefficients)
    if name == "logistic":
        design = _design(train, stu_column, "log", harmonics)
        start = _ols(design, np.log(np.clip(y, 0.02, 0.98) / (1 - np.clip(y, 0.02, 0.98))))
        result = optimize.least_squares(lambda b: _logistic(design @ b) - y, start, method="lm")
        coefficients = result.x
        return FittedModel(name, float(coefficients[-1]),
                           lambda f: _logistic(_design(f, stu_column, "log", harmonics) @ coefficients))
    if name == "isotonic":
        order = np.argsort(train[stu_column].to_numpy())
        x_sorted = train[stu_column].to_numpy()[order]
        fitted = optimize.isotonic_regression(y[order], increasing=False).x
        return FittedModel(name, np.nan, lambda f: np.interp(f[stu_column].to_numpy(), x_sorted, fitted))
    raise ValueError(f"unknown model {name}")


def expanding_predictions(snapshots: pd.DataFrame, name: str, stu_column: str, first_date: pd.Timestamp,
                          harmonics: int) -> pd.DataFrame:
    """Out-of-sample fair position for every week from `first_date`: refit on all earlier weeks, predict
    this one. Also returns the expanding-mean benchmark and the slope of each fit."""
    data = snapshots.dropna(subset=["position", stu_column]).reset_index(drop=True)
    data = data[data[stu_column] > 0].reset_index(drop=True)
    prediction, naive, slope = (np.full(len(data), np.nan) for _ in range(3))
    for t in np.flatnonzero(data["date"] >= first_date):
        train = data.iloc[:t]
        model = fit_model(name, train, stu_column, harmonics)
        prediction[t] = model.predict(data.iloc[[t]])[0]
        naive[t] = train["position"].mean()
        slope[t] = model.slope
    return pd.DataFrame({"date": data["date"], "position": data["position"], "stu": data[stu_column],
                         "fair_position": prediction, "naive": naive, "slope": slope})


# ----------------------------------------------------------------------------- scoring
def out_of_sample_r2(actual: pd.Series, predicted: pd.Series, benchmark: pd.Series) -> float:
    """1 - SSE(model) / SSE(benchmark) on rows where all three exist. Above 0 = beats the benchmark."""
    frame = pd.DataFrame({"a": actual, "p": predicted, "b": benchmark}).dropna()
    if len(frame) < 10:
        return np.nan
    return float(1 - ((frame.a - frame.p) ** 2).sum() / ((frame.a - frame.b) ** 2).sum())


def regime_labels(position: pd.Series, mid_low: float, mid_high: float) -> pd.Series:
    """'mid' if the previous week's position was inside [mid_low, mid_high], else 'edge'. The previous
    week is used so a regime is known before the week it labels."""
    previous = position.shift(1)
    labels = np.where(previous.between(mid_low, mid_high), "mid", "edge")
    return pd.Series(np.where(previous.isna(), None, labels), index=position.index)


def score_predictions(predictions: pd.DataFrame, mid_low: float, mid_high: float) -> dict:
    """Out-of-sample R^2 overall, by half of the out-of-sample period, and by regime."""
    scored = predictions.dropna(subset=["fair_position"]).copy()
    scored["regime"] = regime_labels(predictions["position"], mid_low, mid_high).loc[scored.index]
    halves = np.array_split(scored.index.to_numpy(), 2)
    result = {"weeks": int(len(scored)),
              "oos_r2": out_of_sample_r2(scored.position, scored.fair_position, scored.naive),
              "first_oos_week": scored["date"].min()}
    for i, rows in enumerate(halves, start=1):
        part = scored.loc[rows]
        result[f"oos_r2_half{i}"] = out_of_sample_r2(part.position, part.fair_position, part.naive)
        result[f"slope_half{i}"] = float(part["slope"].mean())
    for regime in ("mid", "edge"):
        part = scored[scored.regime == regime]
        result[f"weeks_{regime}"] = int(len(part))
        result[f"oos_r2_{regime}"] = out_of_sample_r2(part.position, part.fair_position, part.naive)
    return result


def score_windows(predictions: pd.DataFrame, windows: list[list[str]]) -> pd.DataFrame:
    """Out-of-sample R^2 by calendar year and for fixed date windows. The fits are unchanged (expanding
    from the first out-of-sample week); a window only selects which predictions are scored."""
    scored = predictions.dropna(subset=["fair_position"])
    periods = [(str(year), f"{year}-01-01", f"{year}-12-31") for year in sorted(scored["date"].dt.year.unique())]
    periods += [(f"{start[:7]} to {end[:7]}", start, end) for start, end in windows]
    rows = []
    for label, start, end in periods:
        part = scored[(scored["date"] >= start) & (scored["date"] <= end)]
        rows.append({"period": label, "weeks": int(len(part)),
                     "oos_r2": out_of_sample_r2(part.position, part.fair_position, part.naive),
                     "mean_abs_error": float((part.position - part.fair_position).abs().mean()),
                     "mean_abs_error_naive": float((part.position - part.naive).abs().mean())})
    return pd.DataFrame(rows)


def score_by_stage(predictions: pd.DataFrame, snapshots: pd.DataFrame, stu_column: str) -> pd.DataFrame:
    """Per season stage: out-of-sample R^2 and the full-sample Spearman of STU with position within
    the stage (descriptive), so a weak stage can be separated from a missing one."""
    scored = predictions.dropna(subset=["fair_position"]).copy()
    scored["stage"] = scored["date"].dt.month.map(SEASON_STAGES)
    sample = snapshots.dropna(subset=["position", stu_column]).copy()
    sample["stage"] = sample["date"].dt.month.map(SEASON_STAGES)
    rows = []
    for stage in dict.fromkeys(SEASON_STAGES.values()):
        part, within = scored[scored.stage == stage], sample[sample.stage == stage]
        rows.append({"stage": stage, "weeks_all": int(len(within)), "weeks_oos": int(len(part)),
                     "stu_median": float(within[stu_column].median()),
                     "spearman_within_stage": float(stats.spearmanr(within[stu_column], within["position"]).statistic),
                     "oos_r2": out_of_sample_r2(part.position, part.fair_position, part.naive)})
    return pd.DataFrame(rows)


def binned_means(snapshots: pd.DataFrame, stu_column: str, bins: int = 5) -> pd.DataFrame:
    """Mean and spread of position by STU quintile: a model-free look at the shape (descriptive)."""
    data = snapshots.dropna(subset=["position", stu_column])
    data = data.assign(bin=pd.qcut(data[stu_column], bins, labels=False, duplicates="drop"))
    table = data.groupby("bin").agg(stu_low=(stu_column, "min"), stu_high=(stu_column, "max"),
                                    weeks=("position", "size"), mean_position=("position", "mean"),
                                    p25=("position", lambda s: s.quantile(0.25)),
                                    p75=("position", lambda s: s.quantile(0.75)))
    return table.reset_index()


def rolling_slope(snapshots: pd.DataFrame, stu_column: str, years: int, harmonics: int,
                  name: str = "log") -> pd.DataFrame:
    """Slope of model `name` fitted on the trailing `years` of weekly snapshots, at each week. Shows
    whether the relationship drifts in a way the expanding fit would average away."""
    data = snapshots.dropna(subset=["position", stu_column])
    data = data[data[stu_column] > 0].reset_index(drop=True)
    slopes = np.full(len(data), np.nan)
    for t in range(len(data)):
        window = data[(data["date"] > data["date"].iloc[t] - pd.DateOffset(years=years))
                      & (data["date"] <= data["date"].iloc[t])]
        if window["date"].iloc[0] <= data["date"].iloc[t] - pd.DateOffset(years=years) + pd.Timedelta(days=14):
            slopes[t] = fit_model(name, window, stu_column, harmonics).slope
    return pd.DataFrame({"date": data["date"], "rolling_slope": slopes})


def release_day_check(daily: pd.DataFrame, surprises: pd.DataFrame, n_permutations: int = 5000,
                      seed: int = 0) -> dict:
    """Does position move against the stock surprise on SAGIS release days?

    Outcome: position at the first close after the release date minus position at the last close
    before it (a two-close window, since the publication hour is not recorded). Spearman correlation
    with the surprise, and a permutation p-value. Releases are a month apart, so they do not overlap.
    """
    dates = daily["date"].to_numpy()
    position = daily["position"].to_numpy()
    rows = []
    for row in surprises.dropna(subset=["surprise"]).itertuples():
        if row.nowcast_weeks == 0:
            continue
        before = np.searchsorted(dates, np.datetime64(row.vintage_date), side="left") - 1
        after = np.searchsorted(dates, np.datetime64(row.vintage_date), side="right")
        if before < 0 or after >= len(dates):
            continue
        rows.append({"vintage_date": row.vintage_date, "surprise": row.surprise,
                     "position_change": position[after] - position[before]})
    events = pd.DataFrame(rows).dropna()
    if len(events) < 12:
        return {"releases": len(events), "spearman": np.nan, "p_permutation": np.nan}
    observed = stats.spearmanr(events.surprise, events.position_change).statistic
    rng = np.random.default_rng(seed)
    ranks_x, ranks_y = stats.rankdata(events.surprise), stats.rankdata(events.position_change)
    null = np.array([np.corrcoef(ranks_x, rng.permutation(ranks_y))[0, 1] for _ in range(n_permutations)])
    return {"releases": int(len(events)), "spearman": float(observed),
            "p_permutation": float((np.abs(null) >= abs(observed)).mean()), "events": events}


def residual_autocorrelation(predictions: pd.DataFrame, max_lag: int = 26) -> pd.Series:
    """Autocorrelation of the out-of-sample residual (position - fair position) at weekly lags. High
    persistence means deviations last and each week adds little independent information."""
    residual = (predictions["position"] - predictions["fair_position"]).dropna()
    return pd.Series([residual.autocorr(lag) for lag in range(1, max_lag + 1)], index=range(1, max_lag + 1))
