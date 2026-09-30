"""Run the band-position study end to end (see research/PREREGISTRATION_BAND.md).

    python -m research.band_position.run

Writes tables and plots to research/band_position/output/ and appends one line per run to
output/variants_log.csv, so the number of configurations tried is always on record.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research.band_position import band, models, pipeline, stocks  # noqa: E402
from research.band_position.inputs import Inputs, inputs_as_of, load_inputs  # noqa: E402
from research.band_position.settings import config_hash, load_settings  # noqa: E402

OUTPUT_DIR = Path(os.getenv("BAND_POSITION_OUTPUT", Path(__file__).with_name("output")))
GRAIN_CLASSES = ("white", "yellow")

# Reference categorical palette (fixed order) and inks, light surface.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_SECONDARY, INK_MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"


# ----------------------------------------------------------------------------- analysis
def fit_all_models(snapshots: pd.DataFrame, settings: dict, first_date: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """Every model x STU version for one class. Returns the score table and the prediction frames."""
    model_settings, regimes = settings["models"], settings["regimes"]
    rows, predictions = [], {}
    for stu_column in settings["stocks"]["variants"]:
        for name in model_settings["names"]:
            frame = models.expanding_predictions(snapshots, name, stu_column, first_date,
                                                 model_settings["fourier_harmonics"])
            predictions[(name, stu_column)] = frame
            score = models.score_predictions(frame, regimes["mid_low"], regimes["mid_high"])
            rows.append({"model": name, "stu": stu_column, "min_seasons": model_settings["min_seasons"], **score})
    return pd.DataFrame(rows), predictions


def sensitivity(snapshots: pd.DataFrame, settings: dict, band_start: pd.Timestamp) -> pd.DataFrame:
    """Primary model at the other pre-registered minimum windows. Reported, never used to choose."""
    model_settings, regimes = settings["models"], settings["regimes"]
    rows = []
    for seasons in model_settings["min_seasons_sensitivity"]:
        first_date = models.first_out_of_sample_date(band_start, seasons)
        frame = models.expanding_predictions(snapshots, model_settings["primary"], settings["stocks"]["primary"],
                                             first_date, model_settings["fourier_harmonics"])
        rows.append({"model": model_settings["primary"], "stu": settings["stocks"]["primary"],
                     "min_seasons": seasons, **models.score_predictions(frame, regimes["mid_low"], regimes["mid_high"])})
    return pd.DataFrame(rows)


def truncation_test(inputs: Inputs, daily_by_class: dict, fair_by_class: dict, settings: dict) -> pd.DataFrame:
    """Rebuild everything from inputs cut at t and compare values at t with the full run.

    Dates are sampled from out-of-sample weekly snapshots, which are the last trading day of their week
    in the full data, so the cut build sees the same snapshot date.
    """
    truncation = settings["truncation"]
    rng = np.random.default_rng(truncation["seed"])
    candidates = fair_by_class["white"].dropna(subset=["fair_position"])["date"].to_numpy()
    dates = np.sort(rng.choice(candidates, size=min(truncation["n_dates"], len(candidates)), replace=False))
    compared = ["hybrid_export", "hybrid_import", "position", "position_implied", settings["stocks"]["primary"]]
    rows = []
    for date in pd.DatetimeIndex(dates):
        cut = inputs_as_of(inputs, date)
        for grain_class in GRAIN_CLASSES:
            rebuilt = pipeline.build_daily(cut, grain_class, settings)
            full_row = daily_by_class[grain_class].set_index("date").loc[date]
            cut_row = rebuilt.set_index("date").loc[date]
            record = {"date": date, "grain_class": grain_class}
            for column in compared:
                record[f"diff_{column}"] = abs(float(full_row[column]) - float(cut_row[column]))
            full_fair = fair_by_class[grain_class].set_index("date")["fair_position"].get(date, np.nan)
            record["diff_fair_position"] = abs(full_fair - pipeline.fair_position_at(rebuilt, date, settings))
            rows.append(record)
    table = pd.DataFrame(rows)
    difference_columns = [c for c in table.columns if c.startswith("diff_")]
    table["max_diff"] = table[difference_columns].max(axis=1)
    table["passed"] = table["max_diff"] <= truncation["tolerance"]
    return table


# ----------------------------------------------------------------------------- plots
def _style(axis: plt.Axes, title: str, y_label: str) -> None:
    """Recessive grid and axes, text in ink tokens."""
    axis.set_facecolor(SURFACE)
    axis.set_title(title, loc="left", color=INK, fontsize=11)
    axis.set_ylabel(y_label, color=INK_SECONDARY, fontsize=9)
    axis.tick_params(colors=INK_MUTED, labelsize=8)
    axis.grid(color="#e6e5e1", linewidth=0.6)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(INK_MUTED)


def plot_price_with_band(daily: pd.DataFrame, grain_class: str, path: Path) -> None:
    """SAFEX front month inside the hybrid band, with the SAGIS band dashed for comparison."""
    figure, axis = plt.subplots(figsize=(11, 4.2), facecolor=SURFACE)
    axis.fill_between(daily["date"], daily["hybrid_export"], daily["hybrid_import"], color=SERIES_COLORS[0],
                      alpha=0.15, linewidth=0, label="Hybrid band (primary)")
    axis.plot(daily["date"], daily["sagis_export"], color=INK_MUTED, linewidth=1, linestyle="--", label="SAGIS band")
    axis.plot(daily["date"], daily["sagis_import"], color=INK_MUTED, linewidth=1, linestyle="--")
    axis.plot(daily["date"], daily["safex"], color=SERIES_COLORS[1], linewidth=1.2, label=f"SAFEX {grain_class} front")
    _style(axis, f"{grain_class.title()} maize: SAFEX inside the parity band", "R/t")
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def plot_positions(daily_by_class: dict, settings: dict, path: Path) -> None:
    """Band position over time for both classes, with the band edges and the mid-band regime marked."""
    figure, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True, facecolor=SURFACE)
    for axis, grain_class, color in zip(axes, GRAIN_CLASSES, SERIES_COLORS):
        daily = daily_by_class[grain_class]
        axis.axhspan(settings["regimes"]["mid_low"], settings["regimes"]["mid_high"], color="#f0efec", zorder=0)
        for edge in (0, 1):
            axis.axhline(edge, color=INK_MUTED, linewidth=0.8)
        axis.plot(daily["date"], daily["position"], color=color, linewidth=1)
        _style(axis, f"{grain_class.title()}: band position (0 = export edge, 1 = import edge; shaded = mid-band)",
               "position")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def plot_position_vs_stu(snapshots: pd.DataFrame, stu_column: str, grain_class: str, path: Path) -> None:
    """Weekly position against stocks-to-use, coloured by season stage (fixed order: harvest first)."""
    figure, axis = plt.subplots(figsize=(7.5, 5), facecolor=SURFACE)
    stages = list(dict.fromkeys(models.SEASON_STAGES.values()))
    for stage, color in zip(stages, SERIES_COLORS):
        part = snapshots[snapshots["season_stage"] == stage]
        axis.scatter(part[stu_column], part["position"], s=14, color=color, alpha=0.7, edgecolor=SURFACE,
                     linewidth=0.4, label=stage)
    axis.set_xscale("log")
    _style(axis, f"{grain_class.title()}: position vs {stu_column} (weekly)", "band position")
    axis.set_xlabel(f"{stu_column} (log scale)", color=INK_SECONDARY, fontsize=9)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def plot_slopes(slope_frames: dict, path: Path) -> None:
    """Expanding-fit slope and trailing-5-year slope of the log model, per class."""
    figure, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True, facecolor=SURFACE)
    for axis, grain_class in zip(axes, GRAIN_CLASSES):
        expanding, rolling = slope_frames[grain_class]
        axis.axhline(0, color=INK_MUTED, linewidth=0.8)
        axis.plot(expanding["date"], expanding["slope"], color=SERIES_COLORS[0], linewidth=1.5, label="expanding fit")
        axis.plot(rolling["date"], rolling["rolling_slope"], color=SERIES_COLORS[1], linewidth=1.2,
                  label="trailing 5 years")
        _style(axis, f"{grain_class.title()}: slope of position on log STU (log model)", "slope")
        axis.legend(frameon=False, fontsize=8, loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def plot_residual_acf(acf_by_class: dict, path: Path) -> None:
    """Autocorrelation of the out-of-sample residual at weekly lags, per class."""
    figure, axis = plt.subplots(figsize=(8, 3.8), facecolor=SURFACE)
    for (grain_class, acf), color in zip(acf_by_class.items(), SERIES_COLORS):
        axis.plot(acf.index, acf.to_numpy(), color=color, linewidth=2, marker="o", markersize=4, label=grain_class)
    axis.axhline(0, color=INK_MUTED, linewidth=0.8)
    _style(axis, "Out-of-sample residual autocorrelation (primary model)", "autocorrelation")
    axis.set_xlabel("lag (weeks)", color=INK_SECONDARY, fontsize=9)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


# ----------------------------------------------------------------------------- main
def run(output_dir: Path = OUTPUT_DIR, skip_truncation: bool = False) -> dict:
    """Execute the pre-registered study and write every table and plot. Returns a summary dict."""
    settings = load_settings()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    inputs = load_inputs()
    primary_stu, primary_model = settings["stocks"]["primary"], settings["models"]["primary"]
    monthly = stocks.monthly_stocks(inputs.balance_sheet)

    daily_by_class, fair_by_class, slope_frames, acf_by_class = {}, {}, {}, {}
    fit_tables, outside_rows, binned_tables, release_rows, baseline_rows = [], [], [], [], []
    for grain_class in GRAIN_CLASSES:
        daily = pipeline.build_daily(inputs, grain_class, settings)
        daily_by_class[grain_class] = daily
        for band_name in ("hybrid", "implied", "sagis"):
            outside_rows.append({"grain_class": grain_class, "band": band_name,
                                 **band.outside_band_summary(daily[f"position_{band_name}"])})
        band_start = daily.dropna(subset=["position"])["date"].min()
        first_date = models.first_out_of_sample_date(band_start, settings["models"]["min_seasons"])
        snapshots = models.weekly_snapshots(daily.dropna(subset=["position"]))

        table, predictions = fit_all_models(snapshots, settings, first_date)
        fit_tables += [table.assign(grain_class=grain_class),
                       sensitivity(snapshots, settings, band_start).assign(grain_class=grain_class)]
        primary = predictions[(primary_model, primary_stu)]
        fair_by_class[grain_class] = primary
        slope_frames[grain_class] = (primary, models.rolling_slope(
            snapshots, primary_stu, settings["models"]["rolling_slope_years"], settings["models"]["fourier_harmonics"]))
        acf_by_class[grain_class] = models.residual_autocorrelation(primary)
        for stu_column in settings["stocks"]["variants"]:
            binned = models.binned_means(snapshots, stu_column).assign(grain_class=grain_class, stu=stu_column)
            binned["spearman_full_sample"] = snapshots[[stu_column, "position"]].corr("spearman").iloc[0, 1]
            binned_tables.append(binned)

        surprises = stocks.release_surprises(monthly, inputs.weekly, grain_class, settings["stocks"]["days_per_month"])
        check = models.release_day_check(daily, surprises)
        release_rows.append({"grain_class": grain_class, **{k: v for k, v in check.items() if k != "events"}})
        baseline_rows.append(pipeline.baseline_model_a(inputs, grain_class, settings))

        plot_price_with_band(daily, grain_class, output_dir / "plots" / f"price_band_{grain_class}.png")
        plot_position_vs_stu(snapshots, primary_stu, grain_class,
                             output_dir / "plots" / f"position_vs_stu_{grain_class}.png")
    plot_positions(daily_by_class, settings, output_dir / "plots" / "band_position.png")
    plot_slopes(slope_frames, output_dir / "plots" / "slope_log_model.png")
    plot_residual_acf(acf_by_class, output_dir / "plots" / "residual_acf.png")

    tables = {"fit_results": pd.concat(fit_tables, ignore_index=True),
              "outside_band": pd.DataFrame(outside_rows), "binned_means": pd.concat(binned_tables, ignore_index=True),
              "release_day_check": pd.DataFrame(release_rows), "baseline_model_a": pd.DataFrame(baseline_rows),
              "residual_acf": pd.DataFrame(acf_by_class)}
    if not skip_truncation:
        tables["truncation_test"] = truncation_test(inputs, daily_by_class, fair_by_class, settings)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    run_hash = config_hash()
    log_path = output_dir / "variants_log.csv"
    log_line = pd.DataFrame([{"run_at": pd.Timestamp.now().isoformat(timespec="seconds"), "config_hash": run_hash,
                              "models_x_stu_per_class": len(settings["models"]["names"]) * len(settings["stocks"]["variants"]),
                              "sensitivity_runs_per_class": len(settings["models"]["min_seasons_sensitivity"])}])
    log_line.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
    summary = {"config_hash": run_hash, "variant_runs_logged": int(len(pd.read_csv(log_path))),
               "truncation_passed": (bool(tables["truncation_test"]["passed"].all())
                                     if "truncation_test" in tables else None)}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
