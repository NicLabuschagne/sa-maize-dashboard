"""Choose the export floor on economic criteria (Addendum 2). No forward returns are computed.

    python -m research.band_position.run_bands

Writes output/bands/: criteria table, sensitivity, truncation test, decision, plots and a variants
log line per run.
"""
from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research.band_position import revealed  # noqa: E402
from research.band_position.inputs import Inputs, inputs_as_of, load_inputs  # noqa: E402
from research.band_position.run import INK_MUTED, SERIES_COLORS, SURFACE, _style  # noqa: E402
from research.band_position.settings import config_hash, load_settings  # noqa: E402

OUTPUT_DIR = Path(os.getenv("BAND_POSITION_OUTPUT", Path(__file__).with_name("output"))) / "bands"
GRAIN_CLASSES = ("yellow", "white")
CANDIDATES = ("b0", "b1", "b2")
LABELS = {"b0": "B0 expanding", "b1": "B1 season rule", "b2": "B2 Kalman"}
EVALUATION = ("2015-05-01", "2026-09-30")
MAX_SHARE_BELOW = 0.10


def criteria_table(frames: dict, inputs: Inputs) -> pd.DataFrame:
    """C1–C3 for every candidate and class over the evaluation window."""
    rows = [revealed.band_criteria(frames[c], inputs.weekly, c, name, *EVALUATION)
            for c in GRAIN_CLASSES for name in CANDIDATES]
    return pd.DataFrame(rows)


def sensitivity_table(inputs: Inputs, settings: dict) -> pd.DataFrame:
    """B2 with the season jump halved and doubled. Reported, never used to choose."""
    rows = []
    for jump in (0.075, 0.30):
        params = replace(revealed.KalmanSettings(), season_jump_sd=jump)
        for grain_class in GRAIN_CLASSES:
            frame = revealed.build_candidates(inputs, grain_class, settings, params)
            rows.append({"season_jump_sd": jump,
                         **revealed.band_criteria(frame, inputs.weekly, grain_class, "b2", *EVALUATION)})
    return pd.DataFrame(rows)


def truncation_test(inputs: Inputs, frames: dict, settings: dict) -> pd.DataFrame:
    """Rebuild every candidate from inputs cut at t; floors at t must match the full build exactly."""
    rng = np.random.default_rng(settings["truncation"]["seed"])
    candidates = frames["yellow"].loc[frames["yellow"]["date"] >= EVALUATION[0], "date"].to_numpy()
    dates = np.sort(rng.choice(candidates, size=settings["truncation"]["n_dates"], replace=False))
    rows = []
    for date in pd.DatetimeIndex(dates):
        cut = inputs_as_of(inputs, date)
        for grain_class in GRAIN_CLASSES:
            rebuilt = revealed.build_candidates(cut, grain_class, settings).set_index("date")
            full = frames[grain_class].set_index("date")
            record = {"date": date, "grain_class": grain_class}
            for name in CANDIDATES:
                a, b = full.at[date, f"floor_{name}"], rebuilt.at[date, f"floor_{name}"]
                record[f"diff_{name}"] = 0.0 if (np.isnan(a) and np.isnan(b)) else abs(a - b)
            rows.append(record)
    table = pd.DataFrame(rows)
    table["max_diff"] = table[[f"diff_{n}" for n in CANDIDATES]].max(axis=1)
    table["passed"] = table["max_diff"] <= settings["truncation"]["tolerance"]
    return table


def decide(criteria: pd.DataFrame, truncation: pd.DataFrame) -> dict:
    """Apply the pre-registered rule for yellow: pass C4, C1 share <= 10%, then lowest C2."""
    yellow = criteria[criteria.grain_class == "yellow"].set_index("band")
    eligible = []
    for name in CANDIDATES:
        passed_c4 = bool((truncation[f"diff_{name}"] <= 1e-9).all())
        passed_c1 = bool(yellow.at[name, "share_below_floor"] <= MAX_SHARE_BELOW)
        if passed_c4 and passed_c1:
            eligible.append(name)
    chosen = min(eligible, key=lambda n: yellow.at[n, "median_abs_distance_usd"]) if eligible else None
    return {"eligible": eligible, "chosen": chosen}


# ----------------------------------------------------------------------------- plots
def plot_floors(frame: pd.DataFrame, weekly: pd.DataFrame, grain_class: str, start: str, end: str, path: Path) -> None:
    """SAFEX against the three candidate floors, with strong export weeks shaded."""
    part = frame[(frame["date"] >= start) & (frame["date"] <= end)]
    figure, axis = plt.subplots(figsize=(11, 4.5), facecolor=SURFACE)
    for week_end in revealed.strong_export_weeks(weekly, grain_class, 30.0):
        if pd.Timestamp(start) <= week_end <= pd.Timestamp(end):
            axis.axvspan(week_end - pd.Timedelta(days=6), week_end, color="#f0efec", linewidth=0, zorder=0)
    for name, color in zip(CANDIDATES, SERIES_COLORS[:3]):
        axis.plot(part["date"], part[f"floor_{name}"], color=color, linewidth=1.3, label=LABELS[name])
    axis.plot(part["date"], part["safex"], color=SERIES_COLORS[3], linewidth=1.3, label="SAFEX (90-day)")
    _style(axis, f"{grain_class.title()}: candidate export floors (shaded = weeks exporting ≥ 30 kt)", "R/t")
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def plot_kalman_state(frame: pd.DataFrame, grain_class: str, path: Path) -> None:
    """B2's competitiveness state with a 2-sd band, and the daily observation coloured by export pace."""
    part = frame[frame["date"] >= "2013-01-01"]
    observation = np.log((part["safex"] + part["export_deductions"]) / part["world"])
    weight = revealed.pace_weight(part["export_pace"].to_numpy(), 5.0, 30.0)
    figure, axis = plt.subplots(figsize=(11, 4.5), facecolor=SURFACE)
    axis.scatter(part["date"], observation, c=weight, cmap="Greys", vmin=-0.3, vmax=1, s=3, label="observation (darker = more exports)")
    axis.fill_between(part["date"], part["state_b2"] - 2 * part["state_sd_b2"], part["state_b2"] + 2 * part["state_sd_b2"],
                      color=SERIES_COLORS[0], alpha=0.2, linewidth=0)
    axis.plot(part["date"], part["state_b2"], color=SERIES_COLORS[0], linewidth=1.5, label="floor state ± 2 sd")
    _style(axis, f"{grain_class.title()}: Kalman export-competitiveness state (log)", "log((SAFEX + costs) / world)")
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def run(output_dir: Path = OUTPUT_DIR) -> dict:
    """Build the three candidates, score them, apply the decision rule, write everything out."""
    settings = load_settings()
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs()
    frames = {c: revealed.build_candidates(inputs, c, settings) for c in GRAIN_CLASSES}
    criteria = criteria_table(frames, inputs)
    sensitivity = sensitivity_table(inputs, settings)
    truncation = truncation_test(inputs, frames, settings)
    decision = decide(criteria, truncation)

    criteria.to_csv(output_dir / "criteria.csv", index=False)
    sensitivity.to_csv(output_dir / "sensitivity.csv", index=False)
    truncation.to_csv(output_dir / "truncation_test.csv", index=False)
    pd.concat(frames.values(), ignore_index=True).to_csv(output_dir / "candidate_floors_daily.csv", index=False)
    for grain_class in GRAIN_CLASSES:
        plot_floors(frames[grain_class], inputs.weekly, grain_class, "2015-05-01", "2026-09-30",
                    output_dir / f"floors_{grain_class}.png")
        plot_kalman_state(frames[grain_class], grain_class, output_dir / f"kalman_state_{grain_class}.png")
    plot_floors(frames["yellow"], inputs.weekly, "yellow", "2020-05-01", "2023-10-31", output_dir / "floors_yellow_2020_2023.png")

    log_path = output_dir / "variants_log.csv"
    pd.DataFrame([{"run_at": pd.Timestamp.now().isoformat(timespec="seconds"), "config_hash": config_hash(),
                   "candidates": len(CANDIDATES), "sensitivity_runs": 2}]).to_csv(
        log_path, mode="a", header=not log_path.exists(), index=False)
    summary = {**decision, "truncation_passed": bool(truncation["passed"].all()),
               "runs_logged": int(len(pd.read_csv(log_path)))}
    (output_dir / "decision.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
