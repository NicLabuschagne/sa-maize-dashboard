"""Run the Addendum 4 edge trading test on the B2 band.

    python -m research.band_position.run_edge

Writes output/edge/: IC tables, zone mean returns, the decision, backtest statistics, equity plots
and a variants log line.
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

from app.data.backtest import CostModel  # noqa: E402
from research.band_position import edge_test  # noqa: E402
from research.band_position.inputs import load_inputs  # noqa: E402
from research.band_position.run import INK_MUTED, SERIES_COLORS, SURFACE, _style  # noqa: E402
from research.band_position.settings import config_hash, load_settings  # noqa: E402

OUTPUT_DIR = Path(os.getenv("BAND_POSITION_OUTPUT", Path(__file__).with_name("output"))) / "edge"
FIRST_DATE = pd.Timestamp("2015-05-01")
PERIODS = {"full": ("2015-05-01", "2026-09-30"), "development": ("2015-05-01", "2022-12-31"),
           "holdout": ("2023-01-01", "2026-09-30")}
GRAIN_CLASSES = ("yellow", "white")


def plot_equity(daily_by_rule: dict, grain_class: str, path: Path) -> None:
    """Cumulative net log P&L of each rule, with the holdout start marked."""
    figure, axis = plt.subplots(figsize=(11, 4.5), facecolor=SURFACE)
    colors = SERIES_COLORS + ["#e87ba4"]
    for (rule, daily), color in zip(daily_by_rule.items(), colors):
        part = daily[daily["date"] >= PERIODS["full"][0]]
        axis.plot(part["date"], part["net"].cumsum(), color=color, linewidth=1.4, label=rule)
    axis.axvline(pd.Timestamp(PERIODS["holdout"][0]), color=INK_MUTED, linewidth=1, linestyle="--")
    axis.axhline(0, color=INK_MUTED, linewidth=0.8)
    _style(axis, f"{grain_class.title()}: cumulative net arb P&L by rule (log; dashed = holdout start)",
           "cumulative log return")
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def run(output_dir: Path = OUTPUT_DIR) -> dict:
    """Build the edge frames, run the IC tests and the backtests, write everything out."""
    settings = load_settings()
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs()
    costs = CostModel()
    ic_rows, zone_rows, backtest_rows, frames = [], [], [], {}
    decision = pd.DataFrame()
    for grain_class in GRAIN_CLASSES:
        frame = edge_test.build_edge_frame(inputs, grain_class, settings, FIRST_DATE)
        frames[grain_class] = frame
        for period, (start, end) in PERIODS.items():
            for zone in ("export_zone", "import_zone"):
                ic_rows.append(edge_test.zone_ic(frame, zone, start, end).assign(grain_class=grain_class, period=period))
                zone_rows.append({"grain_class": grain_class, "period": period,
                                  **edge_test.zone_mean_return(frame, zone, start, end)})
        if grain_class == "yellow":
            dev = edge_test.zone_ic(frame, "export_zone", *PERIODS["development"])
            holdout = edge_test.zone_ic(frame, "export_zone", *PERIODS["holdout"])
            decision = edge_test.decide(dev, holdout)
        daily_by_rule = {}
        for rule, target in edge_test.rule_positions(frame).items():
            daily = edge_test.backtest_rule(frame, target, costs)
            daily_by_rule[rule] = daily
            for period, (start, end) in PERIODS.items():
                backtest_rows.append({"grain_class": grain_class, "rule": rule, "period": period,
                                      **edge_test.summarise_rule(daily, start, end)})
        plot_equity(daily_by_rule, grain_class, output_dir / f"equity_{grain_class}.png")

    backtests = pd.DataFrame(backtest_rows)
    backtests = pd.concat([edge_test.add_deflated_sharpe(backtests[backtests.grain_class == c]) for c in GRAIN_CLASSES],
                          ignore_index=True)
    tables = {"ic": pd.concat(ic_rows, ignore_index=True), "zone_mean_returns": pd.DataFrame(zone_rows),
              "decision": decision, "backtests": backtests,
              "edge_frame_daily": pd.concat(frames.values(), ignore_index=True)}
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    log_path = output_dir / "variants_log.csv"
    new_row = pd.DataFrame([{"run_at": pd.Timestamp.now().isoformat(timespec="seconds"), "config_hash": config_hash(),
                             "ic_tests": 3, "backtest_rules": 5, "stage": "addendum 4"}])
    log = pd.concat([pd.read_csv(log_path), new_row], ignore_index=True) if log_path.exists() else new_row
    log.to_csv(log_path, index=False)
    summary = {"passing_signals": decision.loc[decision["passes"], "signal"].tolist(), "runs_logged": int(len(log))}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
