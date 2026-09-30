"""Run the Addendum 5 band-extreme test.

    python -m research.band_position.run_extremes

Writes output/extremes/: events, group statistics, zone IC, trade view, decision, plots, variants log.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from app.data.backtest import CostModel  # noqa: E402
from config import DB_PATH  # noqa: E402
from research.band_position import extremes  # noqa: E402
from research.band_position.run import INK, INK_MUTED, SERIES_COLORS, SURFACE, _style  # noqa: E402
from research.band_position.settings import config_hash  # noqa: E402

OUTPUT_DIR = Path(os.getenv("BAND_POSITION_OUTPUT", Path(__file__).with_name("output"))) / "extremes"
CLASSES = {"yellow": "YMAZ", "white": "WMAZ"}
PERIODS = {"full": ("2015-05-01", "2026-09-30"), "development": ("2015-05-01", "2022-12-31"),
           "holdout": ("2023-01-01", "2026-09-30")}


def load_tables() -> dict:
    """Warehouse tables used by the dashboard band model, with timestamps at nanosecond precision."""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        tables = {t: con.execute(f"SELECT * FROM {t}").df()
                  for t in ("prices", "macro_snap", "sagis_parity", "balance_sheet", "sagis_weekly")}
    finally:
        con.close()
    for frame in tables.values():
        for column in frame.columns:
            if pd.api.types.is_datetime64_any_dtype(frame[column]):
                frame[column] = frame[column].astype("datetime64[ns]")
    return tables


def plot_events(daily: pd.DataFrame, events: pd.DataFrame, grain_class: str, path: Path) -> None:
    """Position and fair position with every event marked: filled = confirmed by the gap, hollow = not."""
    part = daily[daily["date"] >= PERIODS["full"][0]]
    figure, axis = plt.subplots(figsize=(11, 4.6), facecolor=SURFACE)
    for edge in (0, 1):
        axis.axhline(edge, color=INK_MUTED, linewidth=0.8)
    axis.plot(part["date"], part["fair_position"], color=SERIES_COLORS[2], linewidth=1.6, label="fair position")
    axis.plot(part["date"], part["position"], color=SERIES_COLORS[1], linewidth=1.0, label="position")
    for side, marker in (("long", "^"), ("short", "v")):
        for confirmed, face in ((True, INK), (False, "none")):
            e = events[(events["side"] == side) & (events["confirmed"] == confirmed)]
            label = f"{side} event, {'confirmed' if confirmed else 'not confirmed'} by gap"
            axis.scatter(e["date"], e["position"], marker=marker, s=60, facecolors=face, edgecolors=INK,
                         linewidths=1.2, zorder=5, label=label)
    axis.axvline(pd.Timestamp(PERIODS["holdout"][0]), color=INK_MUTED, linestyle="--", linewidth=1)
    _style(axis, f"{grain_class.title()}: band extremes (▲ long at < 0.1, ▼ short at > 0.9; dashed = holdout)",
           "position (0 = export, 1 = import)")
    axis.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def run(output_dir: Path = OUTPUT_DIR) -> dict:
    """Build events for both classes, score them, apply the decision rule and write everything out."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = load_tables()
    costs = CostModel()
    all_events, group_rows, ic_rows, trades = [], [], [], []
    decision = {}
    for grain_class, symbol in CLASSES.items():
        daily = extremes.build_daily(tables, grain_class, symbol)
        events = pd.concat([extremes.find_events(daily, "long"), extremes.find_events(daily, "short")],
                           ignore_index=True).sort_values("date")
        events = extremes.event_outcomes(daily, events).assign(grain_class=grain_class)
        all_events.append(events)
        group_rows.append(extremes.group_table(events, PERIODS).assign(grain_class=grain_class))
        for period, (start, end) in PERIODS.items():
            ic_rows.append({"grain_class": grain_class, "period": period, **extremes.zone_gap_ic(daily, start, end)})
        trades.append(extremes.trade_view(daily, events, costs))
        if grain_class == "yellow":
            decision = extremes.decide(group_rows[-1])
        plot_events(daily, events, grain_class, output_dir / f"events_{grain_class}.png")

    trade_table = pd.concat(trades, ignore_index=True)
    trade_summary = (trade_table.groupby(["grain_class", "side", "confirmed"])
                     .agg(trades=("net", "size"), mean_net=("net", "mean"), hit_rate=("net", lambda s: (s > 0).mean()),
                          total_net=("net", "sum"))
                     .reset_index())
    outputs = {"events": pd.concat(all_events, ignore_index=True), "groups": pd.concat(group_rows, ignore_index=True),
               "zone_gap_ic": pd.DataFrame(ic_rows), "trades": trade_table, "trade_summary": trade_summary}
    for name, table in outputs.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    log_path = output_dir / "variants_log.csv"
    row = pd.DataFrame([{"run_at": pd.Timestamp.now().isoformat(timespec="seconds"), "config_hash": config_hash(),
                         "stage": "addendum 5", "thresholds": "0.1/0.9 entry, 0.2/0.8 re-arm, gap 0.2, 40d"}])
    log = pd.concat([pd.read_csv(log_path), row], ignore_index=True) if log_path.exists() else row
    log.to_csv(log_path, index=False)
    summary = {**decision, "runs_logged": int(len(log))}
    (output_dir / "decision.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
