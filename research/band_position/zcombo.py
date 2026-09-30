"""Band extreme AND a short-window z-score of the fundamental gap (Addendum 6).

    z_w      (gap - rolling mean) / rolling sd over w days, w = 63 or 126
    long     position < 0.1 and z >= 1        short   position > 0.9 and z <= -1

Compared with the same extremes when z does not agree. Outcomes reuse Addendum 5's event machinery.

    python -m research.band_position.zcombo
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
from scipy import stats  # noqa: E402

from app.data.backtest import CostModel  # noqa: E402
from research.band_position import extremes  # noqa: E402
from research.band_position.edge_test import holm  # noqa: E402
from research.band_position.run import INK, INK_MUTED, SERIES_COLORS, SURFACE, _style  # noqa: E402
from research.band_position.run_extremes import CLASSES, PERIODS, load_tables  # noqa: E402
from research.band_position.settings import config_hash  # noqa: E402

OUTPUT_DIR = Path(os.getenv("BAND_POSITION_OUTPUT", Path(__file__).with_name("output"))) / "zcombo"
WINDOWS = {"3m": 63, "6m": 126}
Z_THRESHOLD = 1.0
REARM_DAYS = 5
GROUPS = ("combined (extreme AND z agrees)", "extreme, z disagrees", "extreme alone")


def gap_z(gap: pd.Series, window: int) -> pd.Series:
    """Gap relative to its own recent mean, in units of its recent standard deviation."""
    min_periods = int(window * 0.8)
    return (gap - gap.rolling(window, min_periods=min_periods).mean()) / gap.rolling(window, min_periods=min_periods).std()


def conditions(daily: pd.DataFrame, z: pd.Series) -> dict[tuple[str, str], pd.Series]:
    """Boolean day masks for each (side, group). Days without a fair position are never in a group."""
    known = daily["fair_position"].notna() & z.notna()
    low, high = (daily["position"] < extremes.LONG_ENTRY) & known, (daily["position"] > extremes.SHORT_ENTRY) & known
    agree_long, agree_short = z >= Z_THRESHOLD, z <= -Z_THRESHOLD
    return {("long", GROUPS[0]): low & agree_long, ("long", GROUPS[1]): low & ~agree_long, ("long", GROUPS[2]): low,
            ("short", GROUPS[0]): high & agree_short, ("short", GROUPS[1]): high & ~agree_short, ("short", GROUPS[2]): high}


def entries(mask: pd.Series) -> np.ndarray:
    """Row numbers where the condition holds after at least REARM_DAYS trading days without it."""
    values = mask.fillna(False).to_numpy(dtype=bool)
    rows = []
    for i in np.flatnonzero(values):
        if not values[max(0, i - REARM_DAYS):i].any():
            rows.append(i)
    return np.array(rows, dtype=int)


def build_events(daily: pd.DataFrame, z: pd.Series) -> pd.DataFrame:
    """Entries for every (side, group) with forward outcomes from Addendum 5's `event_outcomes`."""
    frames = []
    for (side, group), mask in conditions(daily, z).items():
        rows = entries(mask)
        if len(rows) == 0:
            continue
        events = pd.DataFrame({"row": rows, "date": daily["date"].to_numpy()[rows], "side": side, "group": group,
                               "position": daily["position"].to_numpy()[rows], "gap": daily["gap"].to_numpy()[rows],
                               "z": z.to_numpy()[rows]})
        frames.append(extremes.event_outcomes(daily, events))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarise(events: pd.DataFrame) -> pd.DataFrame:
    """Mean, median, hit rate and event t of the forward arb return by side, group, period and horizon."""
    rows = []
    for (side, group), part in events.groupby(["side", "group"]):
        for period, (start, end) in PERIODS.items():
            p = part[(part["date"] >= start) & (part["date"] <= end)]
            for h in extremes.HORIZONS:
                r = p[f"fwd_arb_{h}"].dropna()
                rows.append({"side": side, "group": group, "period": period, "horizon": h, "entries": len(r),
                             "mean_arb": r.mean() if len(r) else np.nan, "median_arb": r.median() if len(r) else np.nan,
                             "hit_rate": (r > 0).mean() if len(r) else np.nan,
                             "t_entries": float(stats.ttest_1samp(r, 0).statistic) if len(r) >= 3 else np.nan,
                             "mean_safex": p[f"fwd_safex_{h}"].dropna().mean() if len(p) else np.nan})
    return pd.DataFrame(rows)


def decide(table: pd.DataFrame, events: pd.DataFrame) -> dict:
    """Addendum 6 rule per window (yellow, long, 40 days) plus holdout Welch t, combined vs disagrees."""
    t = table[(table.side == "long") & (table.horizon == extremes.PRIMARY_HORIZON)].set_index(["period", "group"])

    def mean(period: str, group: str) -> float:
        return float(t.at[(period, group), "mean_arb"]) if (period, group) in t.index else np.nan

    holdout = events[(events.side == "long") & (events.date >= PERIODS["holdout"][0])]
    a = holdout.loc[holdout.group == GROUPS[0], f"fwd_arb_{extremes.PRIMARY_HORIZON}"].dropna()
    b = holdout.loc[holdout.group == GROUPS[1], f"fwd_arb_{extremes.PRIMARY_HORIZON}"].dropna()
    welch = stats.ttest_ind(a, b, equal_var=False) if len(a) >= 2 and len(b) >= 2 else None
    checks = {"development: combined > z disagrees": mean("development", GROUPS[0]) > mean("development", GROUPS[1]),
              "holdout: combined > z disagrees": mean("holdout", GROUPS[0]) > mean("holdout", GROUPS[1]),
              "holdout: combined mean > 0": mean("holdout", GROUPS[0]) > 0}
    return {"checks": {k: bool(v) for k, v in checks.items()}, "passes": bool(all(checks.values())),
            "holdout_welch_t": float(welch.statistic) if welch else np.nan,
            "holdout_welch_p": float(welch.pvalue) if welch else np.nan,
            "holdout_entries_combined": int(len(a)), "holdout_entries_disagree": int(len(b))}


def plot(daily: pd.DataFrame, z: pd.Series, events: pd.DataFrame, grain_class: str, label: str, path: Path) -> None:
    """Position with combined entries marked, and the z-score below with the ±1 thresholds."""
    part = daily["date"] >= PERIODS["full"][0]
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, facecolor=SURFACE,
                                         gridspec_kw={"height_ratios": [3, 2]})
    for edge in (0, 1):
        top.axhline(edge, color=INK_MUTED, linewidth=0.8)
    top.plot(daily.loc[part, "date"], daily.loc[part, "fair_position"], color=SERIES_COLORS[2], linewidth=1.5,
             label="fair position")
    top.plot(daily.loc[part, "date"], daily.loc[part, "position"], color=SERIES_COLORS[1], linewidth=1.0,
             label="position")
    for side, marker in (("long", "^"), ("short", "v")):
        e = events[(events.side == side) & (events.group == GROUPS[0])]
        top.scatter(e["date"], e["position"], marker=marker, s=55, color=INK, zorder=5,
                    label=f"{side}: extreme AND z agrees")
    _style(top, f"{grain_class.title()}: extreme AND {label} gap z-score (dashed = holdout)", "position")
    top.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)
    bottom.plot(daily.loc[part, "date"], z[part], color=SERIES_COLORS[0], linewidth=0.9)
    for level in (-Z_THRESHOLD, Z_THRESHOLD):
        bottom.axhline(level, color=INK_MUTED, linewidth=0.8, linestyle=":")
    _style(bottom, f"Gap z-score ({label})", "z")
    for axis in (top, bottom):
        axis.axvline(pd.Timestamp(PERIODS["holdout"][0]), color=INK_MUTED, linestyle="--", linewidth=1)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def run(output_dir: Path = OUTPUT_DIR) -> dict:
    """Both windows, both classes: events, summaries, trade view, decision per window, plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tables, costs = load_tables(), CostModel()
    all_events, summaries, trades, decisions = [], [], [], {}
    for grain_class, symbol in CLASSES.items():
        daily = extremes.build_daily(tables, grain_class, symbol)
        for label, window in WINDOWS.items():
            z = gap_z(daily["gap"], window)
            events = build_events(daily, z).assign(grain_class=grain_class, window=label)
            all_events.append(events)
            table = summarise(events).assign(grain_class=grain_class, window=label)
            summaries.append(table)
            combined = events[events.group == GROUPS[0]]
            trades.append(extremes.trade_view(daily, combined, costs).assign(grain_class=grain_class, window=label))
            if grain_class == "yellow":
                decisions[label] = decide(table, events)
            plot(daily, z, events, grain_class, label, output_dir / f"zcombo_{grain_class}_{label}.png")
    ps = [decisions[w]["holdout_welch_p"] for w in WINDOWS]
    finite = [p if np.isfinite(p) else 1.0 for p in ps]
    for w, p_holm in zip(WINDOWS, holm(finite)):
        decisions[w]["holdout_welch_p_holm"] = p_holm
    trade_table = pd.concat(trades, ignore_index=True)
    trade_summary = (trade_table.groupby(["grain_class", "window", "side"])
                     .agg(trades=("net", "size"), mean_net=("net", "mean"), hit_rate=("net", lambda s: (s > 0).mean()))
                     .reset_index())
    for name, table in {"events": pd.concat(all_events, ignore_index=True), "summary": pd.concat(summaries, ignore_index=True),
                        "trades": trade_table, "trade_summary": trade_summary}.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    log_path = output_dir / "variants_log.csv"
    row = pd.DataFrame([{"run_at": pd.Timestamp.now().isoformat(timespec="seconds"), "config_hash": config_hash(),
                         "stage": "addendum 6", "variants": "z windows 63, 126; threshold 1; 40d"}])
    log = pd.concat([pd.read_csv(log_path), row], ignore_index=True) if log_path.exists() else row
    log.to_csv(log_path, index=False)
    summary = {"decisions": decisions, "runs_logged": int(len(log))}
    (output_dir / "decision.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
