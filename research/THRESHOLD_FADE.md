# Threshold fade of the release signal (WMAZ, YMAZ)

Scripts: `threshold_fade.py` (rules in its docstring, fixed before the run) and
`threshold_fade_diagnostics.py`. Signal: Model A out-of-sample z, as shown on Home.

The last five years is where the idea came from (the Home replay), so results there are in-sample to
that inspection. 2013 – Aug 2021 uses identical rules and is the check. Configurations tried: 2
thresholds × 2 holding rules = 4.

## Hold until z is back inside ±0.5 (6-release time stop)

| Period | T | Trades | Hit | Avg net / trade | Sharpe | Max DD |
|---|---|---|---|---|---|---|
| Last 5y | 1.0 | 11 | 55% | +4.0% | 0.37 | −21.6% |
| Last 5y | 1.5 | 4 | 25% | +0.3% | 0.02 | −20.5% |
| 2013–2021 | 1.0 | 14 | 29% | −3.8% | −0.26 | −52.1% |
| 2013–2021 | 1.5 | 8 | 38% | −2.0% | −0.10 | −27.5% |

Both books, equal notional, costs incl. a round trip per roll.

## Same entries, fixed 10-trading-day hold

| Period | T | WMAZ Sharpe (n, hit) | YMAZ Sharpe (n, hit) |
|---|---|---|---|
| Last 5y | 1.0 | +0.96 (14, 50%) | +0.79 (19, 63%) |
| Last 5y | 1.5 | +0.51 (4, 50%) | −0.22 (5, 40%) |
| 2013–2021 | 1.0 | +0.15 (25, 56%) | +0.30 (29, 48%) |
| 2013–2021 | 1.5 | −0.05 (14, 50%) | +0.35 (10, 40%) |

## Why holding to fair value fails

- **Z comes back partly because fair value moves, not price.** On the trades that exited on
  reversion, 43% of the closing of the gap came from the fair value moving toward the price. Example:
  YMAZ long at z −1.41 (May 2025). Z went to +0.36 because fair value fell 29% while the price rose
  5%; the trade made −0.4%.
- **The biggest dislocations are regime moves the stock-based fair value lags.** Examples: the
  2015/16 drought (short WMAZ at z 1.3, z went to 2.7, −38%) and the 2022 Ukraine rally. These are the
  worst in-trade drawdowns: −45%, −21%.
- **The edge lives in the first 5–10 days after a release**, as the IC work showed. The fixed short
  hold is positive in both periods at T = 1.0. It is modest in the check period and ≈1 Sharpe per
  product in the last five years.

Stop: with a 10-day hold the horizon is the stop. For a hold-to-fair-value version, the in-trade
drawdowns above show that no price stop is small enough to be comfortable yet wide enough to survive
the normal path.
