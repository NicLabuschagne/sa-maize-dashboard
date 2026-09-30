# Pre-registration: does the balance sheet set SAFEX's position inside the parity band?

Written before any model in `research/band_position/` was fitted. The only checks run beforehand were
descriptive: parity history and gaps, how often SAFEX sits outside the SAGIS band, what drives the
SAGIS band width, and weekly-data coverage (see "Facts established before this document").

## Hypothesis

World prices, freight and USD/ZAR set the band; the domestic balance sheet decides where SAFEX sits
inside it. Expected shape: position falls as stocks-to-use rises, steep mid-band and flat at the edges
(an S-curve). The balance sheet should explain more mid-band than at the edges, and more for white
than for yellow.

## Facts established before this document

- SAGIS Randfontein-basis parity: weekly, 2007-01 to 2026-09, gaps only at year end (4–5 weeks).
  Overlap with SAFEX from 2009-04: ~17 marketing seasons.
- SAFEX front month sits outside the SAGIS band on 37% of days (white) and 25% (yellow). Yellow was
  below SAGIS export parity on 99% of 2022 days. The SAGIS export floor is too high to be a floor
  (Gulf FOB, whereas SA competes with Brazil/Argentina). So the SAGIS band is a comparison, not the
  primary band.
- SAGIS band width (import − export) is 98% explained by ocean freight × USD/ZAR plus rail Durban–
  Randfontein (weekly, 2009+, n = 867). Width is a cost, not something to forecast.

## Band (primary: hybrid)

    world_t        = CBOT corn (USD/bu, 10:00 UTC bar) × 39.3683 × USD/ZAR (10:00 UTC bar)     R/t
    basis_t        = log(SAFEX front_t / world_t)                          own class (white, yellow)
    export_edge_t  = world_t × exp(Q_5%(basis over days < t))
    excess_t       = (SAFEX_t − export_edge_t) / cost_width_t
    cost_width_t   = SAGIS import_randfontein − export_randfontein, latest with available_date ≤ t
    import_edge_t  = export_edge_t + Q_95%(excess over days < t) × cost_width_t
    position_t     = (SAFEX_t − export_edge_t) / (import_edge_t − export_edge_t)      not clipped

Quantiles use prior days only, with at least 2 years (500 trading days) of history. Two comparison
bands are reported beside it, not used for selection: the **SAGIS band as published** and the **pure
implied band** (`world × exp(Q_5%)`, `world × exp(Q_95%)`, width proportional to world price).

Outside-band statistics: share of days below 0 and above 1, number of spells, median and longest
spell in trading days.

## Stocks-to-use (daily, point in time)

| Name | Definition | Role |
|---|---|---|
| `stu_total` | closing stock ÷ trailing-12m (utilisation + exports) | baseline (as used today) |
| `stu_domestic` | closing stock ÷ trailing-12m utilisation | removes export feedback |
| `stu_domestic_nowcast` | weekly-nowcast stock ÷ trailing-12m utilisation | **primary** |

Monthly figures enter on `vintage_date`. The nowcast is the W1 rule from `PREREGISTRATION_WEEKLY.md`
applied daily: closing stock + weekly deliveries + imports − exports for weeks ending after the
reported month and published by the day (export publication gates the week set), minus trailing
utilisation pro-rated by weeks. A projected (CEC-based) carry-out is **not** in this round: its data is
not in the repo yet. It gets its own addendum.

## Models (white and yellow separately)

Fitted on **weekly snapshots** (last trading day of each week), then applied daily later.
`season` = 2 Fourier harmonics on day of the marketing year (May 1 = 0).

| Model | Form |
|---|---|
| L linear | position = a + season + b · STU |
| G log | position = a + season + b · log STU |
| S logistic | position = 1 / (1 + exp(−(a + season + b · log STU))), non-linear least squares |
| I isotonic | position = non-increasing step function of STU, no season |
| N naive | expanding mean of position (benchmark) |

Each model at week t is fitted on weeks before t. First out-of-sample week: the first 1 May with at
least **5 complete marketing years** of band history before it. Sensitivity at 3 and 4 years is
reported, never used to choose.

Primary test: **model G on `stu_domestic_nowcast`**, per class. The other 3 models × 3 STU versions
are reported in full.

## What is reported

1. **Baseline (step A).** Existing Model A (`fairvalue.py`, log real price on log cover + season):
   in-sample R², out-of-sample R² vs expanding mean, and the nominal-log-level in-sample R², to show
   how much of a levels fit is shared trend.
2. **Fit.** Out-of-sample R² vs N for each model × STU version × class. Binned mean position by STU
   quintile. Spearman(STU, position).
3. **Regimes.** Mid-band = previous week's position in [0.2, 0.8]; edges otherwise. Out-of-sample R²
   per regime (each against N on the same weeks).
4. **Release-day check.** On each SAGIS monthly release: surprise = (published closing stock − weekly
   nowcast of the same month-end made the day before) ÷ trailing utilisation. Outcome: change in
   position from the last close before the release to the first close after it. Spearman, with a
   permutation p-value. Expected sign negative.
5. **Leakage.** Truncation test: 50 weekly dates (seed 0); rebuild every input cut at t; band, position,
   STU and fair position at t must match the full build to 1e-9.
6. **Plots.** Price with bands; position over time; position vs STU by season stage; expanding and
   5-year rolling slope of model G; autocorrelation of the out-of-sample residual.

## Decision rule for moving to the trading stage

The fair position is worth trading against only if, for a class, model G (or S or I) on the primary
STU has **out-of-sample R² > 0 against N** and a negative STU slope in both halves of the
out-of-sample period. Tradeability (residual vs forward 5/10/20/40-day change in position, Newey-West
lags ≥ h − 1, effective N) is a separate, later pre-registration.

## Variant log

Every configuration run is appended to `research/band_position/output/variants_log.csv` with its
config hash. The count is reported with the results. Thresholds (quantiles, 0.2/0.8, 5 seasons) are
not tuned.

## Known limitations, stated in advance

- The band edges are implied from SAFEX's own history. Early in the sample, the import edge has not
  yet seen a drought (2015/16), so positions above 1 are expected then.
- SAFEX front month is a future; world parity is spot-like. Carry is in the basis.
- White uses its own basis history but the SAGIS cost width is for yellow imports.
- Weekly snapshots are autocorrelated; ~17 seasons is the honest sample size.
