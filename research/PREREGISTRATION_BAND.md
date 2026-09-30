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

## Addendum 1 (after round 1, before round 2)

Written after the round-1 results (committed `d318fb6`), so everything below is informed by them
and is labelled as such.

**Changes**
- **Price series: constant maturity 90 days** (post-hoc; motivated by jumps at the March→May
  old-/new-crop roll seen in round 1). On each day, log-linear interpolation in days to expiry
  between the two main-month contracts bracketing 90 days. Contracts within 7 days of expiry are
  excluded, as in the dashboard roll rule. If no contract is at or below 90 days, the nearest
  contract above is used. 90 days was chosen because it is the longest gap between SAFEX main months
  (Sep→Dec, Dec→Mar), so two contracts nearly always bracket it. Not tuned; no other tenor is run.
- **Primary model: linear** (post-hoc; it had the best round-1 out-of-sample fit for both classes).
  Its round-2 results are therefore not an independent confirmation. Independent evidence comes
  only from weeks after 30 Sep 2026.
- Yellow is the primary class; white is reported alongside.

**Added reporting**
- Out-of-sample R² for calendar years and for the windows 2020-01 → 2026-09 and 2023-01 → 2026-09.
  Every fit still starts in May 2017 and is expanding; these windows only select which predictions
  are scored.
- Out-of-sample R² and within-stage Spearman by season stage.

**IC of the fair value (first tradeability look)**
- Daily fair position: each day uses the primary model fitted on weekly snapshots before that
  day's week. Daily fair value in R/t = export edge + fair position × band width.
- Signal = fair position − position (positive = cheap vs fair).
- Benchmark signal = expanding mean of position − position (plain mean reversion, no balance sheet).
- Outcomes, entered at the **next** close (d+1) and measured to d+1+h, for h = 5, 10, 20, 40 trading
  days: (a) change in band position; (b) roll-adjusted log return of the SAFEX front month (the
  tradeable outright).
- Statistics: Spearman IC on daily observations. Newey-West t-stat on the slope of outcome on the
  standardised signal, lags = h − 1. Effective N = observations ÷ h. Windows: full OOS, 2020–2026,
  2023–2026.
- This is a first look at tradeability, not a strategy. No thresholds, no costs.

## Addendum 2 (before any band in this section is built): choosing the band without returns

Round 2 showed the pooled fair-value gap does not predict. The desk view (edges only, export
side long, no trustworthy paper calculation) motivates a better **export floor**. The band is chosen
here on economic criteria only. No forward return is computed until the band is fixed.

**Out-of-sample start for everything that follows: 1 May 2015** (the pre-registered 3-season window),
so the 2015/16 drought is inside the sample. Chosen for sample size before any edge result was seen.

### Candidate bands (per class, with its own basis and its own weekly trade flows)

Notation: W = CBOT × USD/ZAR in R/t at 10:00 UTC; C = SAGIS export deductions (Gulf FOB in rand −
export realisation at Randfontein: rail, port, financing), latest by `available_date`; cost width =
SAGIS import − export at Randfontein (freight, insurance, port, rail both ways; the Gulf FOB cancels).
All three candidates use **ceiling = floor + cost width**.

- **B0 expanding (round 2):** floor = W × exp(5th percentile of all prior basis).
- **B1 season rule:** floor basis = 10th percentile of daily basis over the harvest window (1 May –
  31 Jul) of a season, accepted only if that season's weekly exports for weeks ending by 31 Jul exceed
  imports for the same weeks. It is usable from the publication date of the last July week. Otherwise
  the last accepted floor carries on. Floor = W × exp(floor basis).
- **B2 Kalman:** latent state s = log competitiveness factor, with observation
  z = log((SAFEX + C) / W), so that floor = W × exp(s) − C.
  - Each trading day, predict: P += q_within; on the first trading day of each marketing year,
    P += q_season.
  - Export pace e = mean weekly exports (kt) over the latest 4 weeks published by that day;
    weight w = clip((e − 5) / (30 − 5), 0, 1).
  - Update when z < s (price below the floor), variance r_below², regardless of pace. Update when
    z ≥ s, variance r_above² / w, only if w > 0. Otherwise no update.
  - Floor on day t uses the state *before* day t's observation.
  - Initialise at the first observation with P = 0.3².
  - **Settings fixed by prior, not estimated:** daily drift sd 0.002 (≈ 3%/yr), season jump sd 0.15
    (the 2021/22 move was 0.24), r_below 0.02, r_above 0.05. (Changed from "estimated on pre-2015 data":
    the one-sided filter has no proper likelihood, and pre-2015 has 6 seasons.) A season jump sd of
    0.075 and 0.30 is reported as sensitivity, never chosen.

### Criteria (evaluation window 1 May 2015 – 30 Sep 2026; yellow decides, white reported)

- **C1 below floor:** share of days with SAFEX < floor, and the longest spell (trading days).
- **C2 calibration:** in *strong export weeks* (that class's actual exports that week ≥ 30 kt, by
  week end; ex-post, evaluation only), the median |SAFEX − floor| in USD/t, plus the median signed
  distance.
- **C3 speed:** for each season with full-season exports > imports, the weeks from 1 May until the
  floor stays within $10/t of its 31 October value through 31 October. Median across seasons.
- **C4 point in time:** truncation test at 50 dates, exact to 1e-9.

**Decision rule:** among candidates that pass C4 and have C1 share ≤ 10%, choose the lowest C2
median |distance| for yellow. C3 is reported. No other band is built. The edge trading test gets its
own addendum after the band is fixed.

## Addendum 3 (after the Addendum 2 run; second attempt at choosing the band)

Written after seeing the Addendum 2 results (commit `36f31f5`). That rule picked B0, and the report
explains why its criteria could be gamed. This is a **second attempt**, labelled as such. Both
attempts are reported. No forward return has been computed for any band.

### New candidate

- **B2s (Kalman, sea exports):** B2 exactly, except the pace is an estimate of **deep-sea**
  exports: published 4-week mean weekly exports minus a cross-border baseline, floored at 0. The
  baseline is the trailing 12 months of cross-border exports (`exports_whole_border`, each month as
  first published) ÷ 52.18, from the latest monthly release on or before the day. It needs at least
  9 months and is scaled to 12. Same pre-registered filter settings; no sensitivity runs.

Candidates scored: B0, B1, B2, B2s.

### New criteria (evaluation window unchanged; yellow decides, white reported)

Monthly flows for evaluation use each month's latest published values (ex post, evaluation only).

- **C1′ breaches:** share of days SAFEX is more than **$10/t below** the floor (the desk trigger).
  Longest such spell also reported.
- **C5 import seasons:** days in *deficit months* (that class's imports > exports in the month).
  The median SAFEX − floor must be at least **half the median cost width** on the same days.
  Pass/fail.
- **C2′ accuracy:** days in *strong sea-export months* (harbour exports ≥ 100 kt that month): median
  |SAFEX − floor| in USD/t.
- **C3′ speed:** for seasons with harbour exports ≥ 500 kt, the weeks from 1 May to the first day in a
  strong sea-export month of that season with |SAFEX − floor| ≤ $10. Seasons where it never happens
  count as 52 weeks. Median across seasons; reported, not decisive.
- **C4 point in time:** truncation test at 50 dates, exact to 1e-9 (B2s included).

**Decision rule:** a candidate is eligible if it passes C4, C1′ ≤ 10% and C5. Among eligible
candidates, choose the lowest C2′ for yellow. If none is eligible, no band is adopted and the study
is parked with that result.

## Known limitations, stated in advance

- The band edges are implied from SAFEX's own history. Early in the sample, the import edge has not
  yet seen a drought (2015/16), so positions above 1 are expected then.
- SAFEX front month is a future; world parity is spot-like. Carry is in the basis.
- White uses its own basis history but the SAGIS cost width is for yellow imports.
- Weekly snapshots are autocorrelated; ~17 seasons is the honest sample size.
