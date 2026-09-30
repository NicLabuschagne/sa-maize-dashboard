# Band position vs the balance sheet

## Status: parked (30 Sep 2026)

**Where it stands**
- The fair-position model describes where yellow sits in the band (OOS R² ≈ 0.2, including
  2023–26). Its gap does not predict 5–40-day moves (round 2).
- A market-revealed export floor was tried twice (Addenda 2 and 3). Both pre-registered rules
  select B0, which is loose, not accurate (below). No floor has passed every economic test.
- The edge trading test was run on B2 at the user's request (Addendum 4). No signal passes, and the
  desk rule on B2 loses (yellow Sharpe −0.58), because B2 follows price.

**Next steps when resumed**
1. B2s leaks upward in deficit seasons through many small updates. Candidate fix: allow upward
   updates only while the published sea-export pace stays above the threshold for several weeks,
   or only in a confirmed export season. That needs a new addendum.
2. CEC crop estimates for the harvest window, where stocks don't explain position.
3. Then the edge trading test, pre-registered as in the Addendum 2/3 discussion: yellow, export zone
   within $10 of the floor, 20-day arb return, three gap signals, the desk rule as benchmark.

## Edge trading test on B2 (Addendum 4, user's choice of band)

Run: `python -m research.band_position.run_edge` (1 run; outputs in `output/edge/`). B2 failed the
pre-registered import-season test; it was used at the user's request, so every result here is
conditional on a floor known to follow price.

**Pre-registered IC tests (yellow, export zone, 20-day arb return): no signal passes.**

| Signal | Dev IC | Holdout IC | Holdout NW t | Holm p |
|---|---|---|---|---|
| S1 detrended gap | −0.04 | −0.11 | −0.79 | 0.87 |
| S2 raw gap | −0.03 | −0.20 | −1.46 | 0.43 |
| S3 unpriced fair move | +0.02 | −0.07 | −0.56 | 0.87 |
| *Benchmark: mean reversion* | −0.15 | −0.23 | −2.02 | — |

Holdout effective N = 27 (25 zone entries); full sample 86 (80 entries). The import zone never
occurred for yellow on B2, because the ceiling (floor + cost width) sits above price whenever the
floor follows price.

**Backtests (net of dashboard costs, May 2015 – Sep 2026, yellow)**

| Rule | Ann. return | Sharpe | Max DD (log) | Time in market | Entries | Hit rate / trade |
|---|---|---|---|---|---|---|
| T0 desk rule | −10.2% | −0.58 | −1.27 | 62% | 80 | 58% |
| T1 zone & S1 > 0 | −9.1% | −0.60 | −1.07 | 39% | 115 | 50% |
| T2 zone & S2 > 0 | −9.1% | −0.56 | −1.16 | 51% | 112 | 53% |
| T3 zone & S3 > 0 | −7.6% | −0.58 | −0.95 | 37% | 145 | 57% |

Holdout (2023–26) Sharpe: T0 −0.46, T1 −0.43, T2 −0.36, T3 −0.48. Deflated Sharpe 0.02–0.03 for all
(5 trials). White is about flat: T0 Sharpe −0.07 overall and +0.38 in the holdout, not tested.

**Why it fails (diagnostics, labelled post hoc)**
- **The zone isn't selective.** SAFEX is within $10 of the B2 floor on 62% of days, because B2
  follows price.
- **The zone picks the wrong moments.** Long the arb on every day loses 5.5%/yr (the yellow basis
  fell from +0.50 in 2016 to −0.05 in 2023). Inside the B2 zone the loss almost doubles. A floor that
  follows price down flags falling markets, and they kept falling: the mean-reversion benchmark IC
  in the zone is −0.18 (t −2.7), i.e. momentum, not reversion.
- **About half the loss is one episode.** Early 2017, the post-drought record crop collapsed the
  basis from about +0.5 to +0.1.
- **The CBOT continuation has roll gaps.** 22 of 72 daily moves above 5% are in July. Zeroing those
  days barely changes yellow T0 (−10.2% → −9.8%/yr), so they are not the cause.

**Reading.** This does not show that the desk's export-parity arb fails. It shows that a floor
learnt from price (B2) can't stand in for a paper calculation: it moves with price and removes the
anchor the trade relies on. A fair test of the desk rule needs a floor that stays put when price
falls without exports (the B2s leak fix), or a real cost-stack export parity.

## Band choice, second attempt (Addendum 3)

Run: `python -m research.band_position.run_bands` (3 runs logged in `output/bands/variants_log.csv`:
2 for Addendum 2, 1 for Addendum 3). The Addendum 3 run crashed after writing every table and plot,
on appending a new column to the variants log. The log was repaired and the append fixed.
`decision.json` was produced by the runner's own rule functions from the saved tables.
Truncation test 100/100 exact for all four candidates, B2s included.

| Yellow | B0 expanding | B1 season rule | B2 Kalman | B2s Kalman, sea exports |
|---|---|---|---|---|
| C1′ days > $10 below floor (≤ 10%) | 1.7% ✓ | 10.4% ✗ | 5.8% ✓ | 3.8% ✓ |
| C1′ longest breach (days) | 24 | 135 | 23 | 15 |
| C5 deficit-month gap vs ½ width (R735) | R1 233 ✓ | R799 ✓ | R241 ✗ | R542 ✗ |
| C2′ median \|distance\|, strong sea-export months | $33.1 | $22.8 | $6.3 | $6.4 |
| C2′ median signed distance | +$33.1 | +$17.0 | +$5.2 | +$5.3 |
| C3′ weeks to first come within $10 (median, 7 seasons) | 52 (never) | 5.0 | 0.4 | 0.4 |

**The pre-registered rule selects B0 again** (the only candidate passing C1′ and C5). That is the
result on record. What it means in practice:
- **B0 is safe but not a floor.** It is never breached and far below price in deficit seasons. But
  in strong sea-export months, yellow sits a median $33/t above it, and in no export season does it
  come within $10 of price. On the desk's $5–10 trigger it would never signal a trade.
- **B2s is accurate when exports flow.** In strong sea-export months, SAFEX sits a median $5/t
  above it, inside the desk trigger. It first comes within $10 of price a median 0.4 weeks after
  1 May. Part of that speed is B2s partly following price, the same weakness C5 catches. Using sea exports fixed 2015/16. It still fails the import-season test because of
  2024/25: deep-sea exports were near zero, but small positive readings (weekly total minus border
  baseline) added up over many days and let the floor rise with price.
- **B1 misses C1′ by 0.4 points**, all from the 2021 lag (a 135-day breach).
- **White:** only 84 strong sea-export days, from one export season. A deep-sea floor for white
  can't be evaluated, which is consistent with white trading cross-border.

No candidate has passed every economic property. The study is parked here, before any trading test.

## Band choice (Addendum 2): pre-registered result and why it should not be adopted

Run: `python -m research.band_position.run_bands` (2 runs logged: the second after fixing a
truncation-rebuild bug in B1, described below; criteria values identical in both). Evaluation
window May 2015 – Sep 2026. Outputs in `output/bands/`.

| Yellow | B0 expanding | B1 season rule | B2 Kalman |
|---|---|---|---|
| C1 days below floor | 2.4% | 20.0% | 27.6% |
| C1 longest spell below (days) | 67 | 144 | 55 |
| C2 median distance above floor, strong export weeks (USD/t) | 33.7 | 16.9 | 5.7 |
| C3 median weeks to settle | 0.1 | 0.4 | 18.0 |
| C4 truncation (50 dates × 2 classes) | exact | exact (after fix) | exact |

White: B0 58.5, B1 21.0, B2 3.3 USD/t on C2; C1 2.9%, 27.3%, 27.8%. The B2 season-jump sensitivity
(0.075, 0.30) barely changes anything.

**The pre-registered rule selects B0** (the only candidate with C1 ≤ 10%). **It should not be
adopted.** The criteria were badly designed, and each can be gamed by a floor that is not a floor:
- **B0 passes C1 by being too low.** In weeks when SA exported ≥ 30 kt, yellow sat $34/t above it,
  and from 2024 it runs R500–1 000/t under price. It isn't export parity.
- **B2 wins C2 mostly by following price.** It climbs with SAFEX in the deficit seasons 2015/16 and
  2024/25, when export parity isn't binding. The cause is that SAGIS weekly exports include
  cross-border trade. Yellow 2024/25: harbour exports 4 kt for the season, cross-border 15 kt/week.
  The filter read that as exports flowing.
- **C3 rewards inertia.** B0 "settles" immediately because it barely learns.
- **B1 behaves most like the economics** (near price in export years, far below in import years) but
  lags. In May 2021 it carried the 2020/21 floor into a CBOT rally and sat about R1 000/t above
  SAFEX for months.

**Truncation bug (fixed).** In a rebuild cut mid-harvest, B1 treated the partial May–July window as
a finished season and published a floor early. The full build was already correct, so every result
number is unchanged. A regression test now covers it.

White harbour exports are small except in 2017 and 2022; white trades cross-border. A deep-sea
export floor for white barely exists, so the edge work stays on yellow.

## Round 2 (Addendum 1): constant-maturity price, linear primary, recent periods, IC

Run: config hash `c320056ceae8`; **2 configurations run in total** (`output/variants_log.csv`).
Truncation test again 100/100 exact. The linear primary was chosen after round 1, so these
numbers are not an independent confirmation.

**Roll fix.** 90-day constant maturity removes every daily move above 8% (front month: 10 white,
7 yellow, up to 42% at a roll). Out-of-sample fit is about the same as round 1 (white 0.22 → 0.20,
yellow 0.24 → 0.21): the season terms had been partly absorbing the roll jumps.

**Fit: primary model (linear, weekly-nowcast domestic STU), OOS R² vs expanding mean**

| Period | White | Yellow |
|---|---|---|
| Full OOS (May 2017 – Sep 2026) | 0.20 | 0.21 |
| 2020-01 – 2026-09 | 0.08 | 0.15 |
| 2023-01 – 2026-09 | 0.18 | 0.19 |
| 2020 / 2021 / 2022 | −0.25 / 0.28 / −0.21 | −0.95 / 0.41 / 0.46 |
| 2023 / 2024 / 2025 / 2026 YTD | 0.12 / 0.17 / 0.13 / 0.57 | 0.37 / −0.03 / 0.40 / −0.53 |

Single years are noisy (52 weeks, about 1–2 independent moves each). Yellow 2026 is negative, but
its mean error is small (0.095 vs 0.084 for the benchmark): position has sat near its long-run mean.

**By season stage (OOS R², within-stage Spearman of STU vs position)**

| Stage | White | Yellow |
|---|---|---|
| Harvest (May–Jul) | 0.26, +0.05 | 0.09, +0.08 |
| Post-harvest (Aug–Oct) | 0.22, −0.56 | 0.25, −0.59 |
| Mid-season (Nov–Jan) | 0.13, −0.57 | 0.14, −0.48 |
| Pre-harvest (Feb–Apr) | 0.15, −0.23 | 0.37, −0.19 |

Correction to round 1: Feb–Apr has full data (193 weeks per class) and a weaker but right-signed
relationship. Harvest is where stocks and position are unrelated. There, deliveries swing the stock
figure, and the market is pricing the new crop's size.

**IC of the fair value (signal = fair position − position; next-close entry; Newey-West lags h − 1)**

| Window | Outcome | Yellow 5 / 10 / 20 / 40d | White 5 / 10 / 20 / 40d |
|---|---|---|---|
| Full OOS | Δ position | 0.00 / 0.00 / 0.02 / 0.06 | 0.02 / 0.00 / 0.00 / 0.06 |
| Full OOS | SAFEX return | −0.04 / −0.05 / −0.04 / 0.00 | 0.00 / −0.01 / 0.02 / 0.11 |
| 2020–2026 | SAFEX return | 0.00 / −0.01 / 0.00 / 0.07 | 0.01 / 0.01 / 0.04 / 0.15 |
| 2023–2026 | SAFEX return | −0.08 / −0.11 / −0.11 / −0.08 | −0.02 / −0.01 / 0.02 / 0.06 |

No NW |t| reaches 2 in any pooled window (largest 1.65). Effective N at 40 days: 57 (full), 41
(2020+), 22 (2023+). Mean reversion without the balance sheet beats the fair-value signal on
forward Δ position at every horizon (e.g. yellow 40d, 2020+: 0.18 vs 0.08).

Per-year IC at 20 days is large and positive in 2024 and 2025 (yellow return 0.57 and 0.36; white
0.65 and 0.80), and negative in 2021. Pooled windows come out near zero while several single years
are strongly positive. That suggests the signal carries a slow level component that is wrong across
years while the within-year variation is right. This is a hypothesis for a future pre-registration,
not a result.

**Round-2 conclusion**
- **The fair value is a reasonable *description* of where price sits.** OOS R² is about 0.2, and
  above zero in the recent windows. The rand fair value tracks yellow closely from 2022
  (`plots/fair_value_2020.png`).
- **It is not yet a short-horizon *trading* signal.** Pooled IC over 5–40 days is about zero, it
  doesn't beat plain mean reversion, and none of it is significant.
- **Gaps close over months, not weeks.** Deviations are very persistent (weekly residual
  autocorrelation 0.98; 0.37–0.56 at 26 weeks). In 2020–2021, yellow SAFEX sat about R1 000/t below
  fair value for over a year before converging. Horizons beyond 40 days have too few independent
  observations to test here.

---

# Round 1: results

Pre-registration: `research/PREREGISTRATION_BAND.md` (written before any fit). Run:
`python -m research.band_position.run` (config hash `bb776642a53b`, **1 configuration run**; variants
log: `output/variants_log.csv`). Tables in `output/*.csv`, plots in `output/plots/`.

Out-of-sample period: May 2017 – Sep 2026, 487 weekly snapshots per class. Each fair value at week t
is fitted on weeks before t. The benchmark is the expanding mean of position.

## Headline

- **The pre-registered decision rule passes for both classes.** The primary model (log STU on the
  weekly-nowcast domestic ratio) beats the naive benchmark out of sample, with a negative slope in
  both halves. White: OOS R² 0.10 (halves 0.10 / 0.10). Yellow: 0.16 (0.10 / 0.24).
- **The balance sheet explains the band position better than the levels regression explains price.**
  Existing Model A: OOS R² 0.01 (white) and 0.05 (yellow). Band position: 0.10 and 0.16. The
  targets and frequencies differ, so treat this as direction, not an exact comparison.
- **It is not the S-curve.** The *linear* model did best (white 0.22, yellow 0.24). Logistic ≈ log.
  Isotonic (no season) was weakest, which says the season term is doing real work.
- **The mid-band hypothesis is not supported.** Mid-band OOS R² is negative in every parametric model; nearly all the
  skill is at the edges. The balance sheet tells you *which side of the band* price should be on,
  not the fine position inside the middle.
- **White does not beat yellow.** Yellow is better on 8 of 12 model × STU pairs, including every
  log and logistic fit. White wins only with isotonic and with linear on the total ratio. The white band is
  weaker (see approximations), so this is not a clean test of the hypothesis yet.
- **Deviations are very persistent.** OOS residual autocorrelation is 0.98 at 1 week, 0.64–0.72 at
  13 weeks and 0.39–0.57 at 26 weeks. Any reversion trade is slow, and there are few independent bets.

## Results by model (OOS R² vs expanding mean, min 5 seasons)

| STU version | Model | White | Yellow |
|---|---|---|---|
| total (dashboard ratio) | linear | 0.19 | 0.11 |
| | log | 0.01 | 0.07 |
| | logistic | 0.01 | 0.09 |
| | isotonic | 0.09 | 0.02 |
| domestic | linear | 0.19 | 0.25 |
| | log | 0.04 | 0.17 |
| | logistic | 0.03 | 0.19 |
| | isotonic | 0.07 | 0.05 |
| **domestic + weekly nowcast** | linear | **0.22** | **0.24** |
| | **log (primary)** | **0.10** | **0.16** |
| | logistic | 0.08 | 0.19 |
| | isotonic | 0.06 | 0.03 |

Regimes, primary model: mid-band −0.29 (white, 233 weeks), −0.22 (yellow, 330); edges +0.20 (254)
and +0.37 (157). Every parametric model and STU version has the same sign pattern. The only
positive mid-band values are yellow isotonic, at +0.04 and +0.02.

Minimum-window sensitivity, primary model (reported, not chosen): 3 seasons 0.04 / 0.11, 4 seasons
0.02 / 0.12, 5 seasons 0.10 / 0.16 (white / yellow). The shorter windows start in 2015/16, and the
extra out-of-sample weeks are the drought, which the early fits hadn't seen.

## Baseline (existing Model A, monthly)

| | White | Yellow |
|---|---|---|
| In-sample R², log real price | 0.13 | 0.16 |
| In-sample R², log nominal price | 0.05 | 0.15 |
| OOS R² vs expanding mean | 0.01 | 0.05 |
| log-cover slope (t) | −0.25 (−5.0) | −0.25 (−5.3) |

Model A carries no world-price term, so its fit is not inflated by trending CBOT/ZAR; it is simply
low. The "inflated levels fit" warning applies to the tested-and-rejected world-price variant on the
Fair Value page.

## Robust vs weak

**Robust**
- Point-in-time construction: truncation test 100/100 exact (50 dates × 2 classes, max difference 0).
- The STU slope is negative in both OOS halves for every parametric model and STU version.
- Domestic-use denominator beats the dashboard's total-disappearance ratio for yellow (log: 0.17 vs
  0.07). Consistent with the export-feedback argument.
- The weekly nowcast helps white (log: 0.04 → 0.10) and is neutral for yellow.

**Weak**
- Level of fit: R² of 0.10–0.24 means most of the band position is unexplained by stocks.
- The trailing 5-year slope went to about zero for 2022–2024 (`plots/slope_log_model.png`). The
  relationship is not stable within the sample.
- Release-day check: yellow −0.14 (right sign, permutation p = 0.07, 182 releases); white +0.06
  (wrong sign, p = 0.40). Weak evidence that the price reacts to the stock surprise itself.
- Full-sample Spearman(STU, position) is about 0 for every version; binned means are hump-shaped.
  Without the season term, low stocks do *not* mean a high position (`plots/position_vs_stu_*.png`).
  *(Corrected in round 2: the weak window is harvest, May–Jul. Feb–Apr has data and a weaker but
  right-signed relationship.)*

## Band facts

| | Below 0 | Above 1 | Longest spell above (days) |
|---|---|---|---|
| White, hybrid | 2.3% | 12.4% | 205 |
| White, SAGIS | 24.1% | 12.9% | 286 |
| Yellow, hybrid | 2.5% | 12.6% | 145 |
| Yellow, SAGIS | 20.6% | 4.7% | 76 |

Most days above 1 fall before the 2015/16 drought taught the ceiling. After it, white positions sat
in 0–0.3 for 2017–2023, because the white ceiling learnt from 2016 is far above normal prices.

## Inputs approximated

- **Only one origin.** The band uses the US Gulf only. There's no Argentine or Brazilian FOB in the
  repo, so no range across origins.
- **White uses yellow's cost width.** White gets its own basis history, but the SAGIS cost width is
  for yellow imports.
- **The ceiling is learnt from history.** The import edge is the historical 95th percentile, not a
  cost stack. The white edge in particular is dominated by 2016.
- **Some costs are bundled.** Insurance and port costs sit inside SAGIS `cif_rand` and the F.O.R.
  figure, not split out.
- **Utilisation between releases is estimated.** Weekly use is the trailing-12m monthly average
  pro-rated, because SAGIS publishes use only monthly.

## Open questions

1. **Old crop vs new crop at the March→May roll.** Position jumps when the front month switches to
   the new-crop contract (e.g. March 2014, March 2025). A constant-crop or constant-maturity price
   series would remove this. This is probably the largest fixable source of noise.
2. **Projected carry-out.** The Feb–Apr failure in the scatter is exactly where a CEC-based projected
   carry-out should help. It is the next pre-registered addition; the data has to be sourced first.
3. **Linear beat the S-curve.** Adopting it would be a post-hoc change of primary. It should be
   re-tested as the pre-registered primary on releases after this date, or in the CEC round.
4. **Why the mid-band fails.** The balance sheet separates the edges well but can't rank positions
   within the middle. That may be true economics (in mid-band, price is set by flows and sentiment),
   or a band that is too wide for white. The data doesn't separate the two yet.
5. **The 2022–2024 breakdown.** The slope went to zero while SA exported at export parity after
   Ukraine. The question is whether an export-pace term (W2 in the weekly pre-registration) belongs
   in the model.

## Next stage (not run)

The decision rule allows the tradeability stage: OOS residual vs forward 5/10/20/40-day change in
position, Newey-West lags ≥ h − 1, effective N ≈ T / h. Given the persistence above, expect long
horizons to be where any edge is, and very few independent observations. That needs its own
pre-registration and `statsmodels`.
