# Band position vs the balance sheet: results, round 1

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
  Late in the season (Feb–Apr) stocks are always low, and the market is pricing the new crop.

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
