# Pre-registration: improving the SAFEX maize release signal

Committed before any of the evaluations below were run. The point of committing first is that the
hypotheses, target, split and decision rule cannot drift to fit what the data turns out to say.

## Baseline

`S0 = -z_A` — the out-of-sample residual z of Model A (log real front-month price on season and
log months of cover), sign flipped so that a positive signal means "expect the price to rise".

## Target and horizon

Roll-adjusted forward log return of the front month over **10 trading days**, entered at the first
close at least one day after each SAGIS release. Fixed in advance: the earlier work located the edge
at 5-10 days, and choosing the horizon after seeing these results would be another trial.

## Split

- **Development:** releases before 1 May 2020 (marketing years up to 2019/20)
- **Holdout:** releases from 1 May 2020 onward (2020/21 to 2026/27)

Honest caveat: the post-2017 period has already been inspected at the level of the baseline's IC.
No data here is virgin. The mitigations are this document and the multiple-testing correction.

## Hypotheses (each has an economic reason and no parameter fitted to this data)

| # | Signal | Reason |
|---|---|---|
| H1 | `-news`, where news = this release's change in log cover minus the average change for that marketing-year month in prior seasons | The market already knows the level of stocks; what a release delivers is the *surprise* in how they moved. |
| H2 | mean of `-z_A` and `-z_D` (fair-value and parity-basis residuals), equal weight | Two imperfectly correlated signals about the same mispricing; combining them should raise IC (fundamental law). No fitted weights. |
| H3 | `-z_A + trend / expanding sd(trend)`, equal weight | Value plus momentum. SAFEX trends (positive autocorrelation, positive gross trend Sharpe); a dislocation with momentum behind it should persist longer than one without. |
| H4 | `-z` from a Huber-robust expanding fit (c = 1.345, the textbook default) instead of OLS | The 2015/16 drought is a handful of extreme points that bend an OLS line; a robust fit should give cleaner residuals. |
| H5 | White minus yellow: `z_A(yellow) - z_A(white)`, scored against white-minus-yellow forward return | A relative-value pair. The rand and CBOT move both classes together; differencing removes those common shocks and should raise signal-to-noise. |

H1-H4 are scored on white (WMAZ) and yellow (YMAZ) separately; H5 is one test. That is **9 tests**.

## Decision rule

A hypothesis counts as an improvement only if **all** of the following hold:

1. Holdout rank IC is above the baseline's holdout rank IC on the same product and releases.
2. Holdout IC is significant at 5% after **Holm correction across all 9 tests** (p-values from a
   circular block bootstrap, block 6, two-sided).
3. Development IC has the same sign as holdout IC — a signal that flips sign between periods is
   not a signal.

Anything that fails is reported, not dropped. An improvement that passes is then run through the
backtest with the default SAFEX cost model before it goes anywhere near the dashboard.

---

## Results (appended after the single run — the sections above are unchanged since `81d421f`)

Run: `python research/evaluate_hypotheses.py` → `research/results_hypotheses.csv`

| Product | Signal | Dev IC | Holdout IC | Baseline holdout IC | Holm p | Pass |
|---|---|---|---|---|---|---|
| WMAZ | S0 baseline | +0.303 | +0.230 | — | — | — |
| YMAZ | S0 baseline | +0.185 | +0.203 | — | — | — |
| WMAZ | H1 news | −0.003 | +0.106 | +0.230 | 1.00 | no |
| WMAZ | H2 + parity | +0.303 | +0.188 | +0.230 | 0.69 | no |
| WMAZ | H3 value + momentum | +0.214 | +0.268 | +0.230 | 0.22 | no |
| WMAZ | H4 Huber | +0.303 | +0.232 | +0.230 | 0.37 | no |
| YMAZ | H1 news | −0.110 | +0.054 | +0.203 | 1.00 | no |
| YMAZ | H2 + parity | +0.169 | +0.176 | +0.203 | 0.69 | no |
| YMAZ | H3 value + momentum | +0.031 | +0.235 | +0.203 | 0.42 | no |
| YMAZ | H4 Huber | +0.179 | +0.197 | +0.203 | 0.69 | no |
| WMAZ-YMAZ | H5 pair | +0.135 | +0.050 | +0.217* | 1.00 | no |

\*The pre-registration named no baseline for the pair; the mean of the two outright baselines was
fixed in the script before the result was read.

**Outcome: no hypothesis meets the decision rule. The dashboard is unchanged.**

- The baseline replicates out of sample: +0.30 → +0.23 on white, +0.19 → +0.20 on yellow. Its IC peaks
  at 5-10 days and decays with horizon, which is the pattern expected of a real signal, not a leak.
- H3 (value + momentum) beats the baseline on both products with a stable sign, but by about 0.03-0.04
  IC against a standard error near 0.12. It is the one candidate worth re-testing as new releases arrive.
- H4 shows the OLS fit is not being bent by the drought; H2 and H5 confirm the parity and white-yellow
  information lives in relative legs, not in the outright forecast; H1 finds no edge in the size of
  the stock surprise itself.
