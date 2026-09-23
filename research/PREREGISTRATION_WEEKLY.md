# Pre-registration: does SAGIS weekly data improve the fair-value models?

Committed before any evaluation involving returns was run. The only checks run beforehand were
mechanical: the weekly files parse, weekly season totals reconcile to the monthly balance sheet
(`tests/test_weekly.py`), and the monthly stock identity holds to within sundries.

## Why weekly data could help

A monthly SAGIS release reports the month that ended ~25 days earlier. By release day, the weekly
files already show about three weeks of the *following* month's producer deliveries and about two
weeks of its exports and imports. The market sees those weekly prints, so the stock level the models
use is stale on the day they trade. The second channel is demand: the export pace against a normal
season is information the monthly cover ratio carries only with a lag.

## Models, products and baseline

The four fair-value strategy types, as in the dashboard:

| Model | Product(s) | y | Forward outcome (10 trading days) |
|---|---|---|---|
| A outright | WMAZ, YMAZ | log real front-month price | log return, roll-adjusted |
| B calendar spread | WMAZ, YMAZ | annualised 2nd−1st spread | change in spread |
| C white premium | WMAZ−YMAZ | white premium % of yellow | change in premium |
| D parity basis | YMAZ | log(SAFEX / CBOT×USDZAR at 10:00 UTC) | SAFEX minus parity return |

D is yellow only: parity work centres on yellow, and white has no liquid world benchmark.

**Baseline `S0 = −z`** for each model: the existing out-of-sample expanding-fit residual z, sign flipped
so a positive signal means "expect y to rise". Six model-products.

## Hypotheses (no parameter fitted to this data)

**W1: nowcast cover.** Replace each release's closing stock with a nowcast using only weekly data
published by the release date:

    stock_nc = closing_stock
             + Σ weekly deliveries + Σ weekly imports − Σ weekly exports
             − (utilisation_12m / 12) × n_weeks × 7 / 30.44

Σ runs over weeks ending after the release's reported month and published by the release date. The
same weeks are used for all three flows, limited by the slower trade lag (available_date = week end +
12 days). Domestic use is pro-rated from the trailing-12-month monthly average because SAGIS publishes
it only monthly. Cover is `stock_nc / (disappearance_12m / 12)`. Everything else (fit, standardisation,
entry) is unchanged. Signal `−z` from the refitted model.

- A, B, D: own-class nowcast cover.
- C: `log cover_nc(white) − log cover_nc(yellow)`.

**W2: export pace.** At each release, take season-to-date weekly exports through the last week
published by the release date. Compare them with the mean of up to five prior seasons at the same
week number (at least three required). Scale by the trailing-12-month disappearance:

    pace = (std_exports − prior_avg_std_exports) / disappearance_12m

The pace is standardised by its own expanding sd over prior releases (shift 1, min 12). It is added
to the baseline with equal weight, `S = −z + sign × pace_z`. The sign is fixed by economics:

| Model | sign | Reason |
|---|---|---|
| A | + | faster exports draw stocks down, bullish outright |
| B | − | export demand is for prompt grain; it lifts the front leg and narrows the carry |
| C | + on (pace_white − pace_yellow) | relative export demand lifts the relative price |
| D | + | export demand pulls SAFEX up relative to the world price |

W1 and W2 on six model-products: **12 tests**.

## Target, split, decision rule (unchanged from the first pre-registration)

- Forward outcome at **10 trading days**, entered at the first close at least one day after the release.
- **Development:** releases before 1 May 2020. **Holdout:** from 1 May 2020.
- A hypothesis passes only if **all** of the following hold:
  1. its holdout rank IC beats the baseline's on the same releases;
  2. the holdout IC is significant at 5% after **Holm correction across all 12** (circular block
     bootstrap, block 6, two-sided);
  3. the development IC has the same sign as the holdout IC.

Anything that fails is reported, not dropped. A pass then goes through the backtest with the default
cost model before it touches the dashboard.

## Known limitations, stated in advance

- Past-season weekly files are the finalised versions. SAGIS books corrections in an Adjustments column
  in the week they are made rather than restating old weeks, so the finals are close to point-in-time,
  but not exactly.
- Weekly deliveries start in 2006/07 and trade in 2003/04. Both cover the whole priced sample.
- Exports of maize products are not in the weekly files and are left out of the nowcast. They are
  small relative to stocks.
- Not tested here: using weekly data to trade *between* monthly releases. That changes the event set,
  so it is not comparable to these baselines. It would need its own pre-registration.

---

## Results (appended after the single run; the sections above are unchanged since `28ebf0c`)

Run: `python research/evaluate_weekly.py` → `research/results_weekly.csv`

| Model | Hypothesis | Dev IC | Holdout IC | Baseline holdout IC | Raw p | Holm p | Pass |
|---|---|---|---|---|---|---|---|
| A WMAZ | W1 nowcast | +0.293 | +0.217 | +0.230 | 0.07 | 0.66 | no |
| A WMAZ | W2 export pace | +0.167 | **+0.303** | +0.230 | 0.01 | 0.14 | no |
| A YMAZ | W1 nowcast | +0.196 | +0.209 | +0.203 | 0.11 | 0.86 | no |
| A YMAZ | W2 export pace | +0.245 | **+0.257** | +0.203 | 0.04 | 0.42 | no |
| B WMAZ | W1 nowcast | +0.094 | +0.048 | +0.030 | 0.67 | 1.00 | no |
| B WMAZ | W2 export pace | +0.312 | −0.009 | +0.030 | 0.93 | 1.00 | no |
| B YMAZ | W1 nowcast | +0.189 | −0.027 | −0.076 | 0.82 | 1.00 | no |
| B YMAZ | W2 export pace | +0.181 | −0.125 | −0.076 | 0.26 | 1.00 | no |
| C premium | W1 nowcast | +0.078 | +0.167 | +0.153 | 0.10 | 0.86 | no |
| C premium | W2 export pace | +0.014 | +0.006 | +0.153 | 0.95 | 1.00 | no |
| D YMAZ parity | W1 nowcast | +0.188 | +0.160 | +0.146 | 0.12 | 0.86 | no |
| D YMAZ parity | W2 export pace | +0.204 | +0.121 | +0.146 | 0.23 | 1.00 | no |

**Outcome: no hypothesis meets the decision rule. The models are unchanged.**

- **W1 (nowcast cover)** is a wash. It moves the holdout IC by −0.01 to +0.05 and never significantly.
  About two weeks of trade are published by release day. Off-season that trims stock by about 5%; it
  matters only at harvest, when deliveries arrive faster than the monthly figure shows. Stock levels
  change slowly, so the monthly figure is not stale enough for the update to change the ranking.
- **W2 (export pace)** is the one real candidate, and only for the **outright**. It raises the holdout IC
  on both classes (white +0.23 → +0.30, yellow +0.20 → +0.26) with a stable sign, and the raw p-values
  are 0.01 and 0.04. It fails the Holm correction across 12 tests. On spreads, the premium and the
  parity basis it adds noise or hurts. Export demand shows up in the outright price, not in the
  relative legs.
- **B (calendar spread)** has no holdout edge with or without weekly data. Its baseline holdout IC is
  about zero.
- As with H3 (value + momentum) in the first pre-registration, W2 on the outright should be
  re-tested on releases after this date, as a single pre-specified test with no multiple-testing
  penalty.
