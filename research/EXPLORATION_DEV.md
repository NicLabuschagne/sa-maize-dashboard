# Exploration: model choice, new data and features (development sample only)

Scope: can the outright (A), calendar-spread (B) or parity (D) model be improved by a different
specification, a new data source, or a new feature?

Discipline: everything below scores **releases before 1 May 2020 only**. The holdout was not
touched. The rule for sending a candidate to a pre-registered holdout test was fixed in
`explore_dev.py` before its first run:

- It must beat the baseline's mean development IC (10 days) by at least 0.05.
- It must not fall below the baseline on any product.

## New data: SAGIS historic import/export parity

SAGIS publishes a weekly (Friday) indicative parity calculation for US No3 yellow maize: export
parity from 2001 and import parity from 2006. It is now in the warehouse as `sagis_parity` (built by
`ingest/build_parity.py`).

- **Export realisation at Randfontein** (the SAFEX delivery point): Gulf FOB + US$10, converted at the
  day's rand rate, less 30 days' financing at prime, harbour loading and rail to Durban.
- **Import parity at Randfontein**: Gulf FOB + IGC freight + insurance, in rand, plus financing,
  discharge, tariff and rail from Durban.
- **Prime rate**, backed out of the financing line. It matches the published history: 12.5% in
  Jan 2007, 15% in Jan 2009, 9.25% in 2015.
- Treated as available 7 days after the calculation date. A handful of source typos (e.g. a 56,111
  R/t Gulf value) are blanked by a rolling-median filter.

## Results (IC with the 10-day forward outcome, development releases)

| Type | Candidate | WMAZ | YMAZ | Mean | vs baseline |
|---|---|---|---|---|---|
| Outright | A0 baseline (log real price ~ log cover) | +0.303 | +0.185 | +0.244 | — |
| Outright | Position in the import/export parity band | +0.314 | +0.166 | +0.240 | −0.004 |
| Outright | Log basis to the band midpoint | +0.322 | +0.178 | +0.250 | +0.006 |
| Outright | Log basis to export parity | +0.302 | +0.156 | +0.229 | −0.015 |
| Outright | Convex cover (x + x²) | +0.306 | +0.183 | +0.244 | +0.000 |
| Spread | B0 baseline (annualised spread ~ log cover) | +0.094 | +0.168 | +0.131 | — |
| Spread | Carry-adjusted: spread − prime | +0.099 | +0.170 | +0.135 | +0.004 |
| Spread | Convex cover (x + x²) | +0.098 | +0.134 | +0.116 | −0.015 |
| Parity | D0 baseline (log SAFEX / CBOT×ZAR ~ log cover) | | +0.198 | +0.198 | — |
| Parity | Position in the SAGIS band | | +0.192 | +0.192 | −0.006 |
| Parity | Log basis to the band midpoint | | +0.198 | +0.198 | +0.000 |
| Parity | Log basis to export parity | | +0.229 | +0.229 | +0.031 |

**No candidate meets the selection rule, so nothing goes to the holdout.**

## Frequency: release day vs weekly between releases (`explore_sampling.py`)

Same fitted fair value, scored every fifth trading day instead of only after releases:

| Model | Release-day IC (n) | Weekly IC (n) |
|---|---|---|
| A outright WMAZ | +0.303 (70) | +0.109 (289) |
| A outright YMAZ | +0.185 (72) | +0.021 (297) |
| B spread WMAZ | +0.094 (70) | +0.001 (289) |
| B spread YMAZ | +0.168 (69) | +0.154 (284) |

## What this says

1. **The anchor is not the problem.** A regression on cover with seasonality, the SAGIS parity band,
   or a basis to either parity all rank releases almost identically. Changing the anchor mostly moves
   the fair-value *level*. The signal comes from how far the price sits from it after a release, and
   those rankings barely change.
2. **The outright edge is a release-day effect.** Held between releases, the same fair value loses most
   of its IC. The market over- or under-reacts to the SAGIS print and corrects over the following
   5–10 days. That is the effect the model captures, which supports keeping it event-driven.
3. **Carry adjustment does not help the spread.** SA rates ranged 7–15% over the sample, but the
   annualised spread's variation around cover is dominated by other things (see below).
4. **Sample size is the binding constraint.** About 70 releases per product in development gives an IC
   standard error near 0.12. Specification changes worth +0.05 cannot be seen at that size. Only
   genuinely new *information* is likely to move the IC by enough to show.

## Where the remaining edge most likely is (not yet built)

In order of expected value:

1. **Crop Estimates Committee (CEC) production estimates.** Current stock says nothing about the
   crop coming in Feb–May, which is what the new-crop contracts trade. The CEC estimate, and its
   surprise against the previous estimate, is SA's direct equivalent of the "his model vs USDA" set-up.
   SAGIS hosts 446 files from 1999 (286 Word .doc, 160 PDF). It needs a PDF/Word parser, so it is a
   data build, not a quick test.
2. **Old-crop / new-crop alignment.** Pricing the old-crop contract with current stock and the new-crop
   contract with projected stock (current + CEC crop − expected use). This depends on 1.
3. **Supply & Demand Estimates Committee (SDEC) projected ending stocks.** The forward-looking balance
   sheet the market actually trades, published by NAMC. Availability of the history is unconfirmed.
4. **Data not available at home:** SAFEX broker-level positioning, rainfall and NDVI over the maize
   triangle (to anticipate CEC revisions), and Durban port and freight line-ups.

Still open, for re-testing on post-2026-09 releases as single pre-specified tests: H3 (value +
momentum) and W2 (export pace) on the outright.
