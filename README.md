# South Africa Maize Research Dashboard

A point-in-time research platform for SAFEX white and yellow maize: supply & demand, fair value,
analogues and a backtest engine, built on SAGIS balance sheets and JSE futures.

## Pages

| Page | What it does |
|---|---|
| **Home** | Desk monitor — market, fair value, deviation and signal per market, replayable at any past release |
| **Overview** | Cover and price for the selected class |
| **Supply and Demand** | SAGIS series by release, season-to-date overlays, weekly delivery and trade pace, revision tracking, full release tables |
| **Price** | Front month, calendar spreads, white/yellow premium, forward curve |
| **Fair Value** | Four models (outright, calendar spread, white premium, import/export parity) with IC tests, tercile tables and stability splits |
| **Analogues** | Compound state builder, forward-path fans, horizon distributions and a backtest engine with overfitting controls |
| **Positioning** | Trend-model overlay: signal panel, implied flow, validation against CFTC positioning on CBOT corn |
| **Ask** | Natural-language questions answered by SQL against the warehouse |

## Method in one paragraph

Everything is **point-in-time**. Each SAGIS release is stored under its own `vintage_date` and never
restated, so a figure is always what was published that day. Fair value is fitted on an expanding
window ending the month before, so no model scores a point it has seen. Forward returns enter at the
first close at least one day after a release, excluding the announcement move. Parity uses the CBOT
and USD/ZAR prints at 10:00 UTC — the SAFEX mark — because the CBOT settle happens after SAFEX
closes and using it would be look-ahead.

## Running locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/Home.py
```

The committed `data/warehouse.duckdb` contains everything the app needs. Raw source files are not
included.

## Rebuilding the warehouse

Only needed to refresh data. Requires the raw JSE workbooks in `data/raw/` and `LSE_API_KEY` set.

```bash
python ingest/download_sagis.py    # ~350 monthly SAGIS maize releases
python ingest/build_warehouse.py   # prices + point-in-time balance sheet
python ingest/fetch_lse.py         # CPI, USD/ZAR, CBOT corn, 10:00 UTC snapshots
python ingest/build_signals.py     # fitted model output -> signals table
python ingest/download_sagis.py --weekly && python ingest/build_weekly.py   # weekly deliveries, imports, exports
python ingest/build_parity.py      # SAGIS import/export parity band (raw files from sagis-historic-information)
```

## Tests

```bash
python -m pytest -q
```

185 tests covering the parsers (including a reconciliation of weekly to monthly SAGIS), the point-in-time joins, the models, the overfitting statistics,
the positioning nowcast and a headless render of every page.

## Optional: the Ask page

Works out of the box on a free rules-based backend that maps questions to parameterised SQL — no
account, no network, no cost. To use Claude instead, set `ANTHROPIC_API_KEY` (env var, or in
`.streamlit/secrets.toml` locally, or in the Streamlit Cloud app's Secrets panel). Add
`ANTHROPIC_WORKSPACE_ID` only if the key is org-scoped rather than created inside a workspace.

## Data

- **SAGIS** — South African Grain Information Service, monthly maize supply & demand and weekly
  producer deliveries / RSA imports and exports, public.
- **JSE / SAFEX** — physically settled grain futures. Raw workbooks are **not** redistributed here;
  only derived series in the warehouse.
- **London Strategic Edge** — CBOT corn, USD/ZAR, South African CPI.
