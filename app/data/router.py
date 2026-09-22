"""Deterministic natural-language -> SQL router. No model, no network, no cost.

This is the free stand-in for the LLM backend on the Ask page. It classifies a question by
keyword pattern, pulls out the entities it needs (grain class, model, threshold, horizon,
year, top-N), and fills a parameterised SQL template. Because the mapping is rules-based it
cannot invent a number or a column - it either matches an intent and returns real rows, or it
says it did not understand and lists what it can answer.

Adding an intent = adding one entry to INTENTS.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

CLASS_WORDS = {"white": "white", "wmaz": "white", "yellow": "yellow", "ymaz": "yellow"}
MODEL_WORDS = {
    "outright": "A", "flat price": "A", "model a": "A", "fair value": "A",
    "spread": "B", "calendar": "B", "carry": "B", "model b": "B",
    "premium": "C", "white vs yellow": "C", "model c": "C",
    "parity": "D", "basis": "D", "cbot": "D", "model d": "D",
}
HORIZONS = {"5d": "fwd_5d", "5 day": "fwd_5d", "10d": "fwd_10d", "10 day": "fwd_10d",
            "1m": "fwd_1m", "1 month": "fwd_1m", "2m": "fwd_2m", "2 month": "fwd_2m",
            "3m": "fwd_3m", "3 month": "fwd_3m", "6m": "fwd_6m", "6 month": "fwd_6m"}
ATTR_WORDS = {
    "closing stock": "closing_stock", "ending stock": "closing_stock", "carry out": "closing_stock",
    "carryout": "closing_stock", "unutilised": "closing_stock", "stock": "closing_stock",
    "deliveries": "deliveries", "producer deliveries": "deliveries",
    "export": "exports", "import": "imports", "utilisation": "utilisation", "utilization": "utilisation",
    "human consumption": "human_consumption", "animal feed": "animal_feed", "feed": "animal_feed",
}


@dataclass
class Route:
    intent: str
    sql: str
    purpose: str
    note: str = ""


def _cls(q: str, default: str | None = "white") -> str | None:
    for w, c in CLASS_WORDS.items():
        if re.search(rf"\b{w}\b", q):
            return c
    return default


def _model(q: str, default: str = "A") -> str:
    for w, m in sorted(MODEL_WORDS.items(), key=lambda kv: -len(kv[0])):
        if w in q:
            return m
    return default


def _horizon(q: str, default: str = "fwd_10d") -> str:
    for w, c in sorted(HORIZONS.items(), key=lambda kv: -len(kv[0])):
        if w in q:
            return c
    return default


def _num(q: str, default: float) -> float:
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:sigma|sd|σ|standard deviation)", q)
    if m:
        return float(m.group(1))
    m = re.search(r"\b(?:above|over|more than|greater than|below|under|less than)\s+(-?\d+(?:\.\d+)?)", q)
    return float(m.group(1)) if m else default


def _topn(q: str, default: int = 10) -> int:
    m = re.search(r"\b(?:top|first|last|latest)\s+(\d+)\b", q)
    n = int(m.group(1)) if m else default
    return max(1, min(n, 100))


def _attr(q: str, default: str = "closing_stock") -> str:
    for w, a in sorted(ATTR_WORDS.items(), key=lambda kv: -len(kv[0])):
        if w in q:
            return a
    return default


def _year(q: str) -> int | None:
    m = re.search(r"\b(20[0-2]\d)\b", q)
    return int(m.group(1)) if m else None


def route(question: str) -> Route | None:
    """Return the SQL to run for this question, or None if no intent matched."""
    q = question.lower().strip()
    cls, mdl, hz, n = _cls(q), _model(q), _horizon(q), _topn(q)

    # --- current state ----------------------------------------------------------------
    if re.search(r"\b(right now|currently|today|latest|current|where are we|state)\b", q) or \
       re.search(r"\bis (white|yellow|it) (rich|cheap|expensive)\b", q):
        return Route("current_state", f"""
SELECT model, grain_class, round(z, 2) AS z,
       CASE WHEN z > 0.5 THEN 'RICH' WHEN z < -0.5 THEN 'CHEAP' ELSE 'neutral' END AS read,
       round(months_cover, 2) AS months_cover, round(front_close) AS safex_r_t,
       vintage_date::DATE AS release
FROM signals
WHERE vintage_date = (SELECT max(vintage_date) FROM signals)
ORDER BY model, grain_class""".strip(),
            "Latest reading from every model",
            "z is positive when the market is rich to the model's fair value. "
            "A = outright price, B = calendar spread, C = white premium, D = parity basis.")

    # --- history of extreme readings --------------------------------------------------
    if re.search(r"\b(when was|when did|last time|history of|episode\w*|occasion\w*)", q) and \
       re.search(r"(\brich\b|\bcheap\b|\bsigma\b|\bz\b|\bexpensive\b|\bdislocat\w*|\bstretch\w*)", q):
        cheap = bool(re.search(r"\bcheap\b", q))
        thr = _num(q, 1.5)
        cmp_, thr = ("<", -abs(thr)) if cheap else (">", abs(thr))
        return Route("extreme_history", f"""
SELECT vintage_date::DATE AS release, latest_month::DATE AS data_to, round(z, 2) AS z,
       round(months_cover, 2) AS months_cover, round(front_close) AS safex_r_t,
       round(100 * fwd_5d, 1) AS fwd_5d_pct, round(100 * fwd_10d, 1) AS fwd_10d_pct,
       round(100 * fwd_1m, 1) AS fwd_1m_pct
FROM signals
WHERE model = '{mdl}' AND grain_class = '{cls}' AND z {cmp_} {thr}
ORDER BY vintage_date DESC
LIMIT {n}""".strip(),
            f"Releases where {cls} model {mdl} was {'cheap' if cheap else 'rich'} beyond {abs(thr)} sigma",
            "Forward returns are measured from the first close at least one day after the release.")

    # --- does the signal work ---------------------------------------------------------
    if re.search(r"\b(does .*(work|predict)|underperform|outperform|hit rate|accuracy|"
                 r"what happens? after|forward return|backtest|performance)\b", q):
        return Route("bucket_performance", f"""
SELECT CASE WHEN z > 1 THEN 'rich (z > 1)'
            WHEN z < -1 THEN 'cheap (z < -1)'
            ELSE 'middle' END AS bucket,
       count(*) AS n,
       round(100 * avg({hz}), 2) AS avg_pct,
       round(100 * median({hz}), 2) AS median_pct,
       round(100 * avg(CASE WHEN {hz} < 0 THEN 1.0 ELSE 0.0 END)) AS pct_negative
FROM signals
WHERE model = '{mdl}' AND grain_class = '{cls}' AND z IS NOT NULL AND {hz} IS NOT NULL
GROUP BY 1
ORDER BY avg_pct""".strip(),
            f"{cls} model {mdl}: {hz.replace('fwd_', '')} forward outcome by signal bucket",
            "A working rich/cheap signal puts the rich bucket at the bottom with the highest "
            "share of negative outcomes.")

    # --- months of cover / stocks over time -------------------------------------------
    if re.search(r"\b(cover\w*|months of cover|stocks?[- ]to[- ]use|tightness)", q):
        yr = _year(q)
        where = f"AND year(latest_month) = {yr}" if yr else ""
        return Route("cover_history", f"""
SELECT latest_month::DATE AS month, grain_class, round(months_cover, 2) AS months_cover,
       vintage_date::DATE AS release
FROM signals
WHERE model = 'A' AND months_cover IS NOT NULL {where}
ORDER BY latest_month DESC
LIMIT {max(n, 24)}""".strip(),
            "Months of cover as published at each release" + (f", {yr}" if yr else ""),
            "Cover = unutilised stock divided by average monthly disappearance over the trailing 12 months.")

    # --- revisions --------------------------------------------------------------------
    if re.search(r"\b(revis\w*|restat\w*|amended|changed? (the|their) (estimate|number))", q):
        attr = _attr(q)
        cls_bs = _cls(q, "total")
        return Route("revisions", f"""
WITH prelim AS (
  SELECT latest_month AS month, vintage_date, value_t AS preliminary
  FROM balance_sheet
  WHERE attribute = '{attr}' AND grain_class = '{cls_bs}'
    AND period_type = 'latest_month' AND NOT is_final),
rev AS (
  SELECT latest_month - INTERVAL 1 MONTH AS month, value_t AS revised
  FROM balance_sheet
  WHERE attribute = '{attr}' AND grain_class = '{cls_bs}'
    AND period_type = 'prev_month' AND NOT is_final)
SELECT p.month::DATE AS month, round(p.preliminary) AS preliminary_t, round(r.revised) AS revised_t,
       round(r.revised - p.preliminary) AS revision_t,
       round(100.0 * (r.revised / nullif(p.preliminary, 0) - 1), 2) AS revision_pct
FROM prelim p JOIN rev r USING (month)
ORDER BY abs(r.revised - p.preliminary) DESC
LIMIT {n}""".strip(),
            f"Largest revisions to {cls_bs} {attr.replace('_', ' ')}",
            "Compares the first published figure with the same month restated one release later.")

    # --- balance sheet ----------------------------------------------------------------
    if re.search(r"\b(balance sheet|s&d|sagis|supply and demand|produc\w*|deliver\w*|export\w*|import\w*|"
                 r"consumption|utilisation|utilization|stock\w*)", q):
        attr = _attr(q)
        cls_bs = _cls(q, "total")
        yr = _year(q)
        where = f"AND year(b.latest_month) = {yr}" if yr else ""
        return Route("balance_sheet", f"""
SELECT b.vintage_date::DATE AS release, b.latest_month::DATE AS data_to, b.marketing_year,
       b.grain_class, round(b.value_t) AS {attr}_t
FROM balance_sheet b
WHERE b.attribute = '{attr}' AND b.grain_class = '{cls_bs}'
  AND b.period_type = 'latest_month' AND NOT b.is_final {where}
ORDER BY b.vintage_date DESC
LIMIT {max(n, 18)}""".strip(),
            f"{attr.replace('_', ' ')} for {cls_bs} maize, as first published",
            "period_type = 'latest_month' and is_final = FALSE keeps this point-in-time.")

    # --- parity / basis ---------------------------------------------------------------
    if re.search(r"\b(parity|basis|cbot|world price|import parity|export parity|vs corn)\b", q):
        widest = bool(re.search(r"\b(widest|biggest|largest|highest|extreme\w*|narrow\w*|tight\w*)", q))
        order = "ORDER BY s.basis DESC" if widest else "ORDER BY s.vintage_date DESC"
        return Route("parity", f"""
SELECT s.latest_month::DATE AS month, s.grain_class,
       round(100 * (exp(s.basis) - 1)) AS basis_pct,
       round(s.front_close) AS safex_r_t, round(s.world_rand) AS parity_r_t,
       round(s.months_cover, 1) AS months_cover, round(s.z, 2) AS basis_z
FROM signals s
WHERE s.model = 'D' AND s.grain_class = '{cls}' AND s.basis IS NOT NULL
{order}
LIMIT {n}""".strip(),
            f"{cls} SAFEX against world parity" + (" - widest first" if widest else " - most recent first"),
            "Parity = CBOT corn x USD/ZAR at the 10:00 UTC SAFEX mark, converted to R/t.")

    # --- calendar spread ---------------------------------------------------------------
    if re.search(r"\b(calendar spread|carry|contango|backwardation|inver\w*|spread\w*)", q):
        return Route("spread", f"""
SELECT latest_month::DATE AS month, grain_class, round(actual, 1) AS spread_pct_ann,
       round(fair_value, 1) AS fair_spread_pct_ann, round(z, 2) AS z,
       round(months_cover, 2) AS months_cover
FROM signals
WHERE model = 'B' AND grain_class = '{cls}' AND actual IS NOT NULL
ORDER BY latest_month DESC
LIMIT {max(n, 18)}""".strip(),
            f"{cls} calendar spread against its stock-implied fair level",
            "Positive = carry, negative = inversion. Annualised percentage of the front price.")

    # --- white vs yellow ----------------------------------------------------------------
    if re.search(r"\b(white premium|white vs yellow|yellow vs white|wy spread|class spread)\b", q):
        return Route("white_yellow", f"""
SELECT latest_month::DATE AS month, round(actual, 2) AS white_premium_pct,
       round(fair_value, 2) AS fair_premium_pct, round(z, 2) AS z
FROM signals
WHERE model = 'C' AND actual IS NOT NULL
ORDER BY latest_month DESC
LIMIT {max(n, 18)}""".strip(),
            "White maize premium over yellow, against its fair level",
            "Expressed as a percentage of the yellow front-month price.")

    # --- prices / curve -------------------------------------------------------------------
    if re.search(r"\b(price\w*|close\w*|front month|curve\w*|term structure|contract\w*|safex|quote\w*)", q):
        curve = bool(re.search(r"\b(curve|term structure|all contracts|forward)\b", q))
        if curve:
            return Route("curve", f"""
SELECT symbol, expiry, expiry_date::DATE AS expiry_date, close AS close_r_t,
       volume, open_interest
FROM prices
WHERE trade_date = (SELECT max(trade_date) FROM prices) AND open_interest > 0
ORDER BY symbol, expiry_date
LIMIT {max(n, 30)}""".strip(), "Forward curve on the most recent trade date")
        return Route("price_history", f"""
SELECT trade_date::DATE AS trade_date, symbol, expiry, close AS close_r_t, volume, open_interest
FROM prices
WHERE symbol = '{"WMAZ" if cls == "white" else "YMAZ"}' AND open_interest > 500
ORDER BY trade_date DESC, open_interest DESC
LIMIT {max(n, 20)}""".strip(), f"Recent {cls} contract closes, liquid contracts only")

    # --- coverage / what data is here -------------------------------------------------
    if re.search(r"\b(what data|how far back|coverage|history|what do you have|tables?|schema|"
                 r"how many rows)\b", q):
        return Route("coverage", """
SELECT 'prices' AS table_name, count(*) AS rows, min(trade_date)::DATE AS first,
       max(trade_date)::DATE AS last FROM prices
UNION ALL SELECT 'balance_sheet', count(*), min(vintage_date)::DATE, max(vintage_date)::DATE FROM balance_sheet
UNION ALL SELECT 'signals', count(*), min(vintage_date)::DATE, max(vintage_date)::DATE FROM signals
UNION ALL SELECT 'macro_snap', count(*), min(date)::DATE, max(date)::DATE FROM macro_snap
UNION ALL SELECT 'macro', count(*), min(date)::DATE, max(date)::DATE FROM macro""".strip(),
            "What is in the warehouse and how far back it goes")

    return None


EXAMPLES = [
    "Is white rich right now?",
    "When was white last more than 1.5 sigma rich?",
    "Do rich readings underperform at 10 days?",
    "Show me months of cover",
    "Largest revisions to closing stock",
    "Widest parity basis for yellow",
    "Show the forward curve",
    "What data do you have?",
]
