"""Parse SAGIS historic maize import and export parity (weekly, Friday, US No3 yellow basis).

Each workbook stacks one block per year: a row of Friday dates, then labelled rows of costs. Labels
drift a little over the years, so rows are matched by pattern, not position.

Export realisation is quoted at Randfontein (Gulf FOB less rail to Durban and harbour costs) from
2007. Before that it is quoted at the harbour, so it is flagged. Import parity is free-on-rail at the
harbour. Rail from Durban to Randfontein is added to put it on the SAFEX delivery basis.

The prime rate is backed out of the export financing line (30 days of prime on the Gulf value in rand).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

PUB_LAG_DAYS = 7          # indicative Friday calculation; assume it is usable a week later

EXPORT_ROWS = {
    "fob_gulf_usd": r"^CBOT FOB Gulf value",
    "usdzar": r"^Exchange rate",
    "fob_gulf_rand": r"^USA No3Y Maize \(fob\) Gulf",
    "financing_exp": r"^Financing",
    "rail_rand_dbn": r"^Railage",
    "export_realisation": r"^EXPORT REALISATION",
}
IMPORT_ROWS = {
    "freight_usd": r"^Freight rate",
    "cif_rand": r"^Converted to R/t",
    "tariff": r"^Import Tariff",
    "import_for_harbour": r"^F\.O\.R",
}


def _dates(row: pd.Series) -> dict[int, pd.Timestamp] | None:
    d = {j: pd.Timestamp(v) for j, v in row.items() if j and isinstance(v, (pd.Timestamp, np.datetime64))
         or (j and hasattr(v, "year") and hasattr(v, "month"))}
    return d if len(d) >= 3 else None


def parse_block_file(path: Path, rows: dict[str, str]) -> pd.DataFrame:
    df = pd.read_excel(path, header=None)
    recs: dict[pd.Timestamp, dict] = {}
    cur: dict[int, pd.Timestamp] | None = None
    for _, r in df.iterrows():
        d = _dates(r)
        if d:
            cur = d
            continue
        if cur is None or pd.isna(r[0]):
            continue
        label = re.sub(r"\s+", " ", str(r[0])).strip()
        for field, pat in rows.items():
            if re.match(pat, label, re.I):
                for j, dt in cur.items():
                    v = pd.to_numeric(r.get(j), errors="coerce")
                    if pd.notna(v):
                        recs.setdefault(dt, {})[field] = float(v)
                if field == "export_realisation":
                    for dt in cur.values():
                        recs.setdefault(dt, {})["at_harbour"] = "harbour" in label.lower() and "durban" in label.lower()
    out = pd.DataFrame.from_dict(recs, orient="index").sort_index()
    out.index.name = "date"
    return out


def despike(s: pd.Series, window: int = 9, tol: float = 0.25, log: bool = True) -> pd.Series:
    """Blank typos in the source (e.g. a 56,111 R/t Gulf value) - points far from a centred rolling
    median. Centred is fine here: this cleans the record, it is not a signal."""
    x = np.log(s.where(s > 0)) if log else s
    med = x.rolling(window, center=True, min_periods=3).median()
    return s.where((x - med).abs() <= tol)


def parse_parity(raw_dir: Path) -> dict:
    exp = next(raw_dir.glob("Historic-Parity-Export-Maize*"), None)
    imp = next(raw_dir.glob("Historic-Parity-Import-Maize*"), None)
    if exp is None or imp is None:
        return {"parity": pd.DataFrame(), "error": "parity files missing"}
    e = parse_block_file(exp, EXPORT_ROWS)
    i = parse_block_file(imp, IMPORT_ROWS)
    p = e.join(i, how="outer", rsuffix="_imp").sort_index()
    for c in ("fob_gulf_rand", "export_realisation", "cif_rand", "import_for_harbour", "rail_rand_dbn",
              "freight_usd", "usdzar", "fob_gulf_usd"):
        p[c] = despike(p[c])
    # Durban-block label says 'Durban Harbour' but the row sits after railage: it is Randfontein-based.
    p["export_randfontein"] = np.where(p["rail_rand_dbn"].notna(), p["export_realisation"], np.nan)
    p["import_randfontein"] = p["import_for_harbour"] + p["rail_rand_dbn"]
    p["prime_rate"] = despike(p["financing_exp"] / p["fob_gulf_rand"] * 365 / 30 * 100, tol=1.0, log=False)
    p = p.drop(columns=["at_harbour"], errors="ignore").reset_index()
    p = p[p["date"].notna()].reset_index(drop=True)
    p["available_date"] = p["date"] + pd.Timedelta(days=PUB_LAG_DAYS)
    return {"parity": p, "error": None}
