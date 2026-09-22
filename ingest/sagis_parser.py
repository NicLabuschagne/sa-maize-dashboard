"""Parse a SAGIS monthly maize release (.xls) into a long, point-in-time table.

Each release has four column blocks, each split White / Yellow / Total:
    prev_month   - the month before the latest, revised
    latest_month - the latest month, "preliminary"
    ytd          - marketing-year-to-date (May .. latest month)
    ytd_prior    - same window, prior marketing year
Rows are balance-sheet lines: opening stock, deliveries, imports, utilisation
breakdown, exports breakdown, sundries, closing (unutilised) stock, stock by holder.

Output is one row per (period_type, attribute, grain_class) with values in tons.
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd

CLASS_COLS = {"white": 0, "yellow": 1, "total": 2}
BLOCK_START = [3, 6, 9, 13]
BLOCK_TYPES = ["prev_month", "latest_month", "ytd", "ytd_prior"]

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
MONTHS.update({"mei": 5, "mrt": 3, "okt": 10, "des": 12, "sept": 9})

# (regex on lowercased label, canonical attribute). Order matters: first match wins.
LABEL_MAP: list[tuple[str, str]] = [
    (r"^\(a\)\s*opening stock", "opening_stock"),
    (r"^\(b\)\s*acquisition", "acquisition"),
    (r"^deliveries directly from farms", "deliveries"),
    (r"^imports destined for rsa", "imports"),
    (r"^\(c\)\s*utilisation", "utilisation"),
    (r"^processed for the local market", "processed_local"),
    (r"^human consumption", "human_consumption"),
    (r"^animal feed", "animal_feed"),
    (r"^gristing", "gristing"),
    (r"^bio-?fuel", "biofuel"),
    (r"^withdrawn by producers", "withdrawn_producers"),
    (r"^released to end-?consumer", "released_end_consumer"),
    (r"^\(d\)\s*rsa exports", "exports"),
    (r"^products", "exports_products"),
    (r"^african countries", "exports_products_africa"),
    (r"^other countries", "exports_products_other"),
    (r"^whole maize", "exports_whole"),
    (r"^border posts", "exports_whole_border"),
    (r"^harbours", "exports_whole_harbour"),
    (r"^\(e\)\s*sundries", "sundries"),
    (r"^net dispatches", "net_dispatches"),
    (r"^surplus\(-\)/deficit", "surplus_deficit"),
    (r"^\(f\)\s*unutilised stock", "closing_stock"),
    (r"^\(g\)\s*stock stored at", "stock_stored"),
    (r"^storers", "stock_storers_traders"),
    (r"^processors", "stock_processors"),
]
STOP_RE = re.compile(r"^\(h\)")  # section (h) re-uses labels for a separate ledger


def _clean(v: object) -> str:
    return re.sub(r"\s+", " ", str(v)).strip() if pd.notna(v) else ""


def _parse_month_label(text: str) -> tuple[int, int] | None:
    """'Dec/Des 2003' -> (2003, 12); 'Jul 2026' -> (2026, 7). None if not a month."""
    m = re.search(r"([A-Za-z]{3,4})(?:/[A-Za-z]+)?\s+(\d{4})", text)
    if not m:
        return None
    mon = MONTHS.get(m.group(1).lower()[:4]) or MONTHS.get(m.group(1).lower()[:3])
    return (int(m.group(2)), mon) if mon else None


def _detect_unit_multiplier(df: pd.DataFrame) -> float:
    head = " ".join(_clean(v) for v in df.iloc[:6].values.ravel())
    if re.search(r"'?000\s*t\b", head):
        return 1000.0
    return 1.0


def _find_header_row(df: pd.DataFrame) -> int:
    for i in range(min(12, len(df))):
        if all(_clean(df.iat[i, c]).lower().startswith("white") for c in BLOCK_START):
            return i
    raise ValueError("White/Yellow/Total header row not found")


def _block_labels(df: pd.DataFrame, header_row: int) -> list[str]:
    """Best text label for each block from the rows just above the header."""
    labels = []
    for c in BLOCK_START:
        cands = [_clean(df.iat[r, c]) for r in range(max(0, header_row - 3), header_row)]
        cands = [t for t in cands if t and not re.search(r"progressive|preliminary", t, re.I)]
        labels.append(cands[-1] if cands else "")
    return labels


def _vintage_from_name(path: Path) -> tuple[date | None, bool]:
    m = re.search(r"(\d{8})([a-zA-Z]?)\.xls$", path.name)
    if not m:
        return None, False
    d = m.group(1)
    return date(int(d[:4]), int(d[4:6]), int(d[6:8])), m.group(2).lower() == "f"


def parse_sagis_file(path: str | Path) -> dict:
    """Parse one monthly release. Returns {"ok", "rows", "meta", "error"}."""
    path = Path(path)
    vintage, is_final = _vintage_from_name(path)
    if vintage is None:
        return {"ok": False, "rows": None, "meta": {"file": path.name}, "error": "not a monthly release"}
    try:
        df = pd.read_excel(path, header=None)
        header_row = _find_header_row(df)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "rows": None, "meta": {"file": path.name}, "error": str(exc)}

    mult = _detect_unit_multiplier(df)
    labels = _block_labels(df, header_row)
    latest = _parse_month_label(labels[1]) or _parse_month_label(labels[0])
    if latest is None:
        return {"ok": False, "rows": None, "meta": {"file": path.name, "labels": labels},
                "error": "could not parse latest-month label"}
    year, month = latest
    my_start = year if month >= 5 else year - 1

    rows: list[dict] = []
    for i in range(header_row + 1, len(df)):
        label = next((_clean(df.iat[i, c]) for c in (0, 1, 2) if _clean(df.iat[i, c])), "")
        if not label:
            continue
        if STOP_RE.match(label):
            break
        attr = next((a for pat, a in LABEL_MAP if re.match(pat, label.lower())), None)
        if attr is None:
            continue
        for btype, bstart, blabel in zip(BLOCK_TYPES, BLOCK_START, labels):
            for cls, off in CLASS_COLS.items():
                v = df.iat[i, bstart + off]
                if isinstance(v, str):
                    v = v.replace(" ", "").replace(",", "")
                v = pd.to_numeric(v, errors="coerce")
                if pd.isna(v):
                    continue
                rows.append({"period_type": btype, "period_label": blabel,
                             "attribute": attr, "grain_class": cls, "value_t": float(v) * mult})

    out = pd.DataFrame(rows)
    out.insert(0, "vintage_date", pd.Timestamp(vintage))
    out.insert(1, "latest_month", pd.Timestamp(year=year, month=month, day=1))
    out.insert(2, "marketing_year", f"{my_start}/{(my_start + 1) % 100:02d}")
    out.insert(3, "is_final", is_final)
    out.insert(4, "source_file", path.name)
    meta = {"file": path.name, "vintage_date": vintage, "latest_month": (year, month),
            "unit_multiplier": mult, "block_labels": labels, "n_rows": len(out)}
    return {"ok": True, "rows": out, "meta": meta, "error": None}
