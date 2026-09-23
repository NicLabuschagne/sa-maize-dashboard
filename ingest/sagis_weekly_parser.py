"""Parse SAGIS weekly maize files: producer deliveries and RSA imports/exports.

SAGIS has changed layout six times since 2003. Rather than one parser per layout, each sheet is
read the same way: find the week table (week number 1-53 in the first column), read the bilingual
header rows above it, and classify columns by label. Every weekly column is then checked against
its progressive column - progressive must equal the running sum of weekly - so a mis-read column
fails loudly instead of loading quietly.

Point-in-time: each week gets an `available_date`, the day its figure would have been public.
Deliveries appear five days after the Friday week-end, exports twelve (the export file runs a
week behind). Corrections are booked in an Adjustments column in the week they are made rather
than back-filled into old weeks, which keeps the weekly series close to first-published values.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

PUB_LAG_DAYS = {"deliveries": 5, "exports": 12, "imports": 12}
MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
MONTHS.update({"mei": 5, "mrt": 3, "okt": 10, "des": 12})


def season_start(path: Path) -> int:
    m = re.search(r"(20\d{2})-(\d{2})", path.name)
    if not m:
        raise ValueError(f"no season in {path.name}")
    return int(m.group(1))


def _end_date(label: str, default_year: int) -> pd.Timestamp | None:
    """Last date in a week label: '25/04 - 01/05/2026', '3 - 9 May 2008', '02 May/Mei - 08 May/Mei 2026'."""
    tail = str(label).split("-")[-1].strip()
    m = re.search(r"(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?", tail)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), m.group(3)
    else:
        m = re.search(r"(\d{1,2})\s+([A-Za-z]{3})[A-Za-z/]*\s*(\d{2,4})?", tail)
        if not m or m.group(2).lower() not in MONTHS:
            return None
        d, mo, y = int(m.group(1)), MONTHS[m.group(2).lower()], m.group(3)
    year = int(y) if y else default_year
    year = year + 2000 if year < 100 else year
    try:
        return pd.Timestamp(year=year, month=mo, day=d)
    except ValueError:
        return None


def _week_rows(df: pd.DataFrame) -> pd.DataFrame:
    # Holiday weeks are published together: '*35' or '33 - 35'. The combined figure is booked to
    # the last week of the range, the week its publication covers up to.
    wk = pd.to_numeric(df[0].astype(str).str.extract(r"^\s*\*?(?:\d+\s*-\s*)?(\d+)(?:\.0)?\s*$")[0],
                       errors="coerce")
    ok = wk.between(1, 53) & (wk == wk.round()) & df[1].notna()
    rows = df[ok].copy()
    rows["week"] = wk[ok].astype(int)
    return rows.drop_duplicates("week", keep="first")


def _week_ends(rows: pd.DataFrame, start_year: int) -> pd.Series:
    """Anchor on the first label that parses, then step seven days - labels without a year
    (common in old files) take their year from the anchor, not from a guess."""
    for _, r in rows.iterrows():
        yr = start_year if r["week"] < 36 else start_year + 1
        end = _end_date(r[1], yr)
        if end is not None:
            w1 = end - pd.Timedelta(days=7 * (int(r["week"]) - 1))
            return rows["week"].map(lambda w: w1 + pd.Timedelta(days=7 * (w - 1)))
    raise ValueError("no parsable week label")


def _unit(df: pd.DataFrame, first_row: int) -> float:
    head = " ".join(str(v) for v in df.iloc[:first_row].values.ravel() if pd.notna(v))
    return 1000.0 if re.search(r"'000", head) else 1.0


def _num(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def _checked(week: np.ndarray, prog: np.ndarray, where: str) -> None:
    """Progressive must be the running sum of weekly (to rounding)."""
    cs = np.nancumsum(week)
    ok = np.isfinite(prog)
    if ok.sum() and np.nanmax(np.abs(cs[ok] - prog[ok])) > max(2.0, 0.002 * np.nanmax(np.abs(prog))):
        raise ValueError(f"weekly does not sum to progressive in {where}")


def _class_of(text: str) -> str | None:
    t = text.lower()
    if "white" in t or "wit" in t:
        return "white"
    if "yellow" in t or "geel" in t:
        return "yellow"
    if "total maize" in t or "totaal mielies" in t or t.startswith("total_maize"):
        return "total"
    return None


def _flow_of(text: str) -> str | None:
    m = re.search(r"(imports|exports)", text, re.I)
    return m.group(1).lower() if m else None


def _left_label(df: pd.DataFrame, head: range, col: int, fn) -> str | None:
    """First label `fn` recognises, scanning each header row (nearest first) leftward from `col`."""
    for i in head:
        for j in range(col, -1, -1):
            v = df.iat[i, j]
            if pd.notna(v) and (hit := fn(str(v))):
                return hit
    return None


def _frame(rows: pd.DataFrame, start: int, cls: str, flow: str, week: np.ndarray,
           prog: np.ndarray, mult: float, src: str) -> pd.DataFrame:
    ends = _week_ends(rows, start)
    return pd.DataFrame({
        "season": f"{start}/{(start + 1) % 100:02d}", "week": rows["week"].to_numpy(),
        "week_end": ends.to_numpy(), "grain_class": cls, "flow": flow,
        "tons_week": week * mult, "tons_prog": prog * mult, "source_file": src})


# ----------------------------------------------------------------------------- deliveries
def parse_deliveries(path: Path) -> pd.DataFrame:
    start, out = season_start(path), []
    xl = pd.ExcelFile(path)
    for sh in xl.sheet_names:
        df = xl.parse(sh, header=None)
        rows = _week_rows(df)
        if rows.empty:
            continue
        r0 = rows.index.min()
        mult = _unit(df, r0)
        sheet_cls = _class_of(sh)
        # header row naming the weekly total, somewhere in the six rows above the data
        hdr = next(i for i in range(r0 - 1, max(r0 - 7, -1), -1)
                   if any("week total" in str(v).lower() for v in df.iloc[i]))
        wt_cols = [j for j, v in df.iloc[hdr].items() if "week total" in str(v).lower()]
        for j in wt_cols:
            cls = sheet_cls or _left_label(df, range(hdr - 1, max(hdr - 6, -1), -1), j, _class_of)
            if cls is None:
                continue
            week, prog = _num(rows[j]), _num(rows[j + 1])
            _checked(week, prog, f"{path.name}:{sh}:{cls}")
            out.append(_frame(rows, start, cls, "deliveries", week, prog, mult, path.name))
    if not out:
        raise ValueError(f"no delivery table in {path.name}")
    return pd.concat(out, ignore_index=True).drop_duplicates(["season", "week", "grain_class"])


# ----------------------------------------------------------------------------- imports / exports
def parse_trade(path: Path) -> pd.DataFrame:
    start, out = season_start(path), []
    xl = pd.ExcelFile(path)
    for sh in xl.sheet_names:
        s = sh.lower()
        if "rsa imp & exp" in s:                              # combined layout, 2003-2014
            df = xl.parse(sh, header=None)
            rows = _week_rows(df)
            r0 = rows.index.min()
            mult = _unit(df, r0)
            head = range(r0 - 1, max(r0 - 6, -1), -1)
            # every block ends in a progressive column; the weekly total sits just before it
            progs = sorted({j for i in head for j, v in df.iloc[i].items() if "prog" in str(v).lower()})
            for p in progs:
                cls = _left_label(df, head, p, _class_of)
                flow = _left_label(df, head, p, _flow_of)
                if cls in ("white", "yellow") and flow:
                    week, prog = _num(rows[p - 1]), _num(rows[p])
                    _checked(week, prog, f"{path.name}:{sh}:{cls}:{flow}")
                    out.append(_frame(rows, start, cls, flow, week, prog, mult, path.name))
        elif ("rsa exports" in s or "imports for rsa" in s) and _class_of(s) in ("white", "yellow"):
            df = xl.parse(sh, header=None)                    # per-sheet layout, 2014 onward
            rows = _week_rows(df)
            if rows.empty:
                continue
            mult = _unit(df, rows.index.min())
            numeric = [j for j in rows.columns if j not in (0, 1, "week")
                       and np.isfinite(_num(rows[j])).sum() >= len(rows) // 2]
            week, prog = _num(rows[numeric[-2]]), _num(rows[numeric[-1]])
            _checked(week, prog, f"{path.name}:{sh}")
            flow = "exports" if "exports" in s else "imports"
            out.append(_frame(rows, start, _class_of(s), flow, week, prog, mult, path.name))
    if not out:
        raise ValueError(f"no trade table in {path.name}")
    return pd.concat(out, ignore_index=True).drop_duplicates(["season", "week", "grain_class", "flow"])


def parse_all(raw_dir: Path) -> dict:
    frames, log = [], []
    for f in sorted(raw_dir.glob("*.xls*")):
        try:
            d = parse_deliveries(f) if "Prod" in f.name else parse_trade(f)
            frames.append(d)
            log.append({"file": f.name, "ok": True, "rows": len(d), "error": None})
        except Exception as exc:  # noqa: BLE001 - record and carry on
            log.append({"file": f.name, "ok": False, "rows": 0, "error": str(exc)})
    w = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(w):
        w["tons_week"] = w["tons_week"].fillna(0.0)
        w["available_date"] = w["week_end"] + pd.to_timedelta(w["flow"].map(PUB_LAG_DAYS), unit="D")
    return {"weekly": w, "log": pd.DataFrame(log)}
