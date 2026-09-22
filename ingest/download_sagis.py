"""Download SAGIS monthly maize releases listed in sagis_maize_urls.txt.

Idempotent: files already present in DATA_DIR/raw/sagis are skipped.
Run:  python ingest/download_sagis.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from urllib.parse import unquote
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR  # noqa: E402

MANIFEST = Path(__file__).with_name("sagis_maize_urls.txt")
OUT_DIR = DATA_DIR / "raw" / "sagis"
UA = "Mozilla/5.0 (research; dashboard mock)"


def download_all(delay_s: float = 0.4) -> dict[str, list[str]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    urls = [u.strip() for u in MANIFEST.read_text().splitlines() if u.strip()]
    result: dict[str, list[str]] = {"downloaded": [], "skipped": [], "failed": []}
    for url in urls:
        name = unquote(url.rsplit("/", 1)[-1])
        dest = OUT_DIR / name
        if dest.exists() and dest.stat().st_size > 0:
            result["skipped"].append(name)
            continue
        try:
            with urlopen(Request(url, headers={"User-Agent": UA}), timeout=60) as r:
                dest.write_bytes(r.read())
            result["downloaded"].append(name)
        except Exception as exc:  # noqa: BLE001 - log and continue
            result["failed"].append(f"{name}: {exc}")
        time.sleep(delay_s)
    return result


if __name__ == "__main__":
    res = download_all()
    print(f"downloaded={len(res['downloaded'])} skipped={len(res['skipped'])} failed={len(res['failed'])}")
    for f in res["failed"]:
        print("FAILED", f)
