"""Download SAGIS maize files listed in a manifest.

Idempotent: files already present are skipped.
Run:  python ingest/download_sagis.py            # monthly S&D releases -> raw/sagis
      python ingest/download_sagis.py --weekly   # weekly deliveries and trade -> raw/sagis_weekly
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
WEEKLY_MANIFEST = Path(__file__).with_name("sagis_weekly_urls.txt")
WEEKLY_DIR = DATA_DIR / "raw" / "sagis_weekly"
UA = "Mozilla/5.0 (research; dashboard mock)"


def download_all(delay_s: float = 0.4, manifest: Path = MANIFEST,
                 out_dir: Path = OUT_DIR) -> dict[str, list[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    urls = [u.strip() for u in manifest.read_text().splitlines() if u.strip()]
    result: dict[str, list[str]] = {"downloaded": [], "skipped": [], "failed": []}
    for url in urls:
        name = unquote(url.rsplit("/", 1)[-1])
        dest = out_dir / name
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
    weekly = "--weekly" in sys.argv
    res = download_all(manifest=WEEKLY_MANIFEST, out_dir=WEEKLY_DIR) if weekly else download_all()
    print(f"downloaded={len(res['downloaded'])} skipped={len(res['skipped'])} failed={len(res['failed'])}")
    for f in res["failed"]:
        print("FAILED", f)
