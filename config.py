"""Central config. Paths and settings come from env vars with sane defaults."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DASHBOARD_DATA_DIR", ROOT / "data"))
APP_TITLE = os.getenv("DASHBOARD_TITLE", "Dashboard")
DB_PATH = Path(os.getenv("DASHBOARD_DB", DATA_DIR / "warehouse.duckdb"))
