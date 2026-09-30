"""Load the band-position study settings from TOML.

The file path comes from the BAND_POSITION_CONFIG environment variable, falling back to the
config.toml that sits next to this module, so no path is hard-coded in the analysis code.
"""
from __future__ import annotations

import hashlib
import os
import tomllib
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.toml")


def config_path() -> Path:
    """Where the settings file lives: env var override, else the file shipped with the module."""
    return Path(os.getenv("BAND_POSITION_CONFIG", DEFAULT_CONFIG_PATH))


def load_settings(path: Path | None = None) -> dict:
    """Parse the TOML settings into a nested dict. Kept as plain dicts so tests can override one value."""
    with open(path or config_path(), "rb") as handle:
        return tomllib.load(handle)


def config_hash(path: Path | None = None) -> str:
    """Short hash of the settings file, logged with every run so each variant tried can be counted."""
    return hashlib.sha256(Path(path or config_path()).read_bytes()).hexdigest()[:12]
