"""
config.py – YAML configuration loader
======================================
Loads config.yaml from the project root and exposes it as a nested dict.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


_CONFIG_CACHE: Dict[str, Any] | None = None
_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # …/digital twin/


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return _PROJECT_ROOT


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load and cache the YAML configuration.

    Parameters
    ----------
    path : str or Path, optional
        Explicit path to a YAML file.  Defaults to ``<project_root>/config.yaml``.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    if path is None:
        path = _PROJECT_ROOT / "config.yaml"
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        _CONFIG_CACHE = yaml.safe_load(fh)

    return _CONFIG_CACHE


def get(section: str, key: str | None = None, default: Any = None) -> Any:
    """Convenience accessor for nested config values.

    Examples
    --------
    >>> get("data", "n_patients")
    500
    >>> get("model")
    {…entire model section…}
    """
    cfg = load_config()
    section_data = cfg.get(section, {})
    if key is None:
        return section_data
    return section_data.get(key, default)
