"""
logging_setup.py – Consistent logging across the project
=========================================================
Call ``setup_logging()`` once at entry-point scripts; then use
``logging.getLogger(__name__)`` in every module.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.utils.config import get_project_root


def setup_logging(level: int = logging.INFO, log_file: str | None = None) -> None:
    """Configure root logger for console + optional file output.

    Parameters
    ----------
    level : int
        Logging level (default ``logging.INFO``).
    log_file : str, optional
        If given, also write logs to this file inside ``<project_root>/logs/``.
    """
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_dir = get_project_root() / "logs"
        log_dir.mkdir(exist_ok=True)
        fh = logging.FileHandler(log_dir / log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        handlers.append(fh)

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers,
                        force=True)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
