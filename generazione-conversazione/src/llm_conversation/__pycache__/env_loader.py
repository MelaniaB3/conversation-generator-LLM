"""Minimal .env loader that does not require external dependencies.

This module reads a `.env` file (by default from the repository root) and
populates os.environ with any variables that are not already set. It is
intentionally lightweight and supports simple KEY=VALUE lines with optional
quoting. Lines starting with `#` are ignored.
"""
from __future__ import annotations

from pathlib import Path
import os
from typing import Iterable


def _parse_line(line: str) -> tuple[str, str] | None:
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    if "=" not in s:
        return None
    key, val = s.split("=", 1)
    key = key.strip()
    val = val.strip()
    # Remove surrounding quotes if present
    if len(val) >= 2 and ((val[0] == val[-1] == '"') or (val[0] == val[-1] == "'")):
        val = val[1:-1]
    return key, val


def load_dotenv(path: Path | str = ".env") -> None:
    """Load environment variables from a `.env` file into os.environ.

    Existing environment variables are not overwritten.
    """
    p = Path(path)
    if not p.exists():
        return

    try:
        with p.open("r", encoding="utf-8") as f:
            for raw in f:
                parsed = _parse_line(raw)
                if not parsed:
                    continue
                k, v = parsed
                if k not in os.environ:
                    os.environ[k] = v
    except Exception:
        # Fail silently: loader is best-effort and should not raise in production
        return


def load_from_iter(lines: Iterable[str]) -> None:
    """Helper to load env vars from an iterable of strings (useful for tests)."""
    for raw in lines:
        parsed = _parse_line(raw)
        if not parsed:
            continue
        k, v = parsed
        if k not in os.environ:
            os.environ[k] = v
