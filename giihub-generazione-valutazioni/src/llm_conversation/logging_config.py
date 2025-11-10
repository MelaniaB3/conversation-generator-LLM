"""
Derived from: https://github.com/famiu/llm_conversation (original repository)
Modified by: Melania Balestri, 2025  adaptations for APP/IE matrix configuration and hosted usage
License: GNU AGPL v3.0 (see LICENSE in project root)
"""

"""Logging configuration module for LLM Conversation package."""

import logging
import os
import sys
from pathlib import Path

try:
    # Import rich at runtime using importlib to avoid static-analysis unresolved import errors.
    import importlib

    rich_console = importlib.import_module("rich.console")
    rich_logging = importlib.import_module("rich.logging")
    Console = getattr(rich_console, "Console")
    RichHandler = getattr(rich_logging, "RichHandler")
    _HAS_RICH = True
except Exception:
    # Rich library not available or import failed; fall back to stdlib logging handlers.
    Console = None  # type: ignore
    RichHandler = None  # type: ignore
    _HAS_RICH = False


def setup_logging() -> None:
    """Set up logging configuration based on environment variables.

    Env vars:
        LLM_CONVERSATION_LOG_LEVEL: Log level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL) (default: disabled)
        LLM_CONVERSATION_LOG_FILE: Log file path (default: stderr only)
    """
    log_level_str = os.getenv("LLM_CONVERSATION_LOG_LEVEL")

    if not log_level_str:
        return  # logging disabled unless explicitly set

    try:
        log_level = getattr(logging, log_level_str.upper())
    except AttributeError:
        print(f"Invalid log level: {log_level_str}. Logging will be disabled.", file=sys.stderr)
        return

    log_file_str = os.getenv("LLM_CONVERSATION_LOG_FILE")
    # Create a logger instance
    logger = logging.getLogger("llm_conversation")
    logger.setLevel(log_level)

    log_file = Path(log_file_str) if log_file_str else None

    # Console output handler: prefer Rich if available, otherwise fall back to a StreamHandler.
    if _HAS_RICH and RichHandler is not None and Console is not None:
        rich_handler = RichHandler(
            console=Console(stderr=True),
            show_time=True,
            show_level=True,
            show_path=True,
            markup=True,
            rich_tracebacks=True,
        )
        logger.addHandler(rich_handler)
    else:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(
            logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(stream_handler)

    # File handler (if log file is specified)
    if log_file:
        if log_file.is_dir():
            raise ValueError(f"Log file path {log_file} is a directory, not a file.")

        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.

    Args:
        name: Name for the logger, typically __name__

    Returns:
        Logger instance
    """
    # Prevent a "No handlers could be found" warning and ensure the program doesn't output logs unless configured.
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.addHandler(logging.NullHandler())

    return logger
