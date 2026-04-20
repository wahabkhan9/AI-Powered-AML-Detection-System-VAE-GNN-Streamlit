"""
utils/logger.py
===============
Centralised logging configuration for the AML system.

Usage
-----
    from utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Hello from %s", __name__)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


LOG_DIR = Path("logs")
_configured = False


def configure_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False,
) -> None:
    """
    Configure root logger once for the entire process.

    Parameters
    ----------
    level      : logging level string ("DEBUG", "INFO", "WARNING", "ERROR")
    log_file   : optional file path to write logs to
    json_format: emit JSON-structured log lines (for log aggregators)
    """
    global _configured
    if _configured:
        return
    _configured = True

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    if json_format:
        fmt = (
            '{"time": "%(asctime)s", "level": "%(levelname)s", '
            '"name": "%(name)s", "msg": %(message)r}'
        )
    else:
        fmt = "%(asctime)s [%(levelname)-8s] %(name)-30s – %(message)s"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(LOG_DIR / log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(fmt))
        handlers.append(file_handler)

    logging.basicConfig(
        level=numeric_level,
        format=fmt,
        handlers=handlers,
        force=True,
    )

    # Silence noisy third-party loggers
    for lib in ["urllib3", "botocore", "boto3", "s3transfer", "filelock"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Return a named logger, optionally overriding its level."""
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger
