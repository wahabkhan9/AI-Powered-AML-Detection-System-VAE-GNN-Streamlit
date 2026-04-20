"""
utils/io_utils.py
=================
File I/O helpers: safe parquet read/write, model checkpointing,
JSON utilities, and data validation helpers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parquet
# ---------------------------------------------------------------------------
def safe_read_parquet(path: Path, **kwargs) -> pd.DataFrame:
    """Read a parquet file; return empty DataFrame if not found."""
    if not path.exists():
        log.warning("File not found: %s – returning empty DataFrame.", path)
        return pd.DataFrame()
    return pd.read_parquet(path, **kwargs)


def safe_write_parquet(df: pd.DataFrame, path: Path, **kwargs) -> None:
    """Write a DataFrame to parquet, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="snappy", **kwargs)
    log.debug("Wrote %d rows → %s", len(df), path)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------
def load_json(path: Path, default: Optional[Any] = None) -> Any:
    if not path.exists():
        log.warning("JSON not found: %s", path)
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, default=str)
    log.debug("JSON saved → %s", path)


# ---------------------------------------------------------------------------
# PyTorch model checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(
    model: torch.nn.Module,
    path: Path,
    extra: Optional[Dict] = None,
) -> None:
    """Save model state dict with optional metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict = {"state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    log.info("Checkpoint saved → %s", path)


def load_checkpoint(
    model: torch.nn.Module,
    path: Path,
    device: Optional[torch.device] = None,
) -> Dict:
    """Load state dict into model; return extra metadata dict."""
    device = device or torch.device("cpu")
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["state_dict"])
    log.info("Checkpoint loaded ← %s", path)
    return {k: v for k, v in payload.items() if k != "state_dict"}


# ---------------------------------------------------------------------------
# Pickle
# ---------------------------------------------------------------------------
def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.debug("Pickle saved → %s", path)


def load_pickle(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Pickle not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# File integrity
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    """Return hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# DataFrame validation
# ---------------------------------------------------------------------------
def assert_required_columns(df: pd.DataFrame, required: list[str], name: str = "DataFrame") -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
