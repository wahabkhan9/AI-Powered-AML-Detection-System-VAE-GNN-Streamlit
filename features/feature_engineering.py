"""
features/feature_engineering.py
================================
Centralised feature engineering pipeline.

Provides a sklearn-compatible transformer and standalone helper functions
used by both the training agents and the inference API.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRANSACTION_FEATURES = [
    "amount_usd",
    "log_amount",
    "hour_of_day",
    "day_of_week",
    "is_cross_border",
    "round_amount",
    "rapid_movement",
    "structuring_flag",
    "amount_z_score_sender",
]

JURISDICTION_RISK_MAP = {"low": 0.0, "medium": 0.5, "high": 1.0}
SAR_THRESHOLD = 10_000.0


# ---------------------------------------------------------------------------
# Raw feature derivation
# ---------------------------------------------------------------------------
def derive_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive ML features from a raw transaction DataFrame.
    Input columns required: timestamp, amount_usd, sender_id, receiver_id,
                            country_origin, country_dest, round_amount,
                            rapid_movement, structuring_flag, is_suspicious.
    """
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["hour_of_day"] = out["timestamp"].dt.hour / 23.0
    out["day_of_week"] = out["timestamp"].dt.dayofweek / 6.0
    out["is_cross_border"] = (out["country_origin"] != out["country_dest"]).astype(float)
    out["log_amount"] = np.log1p(out["amount_usd"])
    out["round_amount"] = out["round_amount"].astype(float)
    out["rapid_movement"] = out["rapid_movement"].astype(float)
    out["structuring_flag"] = out["structuring_flag"].astype(float)

    # Per-sender z-score
    sender_stats = out.groupby("sender_id")["amount_usd"].agg(["mean", "std"])
    out = out.join(sender_stats, on="sender_id", rsuffix="_sender")
    out["amount_z_score_sender"] = (
        (out["amount_usd"] - out["mean"]) / out["std"].clip(lower=1e-6)
    ).clip(-5.0, 5.0)

    return out[TRANSACTION_FEATURES].fillna(0.0)


def derive_node_features(
    txn_df: pd.DataFrame,
    cust_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build node-level features for GNN training / inference.
    """
    sent = txn_df.groupby("sender_id").agg(
        sent_count=("amount_usd", "count"),
        sent_total=("amount_usd", "sum"),
        sent_unique=("receiver_id", "nunique"),
    )
    recv = txn_df.groupby("receiver_id").agg(
        recv_count=("amount_usd", "count"),
        recv_total=("amount_usd", "sum"),
        recv_unique=("sender_id", "nunique"),
    )
    feat = cust_df.set_index("customer_id")[["risk_score", "jurisdiction_risk"]].copy()
    feat["jurisdiction_risk"] = feat["jurisdiction_risk"].map(
        {"low": 0, "medium": 1, "high": 2}
    ).fillna(1)
    feat = feat.join(sent, how="left").join(recv, how="left").fillna(0.0)
    feat["degree"] = feat["sent_count"] + feat["recv_count"]
    feat["total_amount"] = np.log1p(feat.get("sent_total", 0) + feat.get("recv_total", 0))
    feat["unique_counterparties"] = feat.get("sent_unique", 0) + feat.get("recv_unique", 0)
    feat["txn_count"] = feat["sent_count"] + feat["recv_count"]
    feat["avg_txn_amount"] = (
        (feat.get("sent_total", 0) + feat.get("recv_total", 0))
        / feat["txn_count"].clip(lower=1)
    )
    default_cols = [
        "degree", "total_amount", "unique_counterparties",
        "risk_score", "jurisdiction_risk", "avg_txn_amount", "txn_count",
    ]
    cols = feature_cols or default_cols
    return feat[cols].fillna(0.0)


# ---------------------------------------------------------------------------
# sklearn transformer
# ---------------------------------------------------------------------------
class TransactionFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer: raw transaction DataFrame → scaled array.

    Parameters
    ----------
    feature_cols : list of feature column names to include
    """

    def __init__(self, feature_cols: Optional[List[str]] = None) -> None:
        self.feature_cols = feature_cols or TRANSACTION_FEATURES
        self._scaler = StandardScaler()

    def fit(self, X: pd.DataFrame, y=None) -> "TransactionFeatureTransformer":
        derived = derive_transaction_features(X)
        self._scaler.fit(derived[self.feature_cols].values)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        derived = derive_transaction_features(X)
        return self._scaler.transform(derived[self.feature_cols].values).astype(np.float32)

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    @property
    def scaler(self) -> StandardScaler:
        return self._scaler


# ---------------------------------------------------------------------------
# Convenience pipeline factory
# ---------------------------------------------------------------------------
def build_transaction_pipeline() -> Pipeline:
    """Return a ready-to-fit sklearn pipeline for transaction features."""
    return Pipeline([("features", TransactionFeatureTransformer())])


# ---------------------------------------------------------------------------
# Anomaly score post-processing
# ---------------------------------------------------------------------------
def compute_alert_level(score: float, threshold: float) -> str:
    """Map a continuous anomaly score to a human-readable alert level."""
    if score >= threshold * 3.0:
        return "CRITICAL"
    if score >= threshold * 1.5:
        return "HIGH"
    if score >= threshold:
        return "MEDIUM"
    return "LOW"
