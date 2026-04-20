"""
tests/test_features.py
======================
Tests for the feature engineering pipeline.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
from datetime import datetime

from features.feature_engineering import (
    derive_transaction_features,
    TransactionFeatureTransformer,
    compute_alert_level,
    TRANSACTION_FEATURES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_txn_df() -> pd.DataFrame:
    n = 200
    rng = np.random.default_rng(42)
    countries = ["US", "UK", "PA", "MX", "DE"]
    return pd.DataFrame({
        "transaction_id": [f"TXN_{i:06d}" for i in range(n)],
        "timestamp": pd.date_range("2023-01-01", periods=n, freq="1h"),
        "sender_id": [f"CUST_{rng.integers(0, 50):04d}" for _ in range(n)],
        "receiver_id": [f"CUST_{rng.integers(0, 50):04d}" for _ in range(n)],
        "amount_usd": rng.lognormal(mean=7.5, sigma=1.2, size=n),
        "transaction_type": rng.choice(["WIRE_TRANSFER", "ACH", "CASH_DEPOSIT"], n),
        "country_origin": rng.choice(countries, n),
        "country_dest": rng.choice(countries, n),
        "round_amount": rng.choice([True, False], n),
        "rapid_movement": rng.choice([True, False], n),
        "structuring_flag": rng.choice([True, False], n),
        "is_suspicious": rng.choice([True, False], n, p=[0.05, 0.95]),
        "label": rng.choice(["normal", "structuring", "layering"], n),
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestDeriveTransactionFeatures:
    def test_output_columns(self, sample_txn_df):
        result = derive_transaction_features(sample_txn_df)
        for col in TRANSACTION_FEATURES:
            assert col in result.columns, f"Missing feature: {col}"

    def test_no_nulls(self, sample_txn_df):
        result = derive_transaction_features(sample_txn_df)
        assert result.isnull().sum().sum() == 0, "Feature matrix should have no NaNs"

    def test_hour_of_day_range(self, sample_txn_df):
        result = derive_transaction_features(sample_txn_df)
        assert result["hour_of_day"].between(0, 1).all()

    def test_day_of_week_range(self, sample_txn_df):
        result = derive_transaction_features(sample_txn_df)
        assert result["day_of_week"].between(0, 1).all()

    def test_log_amount_positive(self, sample_txn_df):
        result = derive_transaction_features(sample_txn_df)
        assert (result["log_amount"] >= 0).all()

    def test_z_score_clipped(self, sample_txn_df):
        result = derive_transaction_features(sample_txn_df)
        assert result["amount_z_score_sender"].between(-5, 5).all()

    def test_cross_border_binary(self, sample_txn_df):
        result = derive_transaction_features(sample_txn_df)
        assert set(result["is_cross_border"].unique()).issubset({0.0, 1.0})


class TestTransactionFeatureTransformer:
    def test_fit_transform_shape(self, sample_txn_df):
        transformer = TransactionFeatureTransformer()
        X = transformer.fit_transform(sample_txn_df)
        assert X.shape == (len(sample_txn_df), len(TRANSACTION_FEATURES))

    def test_transform_dtype(self, sample_txn_df):
        transformer = TransactionFeatureTransformer()
        X = transformer.fit_transform(sample_txn_df)
        assert X.dtype == np.float32

    def test_scaler_mean_near_zero(self, sample_txn_df):
        transformer = TransactionFeatureTransformer()
        X = transformer.fit_transform(sample_txn_df)
        assert np.abs(X.mean(axis=0)).max() < 0.5

    def test_transform_single_row(self, sample_txn_df):
        transformer = TransactionFeatureTransformer()
        transformer.fit(sample_txn_df)
        single = sample_txn_df.iloc[:1]
        X = transformer.transform(single)
        assert X.shape == (1, len(TRANSACTION_FEATURES))


class TestComputeAlertLevel:
    def test_critical(self):
        assert compute_alert_level(0.9, 0.1) == "CRITICAL"

    def test_high(self):
        assert compute_alert_level(0.15, 0.1) == "HIGH"

    def test_medium(self):
        assert compute_alert_level(0.11, 0.1) == "MEDIUM"

    def test_low(self):
        assert compute_alert_level(0.05, 0.1) == "LOW"
