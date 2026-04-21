"""
tests/test_causal.py
====================
Unit tests for causal inference and what-if analysis module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta


@pytest.fixture(scope="module")
def sample_txn_df() -> pd.DataFrame:
    """Generate a minimal transaction DataFrame for causal tests."""
    rng = np.random.default_rng(42)
    n = 2000
    base_date = datetime(2023, 1, 1)
    return pd.DataFrame({
        "transaction_id": [f"TXN_{i}" for i in range(n)],
        "timestamp": [base_date + timedelta(days=int(rng.integers(0, 365))) for _ in range(n)],
        "sender_id": [f"CUST_{rng.integers(0, 100)}" for _ in range(n)],
        "receiver_id": [f"CUST_{rng.integers(0, 100)}" for _ in range(n)],
        "amount_usd": rng.lognormal(7.5, 1.2, n),
        "is_suspicious": rng.choice([True, False], n, p=[0.05, 0.95]),
        "structuring_flag": rng.choice([True, False], n, p=[0.1, 0.9]),
        "risk_score": rng.uniform(0, 1, n),
        "label": "normal",
    })


class TestAMLCausalAnalyzer:
    def test_what_if_threshold_lower(self, sample_txn_df):
        from causal.causal_inference import AMLCausalAnalyzer
        analyzer = AMLCausalAnalyzer(sample_txn_df)
        result = analyzer.what_if_threshold(0.01, 0.008)
        # Lower threshold → more alerts
        assert result.counterfactual_alerts >= result.baseline_alerts

    def test_what_if_threshold_higher(self, sample_txn_df):
        from causal.causal_inference import AMLCausalAnalyzer
        analyzer = AMLCausalAnalyzer(sample_txn_df)
        result = analyzer.what_if_threshold(0.01, 0.015)
        # Higher threshold → fewer alerts
        assert result.counterfactual_alerts <= result.baseline_alerts

    def test_what_if_threshold_fields(self, sample_txn_df):
        from causal.causal_inference import AMLCausalAnalyzer
        analyzer = AMLCausalAnalyzer(sample_txn_df)
        result = analyzer.what_if_threshold(0.01, 0.008)
        assert result.scenario != ""
        assert result.recommendation != ""
        assert isinstance(result.delta_pct, float)

    def test_what_if_rule(self, sample_txn_df):
        from causal.causal_inference import AMLCausalAnalyzer
        analyzer = AMLCausalAnalyzer(sample_txn_df)
        result = analyzer.what_if_rule("Test Rule", 0.15, 20.0)
        assert result.counterfactual_alerts >= result.baseline_alerts
        assert result.delta_alerts >= 0

    def test_did_causal_estimate(self, sample_txn_df):
        from causal.causal_inference import AMLCausalAnalyzer
        analyzer = AMLCausalAnalyzer(sample_txn_df)
        result = analyzer.estimate_rule_effect(
            rule_name="Test Policy",
            policy_date="2023-07-01",
            treated_column="structuring_flag",
        )
        assert result.estimator in ("DiD", "CausalForestDML")
        assert isinstance(result.ate, float)
        assert result.ate_lower <= result.ate_upper
        assert result.interpretation != ""


class TestWhatIfResult:
    def test_fields(self):
        from causal.causal_inference import WhatIfResult
        r = WhatIfResult(
            scenario="test",
            baseline_alerts=100,
            counterfactual_alerts=120,
            delta_alerts=20,
            delta_pct=20.0,
            baseline_flagged_amount=1_000_000.0,
            counterfactual_flagged_amount=1_200_000.0,
            recommendation="Increase monitoring.",
        )
        assert r.delta_alerts == 20
        assert r.delta_pct == 20.0
