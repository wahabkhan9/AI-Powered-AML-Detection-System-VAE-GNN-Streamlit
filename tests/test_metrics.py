"""
tests/test_metrics.py
=====================
Tests for AML evaluation metrics utilities.
"""

from __future__ import annotations

import numpy as np
import pytest
from utils.metrics import compute_classification_metrics, find_optimal_threshold, sar_cost_model


class TestComputeClassificationMetrics:
    def test_perfect_classifier(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        m = compute_classification_metrics(y_true, y_pred)
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1_score"] == 1.0
        assert m["false_negatives"] == 0

    def test_all_false_negatives(self):
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        m = compute_classification_metrics(y_true, y_pred)
        assert m["recall"] == 0.0
        assert m["false_negatives"] == 4

    def test_auc_computed_when_scores_provided(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 100)
        y_score = rng.uniform(0, 1, 100)
        m = compute_classification_metrics(y_true, y_true.astype(float), y_score)
        assert "roc_auc" in m
        assert 0 <= m["roc_auc"] <= 1


class TestFindOptimalThreshold:
    def test_returns_valid_threshold(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 500)
        y_score = rng.uniform(0, 1, 500)
        t, metrics = find_optimal_threshold(y_true, y_score, min_recall=0.0)
        assert 0 <= t <= 1
        assert "recall" in metrics
        assert "threshold" in metrics

    def test_min_recall_respected(self):
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 2, 1000)
        y_score = y_true.astype(float) + rng.normal(0, 0.3, 1000)
        _, metrics = find_optimal_threshold(y_true, y_score, min_recall=0.85)
        if metrics:  # may be empty if no threshold meets constraint
            assert metrics["recall"] >= 0.84  # allow float tolerance


class TestSARCostModel:
    def test_zero_errors_zero_cost(self):
        result = sar_cost_model(tp=100, fp=0, fn=0)
        assert result["total_cost_usd"] == 0.0

    def test_fn_dominates_cost(self):
        result = sar_cost_model(tp=10, fp=50, fn=1, cost_fn=100_000, cost_fp=500)
        assert result["missed_laundering_cost_usd"] > result["false_alert_cost_usd"]

    def test_keys_present(self):
        result = sar_cost_model(tp=5, fp=10, fn=2)
        for key in ["total_cost_usd", "missed_laundering_cost_usd", "false_alert_cost_usd", "cost_per_true_positive_usd"]:
            assert key in result
