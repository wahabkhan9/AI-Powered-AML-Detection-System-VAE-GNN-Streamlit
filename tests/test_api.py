"""
tests/test_api.py
=================
FastAPI endpoint integration tests (uses TestClient, no real models needed).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

# Patch _state to stub mode before importing app
import sys
from unittest.mock import patch


@pytest.fixture(scope="module")
def client():
    # Import inside fixture so patch takes effect
    from api.main import app, _state
    _state["stub"] = True
    _state["request_count"] = 0
    _state["alert_count"] = 0
    import time
    _state["start_time"] = time.time()
    with TestClient(app) as c:
        yield c


SAMPLE_TXN = {
    "transaction_id": "TXN_TEST_001",
    "timestamp": "2024-06-15T10:30:00",
    "sender_id": "CUST_000001",
    "receiver_id": "CUST_000002",
    "amount_usd": 9500.0,
    "transaction_type": "WIRE_TRANSFER",
    "country_origin": "US",
    "country_dest": "PA",
    "round_amount": False,
    "rapid_movement": False,
    "structuring_flag": True,
}


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        r = client.get("/api/v1/health")
        assert r.status_code == 200

    def test_health_has_status_field(self, client):
        r = client.get("/api/v1/health")
        assert r.json()["status"] == "healthy"


class TestScoreTransaction:
    def test_score_returns_200(self, client):
        r = client.post("/api/v1/score/transaction", json=SAMPLE_TXN)
        assert r.status_code == 200

    def test_score_response_fields(self, client):
        r = client.post("/api/v1/score/transaction", json=SAMPLE_TXN)
        data = r.json()
        for field in ["transaction_id", "anomaly_score", "threshold", "is_flagged", "alert_level"]:
            assert field in data, f"Missing field: {field}"

    def test_score_transaction_id_matches(self, client):
        r = client.post("/api/v1/score/transaction", json=SAMPLE_TXN)
        assert r.json()["transaction_id"] == SAMPLE_TXN["transaction_id"]

    def test_negative_amount_rejected(self, client):
        bad = {**SAMPLE_TXN, "amount_usd": -100.0}
        r = client.post("/api/v1/score/transaction", json=bad)
        assert r.status_code == 422


class TestBatchScore:
    def test_batch_returns_200(self, client):
        batch = {"transactions": [SAMPLE_TXN] * 5}
        r = client.post("/api/v1/score/batch", json=batch)
        assert r.status_code == 200

    def test_batch_count(self, client):
        batch = {"transactions": [SAMPLE_TXN] * 10}
        r = client.post("/api/v1/score/batch", json=batch)
        assert r.json()["total"] == 10


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        r = client.get("/api/v1/metrics")
        assert r.status_code == 200

    def test_metrics_contains_gauge(self, client):
        r = client.get("/api/v1/metrics")
        assert "aml_requests_total" in r.text


class TestAlertsEndpoint:
    def test_alerts_returns_200(self, client):
        r = client.get("/api/v1/alerts")
        assert r.status_code == 200

    def test_alerts_has_page_field(self, client):
        r = client.get("/api/v1/alerts")
        assert "page" in r.json()
