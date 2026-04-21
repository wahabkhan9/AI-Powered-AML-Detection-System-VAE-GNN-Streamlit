"""
tests/test_orchestrator.py
==========================
Tests for the multi-agent OrchestratorAgent (Commander).
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from agents.orchestrator_agent import (
    analyst_node,
    detective_node,
    commander_node,
    OrchestratorAgent,
    AMLAgentState,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
SAMPLE_TRANSACTIONS = [
    {
        "transaction_id": f"TXN_{i:04d}",
        "timestamp": "2024-06-15T10:30:00",
        "sender_id": f"CUST_{i % 5:04d}",
        "receiver_id": f"CUST_{(i + 1) % 5:04d}",
        "amount_usd": float(5000 + i * 100),
        "transaction_type": "WIRE_TRANSFER",
        "country_origin": "US",
        "country_dest": "PA",
        "round_amount": False,
        "rapid_movement": False,
        "structuring_flag": i % 3 == 0,
        "is_suspicious": False,
        "label": "normal",
    }
    for i in range(20)
]


def _empty_state() -> AMLAgentState:
    return {
        "transactions": SAMPLE_TRANSACTIONS,
        "vae_scores": [],
        "gan_scores": [],
        "combined_anomaly_scores": [],
        "flagged_transaction_ids": [],
        "customer_ids": [],
        "gnn_risk_scores": {},
        "suspicious_clusters": [],
        "network_summary": "",
        "sar_narratives": [],
        "sar_ids": [],
        "final_risk_level": "UNKNOWN",
        "final_risk_score": 0.0,
        "action_recommendation": "",
        "processing_complete": False,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestAnalystNode:
    def test_returns_vae_scores(self):
        """Analyst should populate vae_scores with one score per transaction."""
        state = _empty_state()
        with patch("agents.orchestrator_agent._registry") as mock_reg:
            mock_reg.load = MagicMock()
            mock_reg.vae_threshold = 0.005
            n = len(SAMPLE_TRANSACTIONS)
            mock_reg.score_transactions.return_value = {
                "vae": np.random.uniform(0, 0.01, n).astype(np.float32),
                "gan": np.random.uniform(0, 1, n).astype(np.float32),
            }
            result = analyst_node(state)
        assert len(result["vae_scores"]) == n
        assert len(result["gan_scores"]) == n
        assert len(result["combined_anomaly_scores"]) == n

    def test_flagged_ids_subset_of_transactions(self):
        state = _empty_state()
        with patch("agents.orchestrator_agent._registry") as mock_reg:
            mock_reg.load = MagicMock()
            mock_reg.vae_threshold = 0.005
            n = len(SAMPLE_TRANSACTIONS)
            mock_reg.score_transactions.return_value = {
                "vae": np.full(n, 0.01, dtype=np.float32),  # all above threshold
                "gan": np.zeros(n, dtype=np.float32),
            }
            result = analyst_node(state)
        all_ids = {t["transaction_id"] for t in SAMPLE_TRANSACTIONS}
        assert all(fid in all_ids for fid in result["flagged_transaction_ids"])


class TestDetectiveNode:
    def test_network_summary_populated(self):
        state = _empty_state()
        state["flagged_transaction_ids"] = [SAMPLE_TRANSACTIONS[0]["transaction_id"]]
        with patch("agents.orchestrator_agent._registry") as mock_reg:
            mock_reg.load = MagicMock()
            mock_reg.risk_df = __import__("pandas").DataFrame()
            mock_reg.get_gnn_risk.return_value = 0.75
            result = detective_node(state)
        assert len(result["network_summary"]) > 0
        assert "gnn_risk_scores" in result

    def test_no_flagged_empty_clusters(self):
        state = _empty_state()
        state["flagged_transaction_ids"] = []
        with patch("agents.orchestrator_agent._registry") as mock_reg:
            mock_reg.load = MagicMock()
            mock_reg.risk_df = __import__("pandas").DataFrame()
            mock_reg.get_gnn_risk.return_value = 0.0
            result = detective_node(state)
        assert result["suspicious_clusters"] == []


class TestCommanderNode:
    def test_risk_levels_valid(self):
        state = _empty_state()
        for score_override in [0.0, 0.3, 0.6, 0.9]:
            state["combined_anomaly_scores"] = [score_override] * 20
            state["gnn_risk_scores"] = {"CUST_0001": score_override}
            state["flagged_transaction_ids"] = SAMPLE_TRANSACTIONS[:int(20 * score_override)]
            result = commander_node(state)
            assert result["final_risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

    def test_processing_complete_flag(self):
        state = _empty_state()
        state["combined_anomaly_scores"] = [0.1] * 20
        state["gnn_risk_scores"] = {}
        state["flagged_transaction_ids"] = []
        result = commander_node(state)
        assert result["processing_complete"] is True

    def test_action_recommendation_nonempty(self):
        state = _empty_state()
        state["combined_anomaly_scores"] = [0.5] * 20
        state["gnn_risk_scores"] = {"CUST_0001": 0.8}
        state["flagged_transaction_ids"] = [t["transaction_id"] for t in SAMPLE_TRANSACTIONS[:10]]
        result = commander_node(state)
        assert len(result["action_recommendation"]) > 10


class TestOrchestratorAgent:
    def test_process_returns_complete_state(self):
        """Integration test: full pipeline should return a complete state."""
        agent = OrchestratorAgent()
        with patch("agents.orchestrator_agent._registry") as mock_reg:
            import pandas as pd
            n = len(SAMPLE_TRANSACTIONS)
            mock_reg.load = MagicMock()
            mock_reg.vae_threshold = 0.005
            mock_reg.risk_df = pd.DataFrame()
            mock_reg.score_transactions.return_value = {
                "vae": np.random.uniform(0, 0.01, n).astype(np.float32),
                "gan": np.zeros(n, dtype=np.float32),
            }
            mock_reg.get_gnn_risk.return_value = 0.3

            # Patch narrator to avoid LLM call
            with patch("agents.orchestrator_agent.narrator_node", side_effect=lambda s: {**s, "sar_narratives": [], "sar_ids": []}):
                result = agent.process(SAMPLE_TRANSACTIONS)

        assert result["processing_complete"] is True
        assert result["final_risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
        assert isinstance(result["final_risk_score"], float)