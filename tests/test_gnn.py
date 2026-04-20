"""
tests/test_gnn.py
=================
Unit tests for the CustomerRiskGNN model.
"""

from __future__ import annotations

import pytest
import torch
from models.gnn import CustomerRiskGNN


@pytest.fixture(scope="module")
def small_graph():
    """Minimal graph: 10 nodes, fully-connected with self-loops."""
    n, f = 10, 7
    x = torch.randn(n, f)
    src = torch.arange(n).repeat_interleave(n)
    dst = torch.arange(n).repeat(n)
    edge_index = torch.stack([src, dst], dim=0)
    return x, edge_index


class TestCustomerRiskGNN:
    def test_forward_shape(self, small_graph):
        x, edge_index = small_graph
        model = CustomerRiskGNN(in_channels=7, hidden_channels=16, out_channels=2, heads=2)
        logits = model(x, edge_index)
        assert logits.shape == (10, 2), "Logits should be [N, 2]"

    def test_predict_proba_sums_to_one(self, small_graph):
        x, edge_index = small_graph
        model = CustomerRiskGNN(in_channels=7, hidden_channels=16, out_channels=2, heads=2)
        proba = model.predict_proba(x, edge_index)
        assert torch.allclose(proba.sum(dim=1), torch.ones(10), atol=1e-5)

    def test_predict_binary(self, small_graph):
        x, edge_index = small_graph
        model = CustomerRiskGNN(in_channels=7, hidden_channels=16, out_channels=2, heads=2)
        preds = model.predict(x, edge_index)
        assert preds.shape == (10,)
        assert set(preds.tolist()).issubset({0, 1})

    def test_gradient_flows(self, small_graph):
        x, edge_index = small_graph
        model = CustomerRiskGNN(in_channels=7, hidden_channels=16, out_channels=2, heads=2)
        model.train()
        logits = model(x, edge_index)
        loss = logits.sum()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"

    def test_dropout_off_at_eval(self, small_graph):
        x, edge_index = small_graph
        model = CustomerRiskGNN(in_channels=7, hidden_channels=16, out_channels=2, heads=2, dropout=0.9)
        model.eval()
        with torch.no_grad():
            p1 = model.predict_proba(x, edge_index)
            p2 = model.predict_proba(x, edge_index)
        assert torch.allclose(p1, p2), "Predictions should be deterministic at eval"
