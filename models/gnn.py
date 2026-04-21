"""
models/gnn.py
=============
Graph Neural Network (GNN) for customer risk classification.

Architecture
------------
Two-layer Graph Attention Network (GATv2) followed by a linear classifier.
Node features: degree, total_amount, unique_counterparties, risk_score,
               jurisdiction_risk_encoded, avg_txn_amount, txn_count.

The graph is a directed transaction graph where each node is a customer and
each edge represents one or more transactions between them.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import torch-geometric; provide a CPU-only fallback message.
# ---------------------------------------------------------------------------
try:
    from torch_geometric.nn import GATv2Conv, global_mean_pool
    from torch_geometric.data import Data, DataLoader as GeoDataLoader
    _HAS_PYGEOMETRIC = True
except ImportError:
    _HAS_PYGEOMETRIC = False
    log.warning(
        "torch-geometric not installed. Using manual message-passing fallback. "
        "Install torch-geometric for full GATv2 support."
    )


# ---------------------------------------------------------------------------
# Manual sparse message-passing fallback (no torch-geometric dependency)
# ---------------------------------------------------------------------------
class SimpleGraphConv(nn.Module):
    """
    Minimal mean-aggregation graph convolution.
    h_i^(l+1) = σ( W · MEAN({h_j : j ∈ N(i) ∪ {i}}) )
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        n = x.size(0)
        # Aggregate neighbors
        src, dst = edge_index[0], edge_index[1]
        agg = torch.zeros_like(x)
        count = torch.ones(n, 1, device=x.device)
        agg.index_add_(0, dst, x[src])
        neighbor_count = torch.zeros(n, 1, device=x.device)
        neighbor_count.index_add_(0, dst, torch.ones(dst.size(0), 1, device=x.device))
        neighbor_count = neighbor_count.clamp(min=1)
        # Self + neighbour mean
        agg = (x + agg) / (count + neighbor_count)
        return F.leaky_relu(self.linear(agg), negative_slope=0.2)


# ---------------------------------------------------------------------------
# GNN model (GATv2 if torch-geometric available else SimpleGraphConv)
# ---------------------------------------------------------------------------
class CustomerRiskGNN(nn.Module):
    """
    Two-layer GNN for binary node classification (suspicious / normal).

    Parameters
    ----------
    in_channels   : number of input node features
    hidden_channels : width of hidden GNN layers
    out_channels  : 2 (binary classification)
    heads         : number of attention heads per GATv2 layer
    dropout       : dropout probability on node embeddings
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        if _HAS_PYGEOMETRIC:
            self.conv1 = GATv2Conv(
                in_channels, hidden_channels, heads=heads,
                concat=True, dropout=dropout, add_self_loops=True,
            )
            self.conv2 = GATv2Conv(
                hidden_channels * heads, hidden_channels, heads=1,
                concat=False, dropout=dropout, add_self_loops=True,
            )
        else:
            self.conv1 = SimpleGraphConv(in_channels, hidden_channels)
            self.conv2 = SimpleGraphConv(hidden_channels, hidden_channels)

        self.bn1 = nn.BatchNorm1d(hidden_channels * (heads if _HAS_PYGEOMETRIC else 1))
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )
        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x           : [N, in_channels] node feature matrix
        edge_index  : [2, E] COO edge list
        edge_weight : [E] optional edge weights (ignored by fallback)

        Returns
        -------
        logits : [N, 2] unnormalised class scores
        """
        # Layer 1
        if _HAS_PYGEOMETRIC:
            h = self.conv1(x, edge_index)
        else:
            h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.leaky_relu(h, negative_slope=0.2)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 2
        if _HAS_PYGEOMETRIC:
            h = self.conv2(h, edge_index)
        else:
            h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.leaky_relu(h, negative_slope=0.2)
        h = F.dropout(h, p=self.dropout, training=self.training)

        return self.classifier(h)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return softmax probability vector [N, 2]."""
        self.eval()
        logits = self.forward(x, edge_index)
        return F.softmax(logits, dim=1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Return binary predictions [N]."""
        proba = self.predict_proba(x, edge_index)
        return (proba[:, 1] >= threshold).long()
