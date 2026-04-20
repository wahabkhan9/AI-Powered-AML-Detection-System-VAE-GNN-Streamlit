"""
gnn/graph_builder.py
====================
Transaction graph construction utilities.

Provides functions to build NetworkX and PyTorch-compatible graph
representations from raw transaction data, including:
  - Customer-level directed graphs (node = customer, edge = transaction)
  - Community detection helpers
  - Subgraph extraction for visualisation
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False
    log.warning("networkx not installed; graph visualisation disabled.")


# ---------------------------------------------------------------------------
# Core graph builder
# ---------------------------------------------------------------------------
def build_transaction_graph(
    txn_df: pd.DataFrame,
    max_edges: int = 500_000,
) -> "nx.DiGraph":
    """
    Build a directed transaction graph.

    Parameters
    ----------
    txn_df    : transaction DataFrame (sender_id, receiver_id, amount_usd, ...)
    max_edges : cap to avoid memory blow-up in visualisation contexts

    Returns
    -------
    nx.DiGraph with node attr 'is_suspicious' and edge attrs
    'weight', 'amount_usd', 'transaction_type'.
    """
    if not _HAS_NX:
        raise ImportError("networkx is required for graph_builder.")

    sample = txn_df.head(max_edges)
    G = nx.DiGraph()

    for _, row in sample.iterrows():
        src = row["sender_id"]
        dst = row["receiver_id"]
        if not G.has_node(src):
            G.add_node(src, is_suspicious=bool(row.get("is_suspicious", False)))
        if not G.has_node(dst):
            G.add_node(dst, is_suspicious=bool(row.get("is_suspicious", False)))
        if G.has_edge(src, dst):
            G[src][dst]["weight"] += 1
            G[src][dst]["amount_usd"] += float(row["amount_usd"])
        else:
            G.add_edge(
                src, dst,
                weight=1,
                amount_usd=float(row["amount_usd"]),
                transaction_type=row.get("transaction_type", "UNKNOWN"),
            )

    log.info(
        "Graph built: %d nodes, %d edges",
        G.number_of_nodes(), G.number_of_edges(),
    )
    return G


# ---------------------------------------------------------------------------
# Subgraph extraction for visualisation
# ---------------------------------------------------------------------------
def extract_ego_subgraph(
    G: "nx.DiGraph",
    node_id: str,
    radius: int = 2,
    max_nodes: int = 150,
) -> "nx.DiGraph":
    """Return ego-centric subgraph around `node_id` up to `radius` hops."""
    if not G.has_node(node_id):
        log.warning("Node %s not in graph.", node_id)
        return nx.DiGraph()
    ego = nx.ego_graph(G, node_id, radius=radius, undirected=True)
    if ego.number_of_nodes() > max_nodes:
        # Keep the highest-degree neighbours
        sorted_nodes = sorted(
            ego.nodes, key=lambda n: G.degree(n), reverse=True
        )[:max_nodes]
        ego = ego.subgraph(sorted_nodes).copy()
    return ego


# ---------------------------------------------------------------------------
# Community detection
# ---------------------------------------------------------------------------
def detect_communities(
    G: "nx.DiGraph",
    algorithm: str = "louvain",
) -> Dict[str, int]:
    """
    Detect communities and return a dict {node_id: community_label}.

    Parameters
    ----------
    algorithm : 'louvain' (requires python-louvain) or 'greedy_modularity'
    """
    UG = G.to_undirected()
    if algorithm == "louvain":
        try:
            import community as community_louvain  # python-louvain
            partition = community_louvain.best_partition(UG)
            return partition
        except ImportError:
            log.warning("python-louvain not installed; falling back to greedy_modularity.")

    communities = nx.community.greedy_modularity_communities(UG)
    partition = {}
    for i, comm in enumerate(communities):
        for node in comm:
            partition[node] = i
    return partition


# ---------------------------------------------------------------------------
# Graph-level statistics
# ---------------------------------------------------------------------------
def compute_graph_stats(G: "nx.DiGraph") -> Dict:
    """Compute summary statistics for the transaction graph."""
    UG = G.to_undirected()
    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_clustering": nx.average_clustering(UG),
        "num_weakly_connected_components": nx.number_weakly_connected_components(G),
        "num_strongly_connected_components": nx.number_strongly_connected_components(G),
    }


# ---------------------------------------------------------------------------
# Convert to torch COO for GNN training (lightweight)
# ---------------------------------------------------------------------------
def to_coo(
    txn_df: pd.DataFrame,
    id_map: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert transaction DataFrame to COO edge arrays.

    Returns
    -------
    src : [E] int array
    dst : [E] int array
    """
    valid = txn_df[
        txn_df["sender_id"].isin(id_map) & txn_df["receiver_id"].isin(id_map)
    ]
    src = valid["sender_id"].map(id_map).values.astype(np.int64)
    dst = valid["receiver_id"].map(id_map).values.astype(np.int64)
    return src, dst
