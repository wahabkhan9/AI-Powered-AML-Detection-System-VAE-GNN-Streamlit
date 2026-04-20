"""gnn package – graph construction and GNN training utilities."""
from .graph_builder import build_transaction_graph, extract_ego_subgraph, detect_communities

__all__ = ["build_transaction_graph", "extract_ego_subgraph", "detect_communities"]
