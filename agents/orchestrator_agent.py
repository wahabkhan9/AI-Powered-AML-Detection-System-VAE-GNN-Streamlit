"""
agents/orchestrator_agent.py
=============================
The Commander – LangGraph-based multi-agent orchestrator.

Implements a stateful directed graph of agents:

  ┌──────────┐     ┌─────────────────┐     ┌────────────────────┐     ┌──────────────┐
  │  Input   │────►│  Analyst (VAE+  │────►│  Detective (GNN +  │────►│  Narrator    │
  │  Node    │     │  GAN Detector)  │     │  Graph Investigator│     │  (LLM SAR)   │
  └──────────┘     └─────────────────┘     └────────────────────┘     └──────┬───────┘
                                                                              │
                                                                    ┌─────────▼────────┐
                                                                    │  Commander Final │
                                                                    │  Risk Assessment │
                                                                    └──────────────────┘

If LangGraph is not installed, falls back to a sequential Python orchestrator
with identical state management.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
import pandas as pd
import torch

log = logging.getLogger(__name__)

MODEL_DIR  = Path("models")
REPORTS_DIR = Path("reports")
DATA_DIR   = Path("data")

# ---------------------------------------------------------------------------
# Try LangGraph import
# ---------------------------------------------------------------------------
try:
    from langgraph.graph import StateGraph, END
    _HAS_LANGGRAPH = True
    log.info("LangGraph available – using graph-based orchestration.")
except ImportError:
    _HAS_LANGGRAPH = False
    log.warning("LangGraph not installed – using sequential fallback orchestrator.")


# ---------------------------------------------------------------------------
# Shared state schema
# ---------------------------------------------------------------------------
class AMLAgentState(TypedDict):
    """Shared state passed between all agents in the graph."""
    # Input
    transactions: List[Dict]                  # raw transaction dicts

    # Analyst outputs
    vae_scores: List[float]
    gan_scores: List[float]
    combined_anomaly_scores: List[float]
    flagged_transaction_ids: List[str]

    # Detective outputs
    customer_ids: List[str]
    gnn_risk_scores: Dict[str, float]
    suspicious_clusters: List[List[str]]
    network_summary: str

    # Narrator outputs
    sar_narratives: List[str]
    sar_ids: List[str]

    # Commander outputs
    final_risk_level: str                    # LOW / MEDIUM / HIGH / CRITICAL
    final_risk_score: float
    action_recommendation: str
    processing_complete: bool


# ---------------------------------------------------------------------------
# Model loader (shared across agents)
# ---------------------------------------------------------------------------
class ModelRegistry:
    """Lazy-loads trained models once and caches them."""
    _instance: Optional["ModelRegistry"] = None

    def __new__(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load(self) -> None:
        if self._loaded:
            return

        from features.feature_engineering import derive_transaction_features
        from models.vae import VAE
        from models.gan import TransactionGAN

        self.derive_features = derive_transaction_features

        # ── VAE ──────────────────────────────────────────────────────────────
        vae_meta_path = MODEL_DIR / "vae_meta.json"
        if vae_meta_path.exists():
            meta = json.loads(vae_meta_path.read_text())
            self.vae = VAE(
                input_dim=meta["input_dim"],
                latent_dim=meta["latent_dim"],
                hidden_dims=tuple(meta["hidden_dims"]),
                beta=meta["beta"],
            )
            self.vae.load_state_dict(torch.load(MODEL_DIR / "vae_model.pth", map_location="cpu"))
            self.vae.eval()
            self.vae_threshold = meta["threshold"]
        else:
            self.vae = None
            self.vae_threshold = 0.01
            log.warning("VAE model not found – using stub scores.")

        # ── Scaler ───────────────────────────────────────────────────────────
        scaler_path = MODEL_DIR / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
        else:
            self.scaler = None

        # ── GAN ──────────────────────────────────────────────────────────────
        gan_meta_path = MODEL_DIR / "gan_meta.json"
        if gan_meta_path.exists():
            gan_meta = json.loads(gan_meta_path.read_text())
            self.gan = TransactionGAN(
                feature_dim=gan_meta["feature_dim"],
                latent_dim=gan_meta["latent_dim"],
            )
            self.gan.discriminator.load_state_dict(
                torch.load(MODEL_DIR / "gan_discriminator.pth", map_location="cpu")
            )
            self.gan.discriminator.eval()
        else:
            self.gan = None
            log.warning("GAN model not found – using stub scores.")

        # ── GNN risk scores (pre-computed) ───────────────────────────────────
        risk_path = REPORTS_DIR / "customer_risk_scores.parquet"
        self.risk_df = pd.read_parquet(risk_path) if risk_path.exists() else pd.DataFrame()

        self._loaded = True
        log.info("ModelRegistry loaded.")

    # ------------------------------------------------------------------
    def score_transactions(self, txn_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Run VAE + GAN anomaly scoring. Returns dict of score arrays."""
        if self.scaler is None or self.vae is None:
            n = len(txn_df)
            return {
                "vae": np.random.uniform(0, 0.02, n).astype(np.float32),
                "gan": np.random.uniform(0, 1, n).astype(np.float32),
            }

        feat = self.derive_features(txn_df)
        X = self.scaler.transform(feat.values).astype(np.float32)
        tensor = torch.tensor(X)

        with torch.no_grad():
            vae_scores = self.vae.anomaly_score(tensor).numpy()

        gan_scores = (
            self.gan.anomaly_score(tensor).numpy()
            if self.gan is not None
            else np.zeros(len(X), dtype=np.float32)
        )
        return {"vae": vae_scores, "gan": gan_scores}

    def get_gnn_risk(self, customer_id: str) -> float:
        if self.risk_df.empty or "customer_id" not in self.risk_df.columns:
            return 0.0
        row = self.risk_df[self.risk_df["customer_id"] == customer_id]
        return float(row["gnn_risk_score"].values[0]) if not row.empty else 0.0


_registry = ModelRegistry()


# ---------------------------------------------------------------------------
# Individual agent node functions
# ---------------------------------------------------------------------------
def analyst_node(state: AMLAgentState) -> AMLAgentState:
    """
    Analyst agent – scores every transaction with VAE + GAN,
    flags transactions above threshold.
    """
    log.info("[Analyst] Scoring %d transactions …", len(state["transactions"]))
    _registry.load()

    txn_df = pd.DataFrame(state["transactions"])
    txn_df["timestamp"] = pd.to_datetime(txn_df.get("timestamp", pd.Timestamp.now()))

    scores = _registry.score_transactions(txn_df)
    vae_scores  = scores["vae"].tolist()
    gan_scores  = scores["gan"].tolist()

    # Combined score: 60% VAE + 40% GAN (VAE more reliable when trained)
    vae_norm = np.array(vae_scores) / (np.array(vae_scores).max() + 1e-9)
    combined = (0.6 * vae_norm + 0.4 * np.array(gan_scores)).tolist()

    threshold = _registry.vae_threshold
    flagged_ids = [
        txn["transaction_id"]
        for txn, score in zip(state["transactions"], vae_scores)
        if score >= threshold
    ]

    log.info("[Analyst] %d/%d transactions flagged.", len(flagged_ids), len(state["transactions"]))

    return {
        **state,
        "vae_scores": vae_scores,
        "gan_scores": gan_scores,
        "combined_anomaly_scores": combined,
        "flagged_transaction_ids": flagged_ids,
    }


def detective_node(state: AMLAgentState) -> AMLAgentState:
    """
    Detective agent – retrieves GNN risk scores for involved customers,
    identifies high-risk clusters.
    """
    log.info("[Detective] Investigating %d flagged transactions …", len(state["flagged_transaction_ids"]))
    _registry.load()

    flagged_set = set(state["flagged_transaction_ids"])
    involved_customers: set[str] = set()
    for txn in state["transactions"]:
        if txn["transaction_id"] in flagged_set:
            involved_customers.add(txn["sender_id"])
            involved_customers.add(txn["receiver_id"])

    gnn_risk_scores: Dict[str, float] = {}
    for cid in involved_customers:
        gnn_risk_scores[cid] = _registry.get_gnn_risk(cid)

    # Simple clustering: group customers with GNN score > 0.6
    high_risk = [cid for cid, score in gnn_risk_scores.items() if score > 0.6]
    clusters = [high_risk] if high_risk else []

    total_flagged = sum(1 for txn in state["transactions"] if txn["transaction_id"] in flagged_set)
    total_amount = sum(
        float(txn["amount_usd"])
        for txn in state["transactions"]
        if txn["transaction_id"] in flagged_set
    )
    avg_gnn = float(np.mean(list(gnn_risk_scores.values()))) if gnn_risk_scores else 0.0

    network_summary = (
        f"Investigated {len(involved_customers)} customers. "
        f"Found {len(high_risk)} high-risk accounts (GNN > 0.6). "
        f"Total flagged amount: USD {total_amount:,.2f}. "
        f"Average GNN risk: {avg_gnn:.4f}."
    )
    log.info("[Detective] %s", network_summary)

    return {
        **state,
        "customer_ids": list(involved_customers),
        "gnn_risk_scores": gnn_risk_scores,
        "suspicious_clusters": clusters,
        "network_summary": network_summary,
    }


def narrator_node(state: AMLAgentState) -> AMLAgentState:
    """
    Narrator agent – generates SAR narratives using LLM (Ollama or OpenAI).
    Falls back to template-based generation if no LLM is available.
    """
    log.info("[Narrator] Drafting SAR narratives …")

    from llm.ollama_writer import OllamaSARWriter
    writer = OllamaSARWriter()

    narratives: list[str] = []
    sar_ids: list[str] = []
    import uuid

    flagged_set = set(state["flagged_transaction_ids"])
    flagged_txns = [t for t in state["transactions"] if t["transaction_id"] in flagged_set]

    # Group by sender
    from collections import defaultdict
    by_sender: Dict[str, list] = defaultdict(list)
    for txn in flagged_txns:
        by_sender[txn["sender_id"]].append(txn)

    for sender_id, txns in list(by_sender.items())[:10]:  # cap at 10 SARs per batch
        gnn_score = state["gnn_risk_scores"].get(sender_id, 0.0)
        vae_avg = float(np.mean([
            state["vae_scores"][i]
            for i, t in enumerate(state["transactions"])
            if t["transaction_id"] in {tx["transaction_id"] for tx in txns}
        ]))

        context = {
            "customer_id": sender_id,
            "transaction_count": len(txns),
            "total_amount_usd": sum(float(t["amount_usd"]) for t in txns),
            "gnn_risk_score": gnn_score,
            "vae_score": vae_avg,
            "network_summary": state["network_summary"],
            "transaction_types": list({t.get("transaction_type", "UNKNOWN") for t in txns}),
        }

        narrative = writer.generate(context)
        sar_id = "SAR-" + uuid.uuid4().hex[:8].upper()
        narratives.append(narrative)
        sar_ids.append(sar_id)

    log.info("[Narrator] Generated %d SAR narratives.", len(narratives))
    return {**state, "sar_narratives": narratives, "sar_ids": sar_ids}


def commander_node(state: AMLAgentState) -> AMLAgentState:
    """
    Commander agent – synthesises all findings into a final risk assessment
    and action recommendation.
    """
    log.info("[Commander] Producing final risk assessment …")

    n_flagged = len(state["flagged_transaction_ids"])
    n_total = len(state["transactions"])
    flag_rate = n_flagged / max(n_total, 1)

    avg_combined = float(np.mean(state["combined_anomaly_scores"])) if state["combined_anomaly_scores"] else 0.0
    avg_gnn = float(np.mean(list(state["gnn_risk_scores"].values()))) if state["gnn_risk_scores"] else 0.0

    final_score = round(0.5 * min(flag_rate * 10, 1.0) + 0.3 * avg_combined + 0.2 * avg_gnn, 4)

    if final_score >= 0.75:
        risk_level = "CRITICAL"
        action = "Immediate escalation to Compliance Officer. Freeze accounts pending investigation. File SAR within 30 days."
    elif final_score >= 0.50:
        risk_level = "HIGH"
        action = "Priority review by AML analyst. Enhanced due diligence required. SAR filing recommended."
    elif final_score >= 0.25:
        risk_level = "MEDIUM"
        action = "Schedule review within 5 business days. Document findings. Monitor account activity."
    else:
        risk_level = "LOW"
        action = "Standard monitoring. No immediate action required."

    log.info(
        "[Commander] Risk level=%s  Score=%.4f  Action: %s",
        risk_level, final_score, action,
    )

    return {
        **state,
        "final_risk_level": risk_level,
        "final_risk_score": final_score,
        "action_recommendation": action,
        "processing_complete": True,
    }


# ---------------------------------------------------------------------------
# Build the LangGraph graph (or sequential fallback)
# ---------------------------------------------------------------------------
def _build_langgraph_graph():
    """Construct the LangGraph StateGraph."""
    graph = StateGraph(AMLAgentState)
    graph.add_node("analyst",   analyst_node)
    graph.add_node("detective", detective_node)
    graph.add_node("narrator",  narrator_node)
    graph.add_node("commander", commander_node)

    graph.set_entry_point("analyst")
    graph.add_edge("analyst",   "detective")
    graph.add_edge("detective", "narrator")
    graph.add_edge("narrator",  "commander")
    graph.add_edge("commander", END)

    return graph.compile()


def _sequential_run(initial_state: AMLAgentState) -> AMLAgentState:
    """Sequential fallback when LangGraph is not installed."""
    state = analyst_node(initial_state)
    state = detective_node(state)
    state = narrator_node(state)
    state = commander_node(state)
    return state


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------
class OrchestratorAgent:
    """
    The Commander – entry point for the multi-agent AML pipeline.

    Usage
    -----
    agent = OrchestratorAgent()
    result = agent.process(transactions)
    """

    def __init__(self) -> None:
        if _HAS_LANGGRAPH:
            self._graph = _build_langgraph_graph()
            log.info("Using LangGraph orchestration.")
        else:
            self._graph = None
            log.info("Using sequential fallback orchestration.")

    def process(self, transactions: List[Dict]) -> AMLAgentState:
        """
        Process a batch of transactions through the full agent pipeline.

        Parameters
        ----------
        transactions : list of transaction dicts (same schema as TransactionRequest)

        Returns
        -------
        AMLAgentState : fully populated state dict with all agent outputs
        """
        initial: AMLAgentState = {
            "transactions": transactions,
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

        if self._graph is not None:
            result = self._graph.invoke(initial)
        else:
            result = _sequential_run(initial)

        # Persist results
        self._save_results(result)
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _save_results(state: AMLAgentState) -> None:
        """Persist orchestration results to reports/."""
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        summary = {
            "final_risk_level": state["final_risk_level"],
            "final_risk_score": state["final_risk_score"],
            "action_recommendation": state["action_recommendation"],
            "n_transactions": len(state["transactions"]),
            "n_flagged": len(state["flagged_transaction_ids"]),
            "n_sars": len(state["sar_ids"]),
            "sar_ids": state["sar_ids"],
            "network_summary": state["network_summary"],
        }
        (REPORTS_DIR / "orchestrator_result.json").write_text(
            json.dumps(summary, indent=2)
        )
        if state["sar_narratives"]:
            pd.DataFrame({
                "sar_id": state["sar_ids"],
                "narrative": state["sar_narratives"],
            }).to_csv(REPORTS_DIR / "sar_narratives.csv", index=False)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Quick smoke-test with synthetic transactions
    fake_txns = [
        {
            "transaction_id": f"TXN_{i:06d}",
            "timestamp": "2024-06-15T10:30:00",
            "sender_id": f"CUST_{random.randint(0, 100):04d}",
            "receiver_id": f"CUST_{random.randint(0, 100):04d}",
            "amount_usd": random.uniform(100, 50000),
            "transaction_type": "WIRE_TRANSFER",
            "country_origin": random.choice(["US", "PA", "KY"]),
            "country_dest": random.choice(["US", "PA", "KY"]),
            "round_amount": False,
            "rapid_movement": False,
            "structuring_flag": False,
            "is_suspicious": False,
            "label": "normal",
        }
        for i in range(50)
    ]

    agent = OrchestratorAgent()
    result = agent.process(fake_txns)
    print(f"\nFinal Risk: {result['final_risk_level']}  Score: {result['final_risk_score']:.4f}")
    print(f"Action: {result['action_recommendation']}")
