"""
agents/orchestrator_agent.py
=============================
The Commander – FULLY LangGraph-based multi-agent orchestrator.

This is a TRUE LangGraph implementation with:
  - TypedDict state schema (AMLAgentState)
  - StateGraph with named nodes
  - CONDITIONAL edges: if no anomalies found → skip detective/narrator
  - MemorySaver checkpointer for state persistence across runs
  - Streaming graph execution

Graph topology:
  [analyst] ──► should_investigate? ──► YES ──► [detective] ──► [narrator] ──► [commander]
                                         NO ──────────────────────────────────► [commander]

Falls back to identical sequential execution if langgraph is not installed.
"""

from __future__ import annotations

import json
import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict

import numpy as np
import pandas as pd
import torch

log = logging.getLogger(__name__)

MODEL_DIR   = Path("models")
REPORTS_DIR = Path("reports")
DATA_DIR    = Path("data")

# ---------------------------------------------------------------------------
# LangGraph import with graceful fallback
# ---------------------------------------------------------------------------
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    _HAS_LANGGRAPH = True
    log.info("LangGraph detected – using graph-based orchestration with MemorySaver.")
except ImportError:
    _HAS_LANGGRAPH = False
    log.warning(
        "langgraph not installed. Sequential fallback active.\n"
        "Install: pip install langgraph langchain-core"
    )


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------
class AMLAgentState(TypedDict):
    """
    Shared mutable state passed between all agent nodes.
    Every field has a sensible default so nodes can be skipped gracefully.
    """
    # ── Input ──────────────────────────────────────────────────────────
    transactions: List[Dict]

    # ── Analyst outputs ────────────────────────────────────────────────
    vae_scores:               List[float]
    gan_scores:               List[float]
    combined_anomaly_scores:  List[float]
    flagged_transaction_ids:  List[str]
    analyst_summary:          str

    # ── Detective outputs ───────────────────────────────────────────────
    customer_ids:             List[str]
    gnn_risk_scores:          Dict[str, float]
    suspicious_clusters:      List[List[str]]
    network_summary:          str

    # ── Narrator outputs ────────────────────────────────────────────────
    sar_narratives:           List[str]
    sar_ids:                  List[str]

    # ── Commander outputs ───────────────────────────────────────────────
    final_risk_level:         str
    final_risk_score:         float
    action_recommendation:    str
    processing_complete:      bool

    # ── Routing metadata ───────────────────────────────────────────────
    skip_investigation:       bool   # True when 0 anomalies found


# ---------------------------------------------------------------------------
# Lazy model registry (singleton)
# ---------------------------------------------------------------------------
class _ModelRegistry:
    """Loads trained models once and caches them in memory."""
    _instance: Optional["_ModelRegistry"] = None

    def __new__(cls) -> "_ModelRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._ready = False
        return cls._instance

    def load(self) -> None:
        if self._ready:
            return

        from features.feature_engineering import derive_transaction_features
        self.derive_features = derive_transaction_features

        # VAE
        meta_path = MODEL_DIR / "vae_meta.json"
        if meta_path.exists():
            from models.vae import VAE
            meta = json.loads(meta_path.read_text())
            vae = VAE(
                input_dim=meta["input_dim"],
                latent_dim=meta["latent_dim"],
                hidden_dims=tuple(meta["hidden_dims"]),
                beta=meta["beta"],
            )
            vae.load_state_dict(torch.load(MODEL_DIR / "vae_model.pth", map_location="cpu"))
            vae.eval()
            self.vae           = vae
            self.vae_threshold = meta["threshold"]
            self.feature_cols  = meta.get("feature_cols", [])
            self.vae_weight    = meta.get("vae_weight", 0.6)
            self.gan_weight    = meta.get("gan_weight", 0.4)
        else:
            self.vae = self.vae_threshold = None
            self.vae_weight = 0.6; self.gan_weight = 0.4
            log.warning("VAE not trained yet – using stub scores.")

        # Scaler
        scaler_path = MODEL_DIR / "scaler.pkl"
        self.scaler = pickle.load(open(scaler_path, "rb")) if scaler_path.exists() else None

        # GAN discriminator
        gan_path = MODEL_DIR / "gan_discriminator.pth"
        gan_meta_path = MODEL_DIR / "gan_meta.json"
        if gan_path.exists() and gan_meta_path.exists():
            from models.gan import TransactionGAN
            gm = json.loads(gan_meta_path.read_text())
            gan = TransactionGAN(feature_dim=gm["feature_dim"], latent_dim=gm["latent_dim"])
            gan.discriminator.load_state_dict(torch.load(gan_path, map_location="cpu"))
            gan.discriminator.eval()
            self.gan_disc = gan.discriminator
            log.info("GAN discriminator loaded into orchestrator.")
        else:
            self.gan_disc = None

        # GNN risk scores
        risk_path = REPORTS_DIR / "customer_risk_scores.parquet"
        self.risk_df = pd.read_parquet(risk_path) if risk_path.exists() else pd.DataFrame()

        self._ready = True
        log.info("ModelRegistry ready.")

    def score_transactions(self, txn_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Run VAE + optional GAN scoring. Returns {'vae': arr, 'gan': arr, 'combined': arr}."""
        n = len(txn_df)
        if self.vae is None or self.scaler is None:
            # Stub mode
            vae_s = np.random.uniform(0, 0.02, n).astype(np.float32)
            gan_s = np.random.uniform(0, 1, n).astype(np.float32)
            return {"vae": vae_s, "gan": gan_s, "combined": vae_s}

        feat = self.derive_features(txn_df)
        X    = self.scaler.transform(feat.values).astype(np.float32)
        t    = torch.tensor(X)

        with torch.no_grad():
            vae_scores = self.vae.anomaly_score(t).numpy()

        if self.gan_disc is not None:
            with torch.no_grad():
                gan_scores = self.gan_disc.anomaly_score(t).numpy()
            vae_norm  = vae_scores / (vae_scores.max() + 1e-9)
            combined  = self.vae_weight * vae_norm + self.gan_weight * gan_scores
        else:
            gan_scores = np.zeros(n, dtype=np.float32)
            combined   = vae_scores / (vae_scores.max() + 1e-9)

        return {"vae": vae_scores, "gan": gan_scores, "combined": combined}

    def get_gnn_risk(self, customer_id: str) -> float:
        if self.risk_df.empty or "customer_id" not in self.risk_df.columns:
            return 0.0
        row = self.risk_df[self.risk_df["customer_id"] == customer_id]
        return float(row["gnn_risk_score"].iloc[0]) if not row.empty else 0.0


_registry = _ModelRegistry()


# ---------------------------------------------------------------------------
# Agent node functions
# ---------------------------------------------------------------------------
def analyst_node(state: AMLAgentState) -> AMLAgentState:
    """
    Analyst Agent – scores every transaction with VAE + GAN fusion.
    Sets skip_investigation=True when nothing is flagged.
    """
    log.info("[Analyst] Scoring %d transactions …", len(state["transactions"]))
    _registry.load()

    txn_df = pd.DataFrame(state["transactions"])
    if txn_df.empty:
        return {**state,
                "vae_scores": [], "gan_scores": [], "combined_anomaly_scores": [],
                "flagged_transaction_ids": [], "analyst_summary": "No transactions.",
                "skip_investigation": True}

    txn_df["timestamp"] = pd.to_datetime(
        txn_df.get("timestamp", pd.Series(["2024-01-01"] * len(txn_df)))
    )
    scores  = _registry.score_transactions(txn_df)
    vae_s   = scores["vae"].tolist()
    gan_s   = scores["gan"].tolist()
    comb_s  = scores["combined"].tolist()

    threshold = _registry.vae_threshold or 0.01
    flagged   = [
        txn["transaction_id"]
        for txn, score in zip(state["transactions"], scores["combined"])
        if score >= threshold
    ]

    skip = len(flagged) == 0
    summary = (
        f"Scored {len(state['transactions'])} transactions. "
        f"Flagged {len(flagged)} (threshold={threshold:.6f}). "
        f"GAN fusion: {'active' if _registry.gan_disc is not None else 'inactive'}."
    )
    log.info("[Analyst] %s", summary)

    return {
        **state,
        "vae_scores":              vae_s,
        "gan_scores":              gan_s,
        "combined_anomaly_scores": comb_s,
        "flagged_transaction_ids": flagged,
        "analyst_summary":         summary,
        "skip_investigation":      skip,
    }


def detective_node(state: AMLAgentState) -> AMLAgentState:
    """
    Detective Agent – loads GNN risk scores, identifies suspicious clusters.
    Only runs when analyst flagged at least one transaction.
    """
    log.info("[Detective] Investigating %d flagged transactions …",
             len(state["flagged_transaction_ids"]))
    _registry.load()

    flagged_set = set(state["flagged_transaction_ids"])
    involved: set[str] = set()
    for txn in state["transactions"]:
        if txn["transaction_id"] in flagged_set:
            involved.add(txn["sender_id"])
            involved.add(txn["receiver_id"])

    gnn_risk: Dict[str, float] = {
        cid: _registry.get_gnn_risk(cid) for cid in involved
    }

    high_risk = [cid for cid, s in gnn_risk.items() if s > 0.6]
    clusters  = [high_risk] if high_risk else []

    flagged_amounts = [
        float(txn["amount_usd"])
        for txn in state["transactions"]
        if txn["transaction_id"] in flagged_set
    ]
    total_amount = sum(flagged_amounts)
    avg_gnn = float(np.mean(list(gnn_risk.values()))) if gnn_risk else 0.0

    summary = (
        f"Investigated {len(involved)} customers across {len(flagged_set)} flagged txns. "
        f"High-risk accounts (GNN>0.6): {len(high_risk)}. "
        f"Total flagged: USD {total_amount:,.2f}. "
        f"Avg GNN risk: {avg_gnn:.4f}."
    )
    log.info("[Detective] %s", summary)

    return {
        **state,
        "customer_ids":        list(involved),
        "gnn_risk_scores":     gnn_risk,
        "suspicious_clusters": clusters,
        "network_summary":     summary,
    }


def narrator_node(state: AMLAgentState) -> AMLAgentState:
    """
    Narrator Agent – drafts SAR narratives via Ollama (local LLM).
    Auto-falls back to rule-based templates if Ollama is not running.
    """
    log.info("[Narrator] Drafting SAR narratives …")

    from llm.ollama_writer import OllamaSARWriter
    writer = OllamaSARWriter()   # auto-detects Ollama; falls back to templates

    flagged_set = set(state["flagged_transaction_ids"])
    by_sender: Dict[str, list] = defaultdict(list)
    for txn in state["transactions"]:
        if txn["transaction_id"] in flagged_set:
            by_sender[txn["sender_id"]].append(txn)

    import uuid
    narratives, sar_ids = [], []

    for sender_id, txns in list(by_sender.items())[:10]:
        gnn_score = state["gnn_risk_scores"].get(sender_id, 0.0)

        # Average VAE score for this sender's transactions
        txn_ids = {t["transaction_id"] for t in txns}
        vae_avg = float(np.mean([
            state["vae_scores"][i]
            for i, t in enumerate(state["transactions"])
            if t["transaction_id"] in txn_ids
        ])) if state["vae_scores"] else 0.0

        context = {
            "customer_id":        sender_id,
            "transaction_count":  len(txns),
            "total_amount_usd":   sum(float(t["amount_usd"]) for t in txns),
            "gnn_risk_score":     gnn_score,
            "vae_score":          vae_avg,
            "network_summary":    state.get("network_summary", ""),
            "transaction_types":  list({t.get("transaction_type", "UNKNOWN") for t in txns}),
        }
        narratives.append(writer.generate(context))
        sar_ids.append("SAR-" + uuid.uuid4().hex[:8].upper())

    log.info("[Narrator] Generated %d SAR narratives.", len(narratives))
    return {**state, "sar_narratives": narratives, "sar_ids": sar_ids}


def commander_node(state: AMLAgentState) -> AMLAgentState:
    """
    Commander Agent – synthesises all findings into final risk score + action.
    Runs whether or not the detective/narrator were skipped.
    """
    log.info("[Commander] Producing final risk assessment …")

    n_total   = len(state["transactions"])
    n_flagged = len(state["flagged_transaction_ids"])
    flag_rate = n_flagged / max(n_total, 1)

    avg_combined = float(np.mean(state["combined_anomaly_scores"])) \
        if state["combined_anomaly_scores"] else 0.0
    avg_gnn = float(np.mean(list(state["gnn_risk_scores"].values()))) \
        if state["gnn_risk_scores"] else 0.0

    final_score = round(
        0.5 * min(flag_rate * 10, 1.0) + 0.3 * avg_combined + 0.2 * avg_gnn, 4
    )

    if final_score >= 0.75:
        level  = "CRITICAL"
        action = (
            "Immediate escalation to Chief Compliance Officer. "
            "Freeze implicated accounts pending investigation. "
            "File SAR with FinCEN within 30 days. Notify BSA officer."
        )
    elif final_score >= 0.50:
        level  = "HIGH"
        action = (
            "Priority review by senior AML analyst within 24 hours. "
            "Enhanced due diligence (EDD) required. SAR filing recommended."
        )
    elif final_score >= 0.25:
        level  = "MEDIUM"
        action = (
            "Schedule analyst review within 5 business days. "
            "Document findings. Continue enhanced monitoring."
        )
    else:
        level  = "LOW"
        action = "Standard automated monitoring. No immediate action required."

    log.info("[Commander] Risk=%s  Score=%.4f  Flagged=%d/%d",
             level, final_score, n_flagged, n_total)

    return {
        **state,
        "final_risk_level":      level,
        "final_risk_score":      final_score,
        "action_recommendation": action,
        "processing_complete":   True,
    }


# ---------------------------------------------------------------------------
# Conditional routing function (the key LangGraph feature)
# ---------------------------------------------------------------------------
def _should_investigate(state: AMLAgentState) -> Literal["investigate", "commander"]:
    """
    Conditional edge: if no transactions were flagged, skip straight to
    Commander for a LOW risk assessment rather than running detective + narrator.
    """
    if state.get("skip_investigation", False):
        log.info("[Router] No anomalies detected → skipping detective/narrator.")
        return "commander"
    log.info("[Router] Anomalies found → running detective + narrator.")
    return "investigate"


# ---------------------------------------------------------------------------
# Build the LangGraph graph
# ---------------------------------------------------------------------------
def _build_graph():
    """
    Constructs the full LangGraph StateGraph with:
      - Conditional routing after analyst
      - MemorySaver checkpointer (persistent state across invocations)
    """
    graph = StateGraph(AMLAgentState)

    # Register nodes
    graph.add_node("analyst",   analyst_node)
    graph.add_node("detective", detective_node)
    graph.add_node("narrator",  narrator_node)
    graph.add_node("commander", commander_node)

    # Entry point
    graph.set_entry_point("analyst")

    # CONDITIONAL EDGE: analyst → investigate OR commander
    graph.add_conditional_edges(
        "analyst",
        _should_investigate,
        {
            "investigate": "detective",   # anomalies found
            "commander":   "commander",   # nothing flagged → skip to commander
        },
    )

    # Linear edges for normal path
    graph.add_edge("detective", "narrator")
    graph.add_edge("narrator",  "commander")
    graph.add_edge("commander", END)

    # Compile with MemorySaver for state persistence across runs
    checkpointer = MemorySaver() if _HAS_LANGGRAPH else None
    if checkpointer:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()


def _sequential_run(state: AMLAgentState) -> AMLAgentState:
    """Exact same logic as the graph, but sequential (fallback)."""
    state = analyst_node(state)
    if not state.get("skip_investigation"):
        state = detective_node(state)
        state = narrator_node(state)
    state = commander_node(state)
    return state


# ---------------------------------------------------------------------------
# Public OrchestratorAgent
# ---------------------------------------------------------------------------
class OrchestratorAgent:
    """
    The Commander – processes a batch of transactions through the full
    multi-agent pipeline and returns a complete AMLAgentState.

    Usage
    -----
    agent = OrchestratorAgent()
    result = agent.process(transactions)
    print(result["final_risk_level"])
    """

    def __init__(self) -> None:
        if _HAS_LANGGRAPH:
            self._graph = _build_graph()
            log.info("OrchestratorAgent ready (LangGraph with MemorySaver).")
        else:
            self._graph = None
            log.info("OrchestratorAgent ready (sequential fallback).")

    # ------------------------------------------------------------------
    def process(
        self,
        transactions: List[Dict],
        thread_id: str = "default",
    ) -> AMLAgentState:
        """
        Process a batch of transactions through the agent graph.

        Parameters
        ----------
        transactions : list of transaction dicts
        thread_id    : LangGraph thread identifier for checkpointing

        Returns
        -------
        AMLAgentState : fully populated state
        """
        initial: AMLAgentState = {
            "transactions":            transactions,
            "vae_scores":              [],
            "gan_scores":              [],
            "combined_anomaly_scores": [],
            "flagged_transaction_ids": [],
            "analyst_summary":         "",
            "customer_ids":            [],
            "gnn_risk_scores":         {},
            "suspicious_clusters":     [],
            "network_summary":         "",
            "sar_narratives":          [],
            "sar_ids":                 [],
            "final_risk_level":        "UNKNOWN",
            "final_risk_score":        0.0,
            "action_recommendation":   "",
            "processing_complete":     False,
            "skip_investigation":      False,
        }

        if self._graph is not None:
            config = {"configurable": {"thread_id": thread_id}}
            result = self._graph.invoke(initial, config=config)
        else:
            result = _sequential_run(initial)

        self._persist(result)
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _persist(state: AMLAgentState) -> None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        summary = {
            "final_risk_level":      state["final_risk_level"],
            "final_risk_score":      state["final_risk_score"],
            "action_recommendation": state["action_recommendation"],
            "n_transactions":        len(state["transactions"]),
            "n_flagged":             len(state["flagged_transaction_ids"]),
            "n_sars":                len(state["sar_ids"]),
            "sar_ids":               state["sar_ids"],
            "analyst_summary":       state.get("analyst_summary", ""),
            "network_summary":       state.get("network_summary", ""),
            "skip_investigation":    state.get("skip_investigation", False),
        }
        (REPORTS_DIR / "orchestrator_result.json").write_text(
            json.dumps(summary, indent=2)
        )
        if state["sar_narratives"]:
            pd.DataFrame({
                "sar_id":    state["sar_ids"],
                "narrative": state["sar_narratives"],
            }).to_csv(REPORTS_DIR / "sar_narratives.csv", index=False)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    fake = [
        {
            "transaction_id":  f"TXN_{i:04d}",
            "timestamp":       "2024-06-15T10:30:00",
            "sender_id":       f"CUST_{random.randint(0, 30):04d}",
            "receiver_id":     f"CUST_{random.randint(0, 30):04d}",
            "amount_usd":      random.uniform(100, 50_000),
            "transaction_type": "WIRE_TRANSFER",
            "country_origin":  random.choice(["US", "PA", "KY"]),
            "country_dest":    random.choice(["US", "DE"]),
            "round_amount":    False,
            "rapid_movement":  random.random() < 0.2,
            "structuring_flag": random.random() < 0.1,
            "is_suspicious":   False,
            "label":           "normal",
        }
        for i in range(50)
    ]
    agent = OrchestratorAgent()
    result = agent.process(fake, thread_id="smoke-test")
    print(f"Risk: {result['final_risk_level']}  Score: {result['final_risk_score']:.4f}")
    print(f"Action: {result['action_recommendation']}")