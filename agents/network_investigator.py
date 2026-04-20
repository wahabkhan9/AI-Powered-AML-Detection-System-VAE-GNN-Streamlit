"""
agents/network_investigator.py
===============================
Network Investigator Agent – builds a transaction graph, trains the GNN,
and produces customer-level risk scores.

Node features per customer
--------------------------
- degree (in + out)
- total_amount (log-scaled)
- unique_counterparties
- risk_score  (from customer table)
- jurisdiction_risk (ordinal 0/1/2)
- avg_txn_amount
- txn_count
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.gnn import CustomerRiskGNN

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
REPORTS_DIR = Path("reports")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_CHANNELS = 64
HEADS = 4
DROPOUT = 0.3
EPOCHS = 80
LR = 5e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 10


JURISDICTION_MAP = {"low": 0, "medium": 1, "high": 2}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_graph(
    txn_df: pd.DataFrame,
    cust_df: pd.DataFrame,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, Dict]:
    """
    Construct node feature matrix X and COO edge_index from transaction data.

    Returns
    -------
    X          : [N, F] float32 node features (scaled)
    edge_index : [2, E] long COO adjacency
    edge_weight: [E] float32 log-amount edge weights
    labels     : [N] int labels
    id_map     : customer_id → node index
    """
    cust_ids = cust_df["customer_id"].tolist()
    id_map: Dict[str, int] = {cid: i for i, cid in enumerate(cust_ids)}
    n_nodes = len(cust_ids)

    # ── node features ────────────────────────────────────────────────────────
    # Aggregate transaction stats per customer
    sent = txn_df.groupby("sender_id").agg(
        sent_count=("amount_usd", "count"),
        sent_total=("amount_usd", "sum"),
        sent_unique=("receiver_id", "nunique"),
    )
    recv = txn_df.groupby("receiver_id").agg(
        recv_count=("amount_usd", "count"),
        recv_total=("amount_usd", "sum"),
        recv_unique=("sender_id", "nunique"),
    )

    feat = cust_df.set_index("customer_id")[["risk_score", "jurisdiction_risk"]].copy()
    feat["jurisdiction_risk"] = feat["jurisdiction_risk"].map(JURISDICTION_MAP).fillna(1)
    feat = feat.join(sent, how="left").join(recv, how="left").fillna(0.0)

    feat["degree"] = feat["sent_count"] + feat["recv_count"]
    feat["total_amount"] = np.log1p(feat["sent_total"] + feat["recv_total"])
    feat["unique_counterparties"] = feat["sent_unique"] + feat["recv_unique"]
    feat["txn_count"] = feat["sent_count"] + feat["recv_count"]
    feat["avg_txn_amount"] = (
        (feat["sent_total"] + feat["recv_total"])
        / feat["txn_count"].clip(lower=1)
    )

    feature_cols = [
        "degree", "total_amount", "unique_counterparties",
        "risk_score", "jurisdiction_risk", "avg_txn_amount", "txn_count",
    ]
    X_raw = feat.reindex(cust_ids)[feature_cols].values.astype(np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw).astype(np.float32)

    # ── edge_index ───────────────────────────────────────────────────────────
    valid_mask = (
        txn_df["sender_id"].isin(id_map) & txn_df["receiver_id"].isin(id_map)
    )
    txn_valid = txn_df[valid_mask]

    src_idx = txn_valid["sender_id"].map(id_map).values
    dst_idx = txn_valid["receiver_id"].map(id_map).values
    edge_index = torch.tensor(
        np.stack([src_idx, dst_idx], axis=0), dtype=torch.long
    )
    edge_weight = torch.tensor(
        np.log1p(txn_valid["amount_usd"].values).astype(np.float32)
    )

    labels = cust_df.set_index("customer_id").reindex(cust_ids)["is_suspicious"]
    labels = labels.fillna(False).astype(int).values

    log.info(
        "Graph: %d nodes, %d edges, %d suspicious (%.2f%%)",
        n_nodes, edge_index.shape[1],
        labels.sum(), 100 * labels.mean(),
    )

    # Persist scaler
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "gnn_node_scaler.pkl", "wb") as f:
        pickle.dump((scaler, feature_cols, id_map), f, protocol=pickle.HIGHEST_PROTOCOL)

    return (
        torch.tensor(X_scaled, dtype=torch.float32),
        edge_index,
        edge_weight,
        labels,
        id_map,
    )


# ---------------------------------------------------------------------------
# Class-weighted focal loss helper
# ---------------------------------------------------------------------------
def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.75,
) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = probs[torch.arange(len(targets)), targets]
    focal_weight = (1 - pt) ** gamma
    alpha_t = torch.where(targets == 1, torch.tensor(alpha), torch.tensor(1 - alpha))
    alpha_t = alpha_t.to(logits.device)
    return (alpha_t * focal_weight * ce).mean()


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class NetworkInvestigatorAgent:
    """Build transaction graph, train GNN, score every customer."""

    def __init__(self) -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def run(self) -> None:
        log.info("=== NetworkInvestigatorAgent starting ===")

        txn_df = pd.read_parquet(DATA_DIR / "transactions.parquet")
        cust_df = pd.read_parquet(DATA_DIR / "customers.parquet")

        X, edge_index, edge_weight, labels, id_map = build_graph(txn_df, cust_df)

        # ── train / val / test masks ─────────────────────────────────────────
        idx = np.arange(len(labels))
        train_idx, test_idx = train_test_split(
            idx, test_size=0.20, random_state=42, stratify=labels
        )
        train_idx, val_idx = train_test_split(
            train_idx, test_size=0.15, random_state=42,
            stratify=labels[train_idx],
        )
        train_mask = torch.zeros(len(labels), dtype=torch.bool)
        val_mask = torch.zeros(len(labels), dtype=torch.bool)
        test_mask = torch.zeros(len(labels), dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        X = X.to(DEVICE)
        edge_index = edge_index.to(DEVICE)
        edge_weight = edge_weight.to(DEVICE)
        y = torch.tensor(labels, dtype=torch.long).to(DEVICE)

        # ── model ────────────────────────────────────────────────────────────
        model = CustomerRiskGNN(
            in_channels=X.shape[1],
            hidden_channels=HIDDEN_CHANNELS,
            out_channels=2,
            heads=HEADS,
            dropout=DROPOUT,
        ).to(DEVICE)
        log.info("GNN parameters: %d", sum(p.numel() for p in model.parameters()))

        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5,
        )

        best_val_recall = 0.0
        best_state = None
        patience_count = 0
        history = []

        for epoch in range(1, EPOCHS + 1):
            model.train()
            logits = model(X, edge_index)
            loss = focal_loss(logits[train_mask], y[train_mask])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_logits = model(X, edge_index)
                val_proba = F.softmax(val_logits, dim=1)[:, 1]
                val_pred = (val_proba[val_mask] >= 0.5).long()
                val_labels = y[val_mask]
                tp = ((val_pred == 1) & (val_labels == 1)).sum().item()
                fn = ((val_pred == 0) & (val_labels == 1)).sum().item()
                val_recall = tp / max(tp + fn, 1)

            scheduler.step(val_recall)
            history.append({"epoch": epoch, "train_loss": loss.item(), "val_recall": val_recall})

            if epoch % 10 == 0 or epoch == 1:
                log.info(
                    "Epoch %3d/%d  loss=%.4f  val_recall=%.4f",
                    epoch, EPOCHS, loss.item(), val_recall,
                )

            if val_recall > best_val_recall:
                best_val_recall = val_recall
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= PATIENCE:
                    log.info("Early stopping at epoch %d", epoch)
                    break

        # ── test evaluation ──────────────────────────────────────────────────
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            test_proba = model.predict_proba(X, edge_index)[:, 1].cpu().numpy()
            test_pred = (test_proba[test_mask.cpu().numpy()] >= 0.5).astype(int)
            test_true = labels[test_mask.cpu().numpy()]

        report = classification_report(test_true, test_pred, digits=4)
        log.info("\nGNN Test Classification Report:\n%s", report)

        # ── persist ──────────────────────────────────────────────────────────
        torch.save(best_state, MODEL_DIR / "gnn_model.pth")
        pd.DataFrame(history).to_csv(REPORTS_DIR / "gnn_training_history.csv", index=False)

        # Customer risk scores
        risk_df = pd.DataFrame({
            "customer_id": list(id_map.keys()),
            "gnn_risk_score": test_proba if len(test_proba) == len(id_map)
            else model.predict_proba(X, edge_index)[:, 1].cpu().numpy(),
            "is_suspicious_true": labels,
        })
        risk_df.to_parquet(REPORTS_DIR / "customer_risk_scores.parquet", index=False)
        log.info("Customer risk scores saved.")
        log.info("=== NetworkInvestigatorAgent finished ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    agent = NetworkInvestigatorAgent()
    agent.run()
