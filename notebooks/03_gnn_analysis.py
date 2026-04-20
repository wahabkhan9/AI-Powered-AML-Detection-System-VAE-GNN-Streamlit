"""
notebooks/03_gnn_analysis.py
=============================
Graph Neural Network results analysis:
  - Node risk score distribution
  - Graph community structure
  - High-risk customer subgraph visualisation
  - GNN training curves
"""
# %% Setup
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORTS_DIR = Path("reports")

# %% Load GNN risk scores
risk_path = REPORTS_DIR / "customer_risk_scores.parquet"
if not risk_path.exists():
    print("GNN risk scores not found. Run agents/network_investigator.py first.")
    sys.exit(0)

risk_df = pd.read_parquet(risk_path)
cust_df = pd.read_parquet("data/customers.parquet")
merged = risk_df.merge(cust_df, on="customer_id", how="left")
print(f"Loaded {len(merged):,} customer risk scores")

# %% Risk score distribution
fig, ax = plt.subplots(figsize=(9, 5))
bins = np.linspace(0, 1, 50)
ax.hist(merged.loc[merged["is_suspicious_true"] == 0, "gnn_risk_score"],
        bins=bins, alpha=0.65, color="#2196F3", label="Normal", density=True)
ax.hist(merged.loc[merged["is_suspicious_true"] == 1, "gnn_risk_score"],
        bins=bins, alpha=0.65, color="#F44336", label="Suspicious", density=True)
ax.axvline(0.5, color="black", linestyle="--", linewidth=1.4, label="Threshold = 0.50")
ax.set_xlabel("GNN Risk Score")
ax.set_ylabel("Density")
ax.set_title("GNN Customer Risk Score Distribution")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "gnn_score_distribution.png", dpi=150)
plt.close()
print("Saved: gnn_score_distribution.png")

# %% Top-risk customers table
top = merged.nlargest(20, "gnn_risk_score")[
    ["customer_id", "gnn_risk_score", "is_suspicious_true", "jurisdiction_risk", "risk_score"]
].reset_index(drop=True)
print("\n── Top 20 High-Risk Customers ──")
print(top.to_string(index=False))
top.to_csv(REPORTS_DIR / "top_risk_customers.csv", index=False)

# %% Risk score vs declared customer risk_score scatter
fig, ax = plt.subplots(figsize=(7, 6))
colors = merged["is_suspicious_true"].map({0: "#2196F3", 1: "#F44336"})
ax.scatter(merged["risk_score"], merged["gnn_risk_score"],
           c=colors, alpha=0.15, s=8)
ax.set_xlabel("Customer Risk Score (rule-based)")
ax.set_ylabel("GNN Risk Score (learned)")
ax.set_title("Rule-based vs GNN Risk Scores")
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color="#2196F3", label="Normal"),
    Patch(color="#F44336", label="Suspicious"),
])
plt.tight_layout()
plt.savefig(REPORTS_DIR / "gnn_vs_rule_scatter.png", dpi=150)
plt.close()

# %% Jurisdiction breakdown of high-risk customers
high_risk = merged[merged["gnn_risk_score"] >= 0.7]
if "jurisdiction_risk" in high_risk.columns:
    jur = high_risk["jurisdiction_risk"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    jur.plot.bar(ax=ax, color=["#F44336", "#FF9800", "#2196F3"])
    ax.set_title("Jurisdiction Distribution of High-Risk Customers (GNN ≥ 0.7)")
    ax.set_xlabel("Jurisdiction Risk")
    ax.set_ylabel("Customer Count")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "gnn_high_risk_jurisdictions.png", dpi=150)
    plt.close()

# %% GNN training history
gnn_hist_path = REPORTS_DIR / "gnn_training_history.csv"
if gnn_hist_path.exists():
    hist = pd.read_csv(gnn_hist_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hist["epoch"], hist["train_loss"], color="#9C27B0", linewidth=1.8)
    axes[0].set_title("GNN Training Loss (Focal)")
    axes[0].set_xlabel("Epoch"); axes[0].grid(alpha=0.3)
    axes[1].plot(hist["epoch"], hist["val_recall"], color="#4CAF50", linewidth=1.8)
    axes[1].set_title("GNN Validation Recall")
    axes[1].set_xlabel("Epoch"); axes[1].grid(alpha=0.3)
    plt.suptitle("GNN Training History", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "gnn_training_history.png", dpi=150)
    plt.close()

print("\nGNN analysis complete. All figures saved to reports/")
