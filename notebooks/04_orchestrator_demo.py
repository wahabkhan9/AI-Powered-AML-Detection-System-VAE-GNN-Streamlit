"""
notebooks/04_orchestrator_demo.py
===================================
End-to-end multi-agent pipeline demonstration.

Shows the full LangGraph orchestrator running:
  Analyst (VAE + GAN) → Detective (GNN) → Narrator (LLM/Ollama) → Commander

Run: python notebooks/04_orchestrator_demo.py
"""
# %% Setup
from __future__ import annotations
import random
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

print("=" * 60)
print("  AML Multi-Agent System – Orchestrator Demo")
print("=" * 60)

# %% Load real transactions (or use synthetic)
txn_path = Path("data/transactions.parquet")
if txn_path.exists():
    txn_df = pd.read_parquet(txn_path)
    # Sample 200 transactions including some suspicious ones
    normal_sample = txn_df[~txn_df["is_suspicious"]].sample(170, random_state=42)
    susp_sample   = txn_df[txn_df["is_suspicious"]].sample(
        min(30, txn_df["is_suspicious"].sum()), random_state=42
    )
    sample_df = pd.concat([normal_sample, susp_sample]).sample(frac=1, random_state=42)
    transactions = sample_df.to_dict(orient="records")
    print(f"Loaded {len(transactions)} transactions from dataset")
    print(f"  Suspicious: {sum(t['is_suspicious'] for t in transactions)}")
else:
    print("No dataset found – generating synthetic transactions for demo …")
    transactions = [
        {
            "transaction_id": f"DEMO_{i:04d}",
            "timestamp": "2024-06-15T10:30:00",
            "sender_id": f"CUST_{random.randint(0, 50):04d}",
            "receiver_id": f"CUST_{random.randint(0, 50):04d}",
            "amount_usd": random.choice([9_500, 9_800, 150_000, random.uniform(100, 50_000)]),
            "transaction_type": random.choice(["WIRE_TRANSFER", "ACH", "CASH_DEPOSIT"]),
            "country_origin": random.choice(["US", "PA", "KY", "BZ", "DE"]),
            "country_dest": random.choice(["US", "PA", "KY", "MX"]),
            "round_amount": random.choice([True, False]),
            "rapid_movement": random.choice([True, False]),
            "structuring_flag": random.random() < 0.25,
            "is_suspicious": random.random() < 0.15,
            "label": "normal",
        }
        for i in range(200)
    ]
    print(f"Generated {len(transactions)} synthetic transactions")

# %% Run the multi-agent orchestrator
print("\n── Running Multi-Agent Pipeline ──────────────────────────")
from agents.orchestrator_agent import OrchestratorAgent

agent = OrchestratorAgent()
result = agent.process(transactions)

# %% Print results
print("\n╔" + "═" * 58 + "╗")
print("║  ORCHESTRATOR RESULTS" + " " * 36 + "║")
print("╠" + "═" * 58 + "╣")
print(f"║  Transactions Processed  : {len(transactions):<30}║")
print(f"║  Flagged Transactions    : {len(result['flagged_transaction_ids']):<30}║")
print(f"║  Customers Investigated  : {len(result['customer_ids']):<30}║")
print(f"║  High-Risk Clusters      : {len(result['suspicious_clusters']):<30}║")
print(f"║  SARs Generated          : {len(result['sar_ids']):<30}║")
print(f"║  Final Risk Level        : {result['final_risk_level']:<30}║")
print(f"║  Final Risk Score        : {result['final_risk_score']:<30.4f}║")
print("╠" + "═" * 58 + "╣")
net_summary = result['network_summary'][:55] if result['network_summary'] else "N/A"
print(f"║  Network: {net_summary:<48}║")
action_str = result['action_recommendation'][:55]
print(f"║  Action : {action_str:<48}║")
print("╚" + "═" * 58 + "╝")

# %% Agent-level breakdown
print("\n── Agent Score Distribution ───────────────────────────────")
if result["vae_scores"]:
    vae_arr = np.array(result["vae_scores"])
    gan_arr = np.array(result["gan_scores"])
    print(f"  VAE Scores  – mean: {vae_arr.mean():.6f}  max: {vae_arr.max():.6f}  p95: {np.percentile(vae_arr, 95):.6f}")
    print(f"  GAN Scores  – mean: {gan_arr.mean():.4f}  max: {gan_arr.max():.4f}  p95: {np.percentile(gan_arr, 95):.4f}")

# %% GNN risk scores
if result["gnn_risk_scores"]:
    scores = list(result["gnn_risk_scores"].values())
    print(f"\n  GNN Risk Scores ({len(scores)} customers):")
    print(f"    Mean: {np.mean(scores):.4f}  Max: {max(scores):.4f}  High-risk (>0.6): {sum(s>0.6 for s in scores)}")

# %% SAR narratives
if result["sar_narratives"]:
    print(f"\n── Sample SAR Narrative (SAR ID: {result['sar_ids'][0]}) ─────")
    print(result["sar_narratives"][0][:600])
    print("…" if len(result["sar_narratives"][0]) > 600 else "")
else:
    print("\n  No SAR narratives generated (no flagged transactions).")

# %% Save results
output_path = Path("reports/orchestrator_demo_result.json")
output_path.parent.mkdir(exist_ok=True)
output_path.write_text(json.dumps({
    "final_risk_level": result["final_risk_level"],
    "final_risk_score": result["final_risk_score"],
    "n_transactions": len(transactions),
    "n_flagged": len(result["flagged_transaction_ids"]),
    "n_sars": len(result["sar_ids"]),
    "action_recommendation": result["action_recommendation"],
    "network_summary": result["network_summary"],
}, indent=2))
print(f"\nResults saved → {output_path}")
print("\n✓ Orchestrator demo complete.")
