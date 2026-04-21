"""
notebooks/05_causal_analysis.py
================================
Causal Inference & What-If Policy Analysis.

Demonstrates:
  1. Threshold sensitivity what-if
  2. New rule impact simulation
  3. Difference-in-Differences policy effect estimation
  4. Visualisation of causal results
"""
# %% Setup
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import asdict

from causal.causal_inference import AMLCausalAnalyzer

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# %% Load data
print("Loading transaction data …")
txn_df = pd.read_parquet("data/transactions.parquet")
txn_df["timestamp"] = pd.to_datetime(txn_df["timestamp"])
print(f"Loaded {len(txn_df):,} transactions")

analyzer = AMLCausalAnalyzer(txn_df)

# %% 1. Threshold sensitivity sweep
print("\n── Threshold Sensitivity Analysis ────────────────────────")
meta = json.loads(Path("models/vae_meta.json").read_text()) if Path("models/vae_meta.json").exists() else {"threshold": 0.01}
current_threshold = meta["threshold"]

pct_changes = list(range(-50, 55, 5))
results = []
for pct in pct_changes:
    new_t = current_threshold * (1 + pct / 100)
    r = analyzer.what_if_threshold(current_threshold, new_t)
    results.append({
        "pct_change": pct,
        "new_threshold": new_t,
        "alerts": r.counterfactual_alerts,
        "delta_pct": r.delta_pct,
        "flagged_amount": r.counterfactual_flagged_amount,
    })

sweep_df = pd.DataFrame(results)
print(sweep_df.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(sweep_df["pct_change"], sweep_df["alerts"], marker="o", color="#1976D2", linewidth=2)
axes[0].axvline(0, color="red", linestyle="--", alpha=0.6, label="Current threshold")
axes[0].set_xlabel("Threshold Change (%)")
axes[0].set_ylabel("Alert Count")
axes[0].set_title("Alert Volume vs Threshold Change")
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(sweep_df["pct_change"], sweep_df["flagged_amount"] / 1e6, marker="s", color="#388E3C", linewidth=2)
axes[1].axvline(0, color="red", linestyle="--", alpha=0.6, label="Current threshold")
axes[1].set_xlabel("Threshold Change (%)")
axes[1].set_ylabel("Flagged Amount (USD millions)")
axes[1].set_title("Flagged Volume vs Threshold Change")
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.suptitle("What-If: Detection Threshold Sensitivity", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "causal_threshold_sensitivity.png", dpi=150)
plt.close()
print("Saved: causal_threshold_sensitivity.png")

# %% 2. New rule what-if scenarios
print("\n── New Rule Impact Simulations ────────────────────────────")
rules = [
    ("Flag all wire transfers > $50K from PA/KY",    0.08, 25),
    ("Flag all cash deposits near $10K (structuring)", 0.15, 35),
    ("Flag rapid succession transfers (< 1 hr apart)", 0.12, 20),
    ("Flag all TBML trade payment anomalies",           0.06, 18),
]

rule_results = []
for rule_name, frac, reduction in rules:
    r = analyzer.what_if_rule(rule_name, frac, reduction)
    rule_results.append({
        "rule": rule_name[:45],
        "new_alerts": r.counterfactual_alerts,
        "delta": r.delta_alerts,
        "delta_pct": r.delta_pct,
    })
    print(f"  {rule_name[:50]:<50}  +{r.delta_alerts:,} alerts  ({r.delta_pct:+.1f}%)")

rule_df = pd.DataFrame(rule_results)
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(rule_df["rule"], rule_df["delta"], color="#E53935", alpha=0.85)
ax.set_xlabel("Additional Alerts Generated")
ax.set_title("Estimated Alert Increase by New Detection Rule")
ax.grid(axis="x", alpha=0.3)
for bar, val in zip(bars, rule_df["delta"]):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
            f"+{val:,}", va="center", fontsize=10)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "causal_new_rules.png", dpi=150)
plt.close()
print("Saved: causal_new_rules.png")

# %% 3. DiD policy effect estimation
print("\n── Difference-in-Differences Policy Effect ────────────────")
effect = analyzer.estimate_rule_effect(
    rule_name="2023 Enhanced Structuring Rule",
    policy_date="2023-07-01",
    treated_column="structuring_flag",
    outcome_column="is_suspicious",
)
print(f"  Estimator   : {effect.estimator}")
print(f"  ATE         : {effect.ate:.4f}")
print(f"  95% CI      : [{effect.ate_lower:.4f}, {effect.ate_upper:.4f}]")
print(f"  Relative    : {effect.relative_effect_pct:.1f}%")
print(f"  Interpretation: {effect.interpretation}")

# Save to JSON
(REPORTS_DIR / "causal_did_result.json").write_text(
    json.dumps(asdict(effect), indent=2)
)

# %% 4. Cost model
print("\n── SAR Cost Model Sensitivity ─────────────────────────────")
from utils.metrics import sar_cost_model

base_tp, base_fp, base_fn = 800, 2000, 150
base_cost = sar_cost_model(base_tp, base_fp, base_fn)
print(f"  Baseline cost: USD {base_cost['total_cost_usd']:,.0f}")
print(f"    Missed laundering: USD {base_cost['missed_laundering_cost_usd']:,.0f}")
print(f"    False alerts:      USD {base_cost['false_alert_cost_usd']:,.0f}")

# After threshold lowering (more recalls, more FP)
improved_cost = sar_cost_model(tp=900, fp=2800, fn=50)
print(f"\n  After improving recall:")
print(f"    Total cost: USD {improved_cost['total_cost_usd']:,.0f}  (delta: {improved_cost['total_cost_usd']-base_cost['total_cost_usd']:+,.0f})")

print("\n✓ Causal analysis complete. Figures saved to reports/")
