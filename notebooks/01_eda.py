"""
notebooks/01_eda.py
====================
Exploratory Data Analysis – AML Transaction Dataset.

Run as plain Python:  python notebooks/01_eda.py
Or convert to notebook:  jupytext --to notebook notebooks/01_eda.py
"""
# %% [markdown]
# # AML EDA – Synthetic Transaction Dataset
# Explores statistical properties of the generated dataset,
# class imbalance, feature correlations, and typology distributions.

# %% Setup
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from features.feature_engineering import derive_transaction_features

sns.set_theme(style="whitegrid", palette="muted")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# %% Load data
print("Loading data…")
txn_df = pd.read_parquet("data/transactions.parquet")
cust_df = pd.read_parquet("data/customers.parquet")
txn_df["timestamp"] = pd.to_datetime(txn_df["timestamp"])
print(f"Transactions: {len(txn_df):,}  |  Customers: {len(cust_df):,}")

# %% Class distribution
print("\n── Class Distribution ──")
dist = txn_df["is_suspicious"].value_counts(normalize=True)
print(dist)

fig, ax = plt.subplots(figsize=(5, 3))
dist.plot.bar(ax=ax, color=["#2196F3", "#F44336"])
ax.set_xticklabels(["Normal", "Suspicious"], rotation=0)
ax.set_title("Transaction Class Distribution")
ax.set_ylabel("Proportion")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "eda_class_dist.png", dpi=120)
plt.close()

# %% Typology breakdown
print("\n── Typology Counts ──")
print(txn_df["label"].value_counts())

# %% Amount distributions by class
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, label, color in zip(axes, ["normal", "suspicious"], ["#2196F3", "#F44336"]):
    is_susp = label == "suspicious"
    data = txn_df[txn_df["is_suspicious"] == is_susp]["amount_usd"]
    data = data.clip(upper=data.quantile(0.99))
    ax.hist(data, bins=60, color=color, alpha=0.8, edgecolor="white")
    ax.set_title(f"Amount Distribution – {label.capitalize()}")
    ax.set_xlabel("USD Amount")
    ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "eda_amount_distributions.png", dpi=120)
plt.close()

# %% Feature correlation matrix
print("\nComputing feature matrix…")
feat_df = derive_transaction_features(txn_df.head(20_000))
feat_df["is_suspicious"] = txn_df.head(20_000)["is_suspicious"].astype(int).values

corr = feat_df.corr()
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
    linewidths=0.5, ax=ax, vmin=-1, vmax=1
)
ax.set_title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "eda_correlation_heatmap.png", dpi=120)
plt.close()

# %% Time-of-day distribution of suspicious activity
txn_df["hour"] = txn_df["timestamp"].dt.hour
hour_dist = txn_df.groupby(["hour", "is_suspicious"]).size().unstack(fill_value=0)
hour_dist.columns = ["Normal", "Suspicious"]
hour_dist_norm = hour_dist.div(hour_dist.sum())

fig, ax = plt.subplots(figsize=(10, 4))
hour_dist_norm.plot(kind="bar", ax=ax, color=["#2196F3", "#F44336"], alpha=0.85)
ax.set_title("Hour-of-Day Distribution by Class")
ax.set_xlabel("Hour (UTC)")
ax.set_ylabel("Proportion")
ax.set_xticklabels(hour_dist_norm.index, rotation=45)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "eda_hour_distribution.png", dpi=120)
plt.close()

# %% Jurisdiction risk vs suspicion rate
if "jurisdiction_risk" in cust_df.columns:
    merged = txn_df.merge(cust_df[["customer_id", "jurisdiction_risk"]], left_on="sender_id", right_on="customer_id", how="left")
    susp_by_jur = merged.groupby("jurisdiction_risk")["is_suspicious"].mean().sort_values()
    print("\n── Suspicion Rate by Jurisdiction Risk ──")
    print(susp_by_jur)

print("\nEDA complete. Figures saved to reports/")
