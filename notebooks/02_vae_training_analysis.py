"""
notebooks/02_vae_training_analysis.py
======================================
Deep-dive analysis of VAE training results:
  - Loss curve visualisation
  - Latent space t-SNE projection
  - Reconstruction error distribution (normal vs suspicious)
  - Threshold sensitivity analysis
"""
# %% Setup
from __future__ import annotations
import json
import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from models.vae import VAE
from features.feature_engineering import derive_transaction_features
from utils.metrics import compute_classification_metrics

REPORTS_DIR = Path("reports")
MODEL_DIR = Path("models")
REPORTS_DIR.mkdir(exist_ok=True)

# %% Load model
meta = json.loads((MODEL_DIR / "vae_meta.json").read_text())
model = VAE(
    input_dim=meta["input_dim"],
    latent_dim=meta["latent_dim"],
    hidden_dims=tuple(meta["hidden_dims"]),
    beta=meta["beta"],
)
model.load_state_dict(torch.load(MODEL_DIR / "vae_model.pth", map_location="cpu"))
model.eval()
print(f"Model loaded  latent_dim={meta['latent_dim']}  threshold={meta['threshold']:.6f}")

with open(MODEL_DIR / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# %% Load data
txn_df = pd.read_parquet("data/transactions.parquet")
txn_df["timestamp"] = pd.to_datetime(txn_df["timestamp"])

# Sample for analysis (stratified)
normal_sample = txn_df[~txn_df["is_suspicious"]].sample(5000, random_state=42)
susp_sample = txn_df[txn_df["is_suspicious"]].sample(
    min(1000, txn_df["is_suspicious"].sum()), random_state=42
)
sample = pd.concat([normal_sample, susp_sample]).reset_index(drop=True)

features = derive_transaction_features(sample)
X = scaler.transform(features.values).astype(np.float32)
y = sample["is_suspicious"].values
tensor = torch.tensor(X)

# %% Reconstruction error distribution
with torch.no_grad():
    scores = model.anomaly_score(tensor).numpy()

threshold = meta["threshold"]
y_pred = (scores >= threshold).astype(int)
metrics = compute_classification_metrics(y.astype(int), y_pred, scores)
print("\n── Classification Metrics ──")
for k, v in metrics.items():
    print(f"  {k}: {v}")

fig, ax = plt.subplots(figsize=(10, 5))
bins = np.linspace(0, np.percentile(scores, 99.5), 80)
ax.hist(scores[~y.astype(bool)], bins=bins, alpha=0.65, color="#2196F3", label="Normal", density=True)
ax.hist(scores[y.astype(bool)], bins=bins, alpha=0.65, color="#F44336", label="Suspicious", density=True)
ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"Threshold = {threshold:.5f}")
ax.set_xlabel("Reconstruction Error (Anomaly Score)")
ax.set_ylabel("Density")
ax.set_title("VAE Anomaly Score Distribution")
ax.legend()
plt.tight_layout()
plt.savefig(REPORTS_DIR / "vae_score_distribution.png", dpi=150)
plt.close()
print("Saved: vae_score_distribution.png")

# %% Threshold sensitivity
thresholds = np.linspace(scores.min(), np.percentile(scores, 99), 200)
recalls, precisions, f1s = [], [], []
for t in thresholds:
    yp = (scores >= t).astype(int)
    tp = ((yp == 1) & (y == 1)).sum()
    fp = ((yp == 1) & (y == 0)).sum()
    fn = ((yp == 0) & (y == 1)).sum()
    r = tp / max(tp + fn, 1)
    p = tp / max(tp + fp, 1)
    f = 2 * p * r / max(p + r, 1e-9)
    recalls.append(r); precisions.append(p); f1s.append(f)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(thresholds, recalls, color="#4CAF50", linewidth=1.8, label="Recall")
ax.plot(thresholds, precisions, color="#2196F3", linewidth=1.8, label="Precision")
ax.plot(thresholds, f1s, color="#FF9800", linewidth=1.8, label="F1")
ax.axvline(threshold, color="black", linestyle="--", linewidth=1.4, label=f"Selected threshold")
ax.set_xlabel("Threshold")
ax.set_ylabel("Score")
ax.set_title("Threshold Sensitivity Analysis")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "vae_threshold_sensitivity.png", dpi=150)
plt.close()
print("Saved: vae_threshold_sensitivity.png")

# %% Latent space t-SNE
print("\nRunning t-SNE on latent space (this may take ~1 min)…")
with torch.no_grad():
    mu, _ = model.encode(tensor)
    latent = mu.numpy()

tsne = TSNE(n_components=2, perplexity=40, random_state=42, n_iter=800)
z_2d = tsne.fit_transform(latent)

fig, ax = plt.subplots(figsize=(9, 7))
for label, color, marker in [(0, "#2196F3", "o"), (1, "#F44336", "x")]:
    mask = y == label
    name = "Normal" if label == 0 else "Suspicious"
    ax.scatter(z_2d[mask, 0], z_2d[mask, 1], c=color, s=6, alpha=0.4,
               marker=marker, label=name)
ax.set_title("VAE Latent Space (t-SNE projection)")
ax.legend(markerscale=3)
ax.axis("off")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "vae_latent_tsne.png", dpi=150)
plt.close()
print("Saved: vae_latent_tsne.png")

# %% Training history
hist_path = REPORTS_DIR / "vae_training_history.csv"
if hist_path.exists():
    hist = pd.read_csv(hist_path)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, color in zip(axes, ["loss", "recon", "kl"], ["#9C27B0", "#2196F3", "#FF9800"]):
        ax.plot(hist[col], color=color, linewidth=1.5)
        ax.set_title(f"Train {col.upper()}")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    plt.suptitle("VAE Training History", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "vae_training_history.png", dpi=150)
    plt.close()
    print("Saved: vae_training_history.png")

print("\nAnalysis complete.")
