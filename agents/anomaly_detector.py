"""
agents/anomaly_detector.py
==========================
AML Anomaly Detector Agent – trains and evaluates the VAE on transaction data.

Responsibilities
----------------
1. Load & preprocess transaction feature matrix.
2. Train VAE on normal transactions only (unsupervised).
3. Select anomaly threshold (95th percentile of training scores).
4. Evaluate on held-out test set; log precision / recall / F1.
5. Persist model weights + scaler to disk.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from models.vae import VAE

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
REPORTS_DIR = Path("reports")

BATCH_SIZE = 1024
EPOCHS = 50
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
LATENT_DIM = 16
HIDDEN_DIMS = (256, 128)
BETA = 1.5
THRESHOLD_PERCENTILE = 95
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLS = [
    "amount_usd",
    "hour_of_day",
    "day_of_week",
    "is_cross_border",
    "round_amount",
    "rapid_movement",
    "structuring_flag",
    "log_amount",
    "amount_z_score_sender",  # requires per-sender stats, computed in features
]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def build_feature_matrix(txn_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Derive features from raw transaction DataFrame.

    Returns
    -------
    features : DataFrame
    labels   : ndarray[bool]  (True = suspicious)
    """
    df = txn_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour_of_day"] = df["timestamp"].dt.hour / 23.0
    df["day_of_week"] = df["timestamp"].dt.dayofweek / 6.0
    df["is_cross_border"] = (df["country_origin"] != df["country_dest"]).astype(float)
    df["round_amount"] = df["round_amount"].astype(float)
    df["rapid_movement"] = df["rapid_movement"].astype(float)
    df["structuring_flag"] = df["structuring_flag"].astype(float)
    df["log_amount"] = np.log1p(df["amount_usd"])

    # Per-sender z-score of transaction amount
    sender_stats = df.groupby("sender_id")["amount_usd"].agg(["mean", "std"])
    df = df.join(sender_stats, on="sender_id", rsuffix="_sender")
    df["amount_z_score_sender"] = (
        (df["amount_usd"] - df["mean"]) / df["std"].clip(lower=1e-6)
    ).clip(-5, 5)

    feat_cols = [
        "amount_usd", "log_amount", "hour_of_day", "day_of_week",
        "is_cross_border", "round_amount", "rapid_movement",
        "structuring_flag", "amount_z_score_sender",
    ]
    features = df[feat_cols].fillna(0.0)
    labels = df["is_suspicious"].values.astype(bool)
    return features, labels


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_epoch(
    model: VAE,
    loader: DataLoader,
    optimizer: optim.Optimizer,
) -> Dict[str, float]:
    model.train()
    total_loss = recon_sum = kl_sum = 0.0
    for (batch,) in loader:
        batch = batch.to(DEVICE)
        x_hat, mu, log_var, _ = model(batch)
        loss, recon, kl = model.loss(batch, x_hat, mu, log_var)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        recon_sum += recon.item()
        kl_sum += kl.item()
    n = len(loader)
    return {"loss": total_loss / n, "recon": recon_sum / n, "kl": kl_sum / n}


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------
class AnomalyDetectorAgent:
    """End-to-end VAE training and inference agent."""

    def __init__(self) -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[VAE] = None
        self.threshold: float = 0.0

    # ------------------------------------------------------------------
    def run(self) -> None:
        log.info("=== AnomalyDetectorAgent starting ===")

        # 1. Load data
        txn_df = pd.read_parquet(DATA_DIR / "transactions.parquet")
        log.info("Loaded %d transactions", len(txn_df))

        # 2. Features
        features, labels = build_feature_matrix(txn_df)
        X = features.values.astype(np.float32)

        # 3. Train / test split (stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.15, random_state=42, stratify=labels
        )

        # 4. Fit scaler on NORMAL training samples only
        normal_mask = ~y_train
        self.scaler = StandardScaler()
        X_train_normal = self.scaler.fit_transform(X_train[normal_mask])
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 5. Build model
        input_dim = X.shape[1]
        self.model = VAE(
            input_dim=input_dim,
            latent_dim=LATENT_DIM,
            hidden_dims=HIDDEN_DIMS,
            beta=BETA,
        ).to(DEVICE)
        log.info("VAE parameters: %d", sum(p.numel() for p in self.model.parameters()))

        # 6. DataLoaders – train on NORMAL only
        train_tensor = torch.from_numpy(X_train_normal)
        train_loader = DataLoader(
            TensorDataset(train_tensor),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=(DEVICE.type == "cuda"),
        )

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        # 7. Train
        history = []
        best_loss = float("inf")
        patience, patience_count = 7, 0

        for epoch in range(1, EPOCHS + 1):
            metrics = train_epoch(self.model, train_loader, optimizer)
            scheduler.step()
            history.append(metrics)
            if epoch % 5 == 0 or epoch == 1:
                log.info(
                    "Epoch %3d/%d  loss=%.6f  recon=%.6f  kl=%.6f",
                    epoch, EPOCHS, metrics["loss"], metrics["recon"], metrics["kl"],
                )
            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                patience_count = 0
                self._save_model()
            else:
                patience_count += 1
                if patience_count >= patience:
                    log.info("Early stopping at epoch %d", epoch)
                    break

        # 8. Load best checkpoint
        self._load_model(MODEL_DIR / "vae_model.pth")

        # 9. Threshold on training scores
        train_all_tensor = torch.from_numpy(X_train_scaled).to(DEVICE)
        train_scores = self.model.anomaly_score(train_all_tensor).cpu().numpy()
        normal_scores = train_scores[~y_train]
        self.threshold = float(np.percentile(normal_scores, THRESHOLD_PERCENTILE))
        log.info("Anomaly threshold (p%d): %.6f", THRESHOLD_PERCENTILE, self.threshold)

        # 10. Evaluate on test set
        test_tensor = torch.from_numpy(X_test_scaled).to(DEVICE)
        test_scores = self.model.anomaly_score(test_tensor).cpu().numpy()
        y_pred = (test_scores >= self.threshold).astype(int)

        report = classification_report(y_test.astype(int), y_pred, digits=4)
        cm = confusion_matrix(y_test.astype(int), y_pred)
        log.info("\nClassification Report:\n%s", report)

        # 11. Persist artefacts
        self._save_scaler()
        meta = {
            "threshold": self.threshold,
            "input_dim": int(input_dim),
            "latent_dim": LATENT_DIM,
            "hidden_dims": list(HIDDEN_DIMS),
            "beta": BETA,
            "feature_cols": list(features.columns),
            "best_train_loss": best_loss,
            "confusion_matrix": cm.tolist(),
        }
        (MODEL_DIR / "vae_meta.json").write_text(json.dumps(meta, indent=2))

        # Save training history
        pd.DataFrame(history).to_csv(REPORTS_DIR / "vae_training_history.csv", index=False)
        log.info("=== AnomalyDetectorAgent finished ===")

    # ------------------------------------------------------------------
    def _save_model(self) -> None:
        path = MODEL_DIR / "vae_model.pth"
        torch.save(self.model.state_dict(), path)

    def _load_model(self, path: Path) -> None:
        state = torch.load(path, map_location=DEVICE)
        self.model.load_state_dict(state)
        log.info("Loaded best model from %s", path)

    def _save_scaler(self) -> None:
        path = MODEL_DIR / "scaler.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("Scaler saved to %s", path)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    agent = AnomalyDetectorAgent()
    agent.run()
