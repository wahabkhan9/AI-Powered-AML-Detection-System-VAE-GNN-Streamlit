"""
agents/anomaly_detector.py
==========================
AML Anomaly Detector Agent – trains the VAE, fuses GAN scores,
and runs SHAP explainability. All three are FULLY integrated.

Steps
-----
1. Load & preprocess transaction feature matrix.
2. Train VAE on NORMAL transactions only (unsupervised).
3. Load trained GAN discriminator (if available) and fuse anomaly scores:
      combined = 0.6 * vae_norm + 0.4 * gan_score
4. Select threshold on combined score (95th percentile of normal samples).
5. Run SHAP KernelExplainer on flagged transactions → feature importance CSV.
6. Evaluate with classification report + confusion matrix.
7. Persist: vae_model.pth, scaler.pkl, vae_meta.json, vae_alerts.parquet,
            shap_feature_importance.csv, shap_values.parquet.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from features.feature_engineering import derive_transaction_features
from models.vae import VAE

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR    = Path("data")
MODEL_DIR   = Path("models")
REPORTS_DIR = Path("reports")

BATCH_SIZE           = 1024
EPOCHS               = 50
LEARNING_RATE        = 3e-4
WEIGHT_DECAY         = 1e-5
LATENT_DIM           = 16
HIDDEN_DIMS          = (256, 128)
BETA                 = 1.5
THRESHOLD_PERCENTILE = 95
DEVICE               = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Score fusion weights (VAE is primary; GAN is secondary signal)
VAE_WEIGHT = 0.6
GAN_WEIGHT = 0.4


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------
def _train_epoch(
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
        recon_sum  += recon.item()
        kl_sum     += kl.item()
    n = max(len(loader), 1)
    return {"loss": total_loss / n, "recon": recon_sum / n, "kl": kl_sum / n}


# ---------------------------------------------------------------------------
# GAN score fusion
# ---------------------------------------------------------------------------
def _load_gan_discriminator(feature_dim: int):
    """
    Load trained GAN discriminator weights.
    Returns the discriminator nn.Module, or None if not yet trained.
    """
    gan_pth  = MODEL_DIR / "gan_discriminator.pth"
    gan_meta = MODEL_DIR / "gan_meta.json"

    if not gan_pth.exists() or not gan_meta.exists():
        log.info(
            "GAN discriminator not found. Run 'python main.py pipeline --steps train_gan' "
            "first for score fusion. Proceeding with VAE-only scoring."
        )
        return None

    try:
        from models.gan import TransactionGAN
        meta = json.loads(gan_meta.read_text())
        gan = TransactionGAN(
            feature_dim=feature_dim,
            latent_dim=meta.get("latent_dim", 32),
        )
        gan.discriminator.load_state_dict(
            torch.load(gan_pth, map_location=DEVICE)
        )
        gan.discriminator.eval()
        gan.discriminator.to(DEVICE)
        log.info(
            "GAN discriminator loaded. Score fusion ACTIVE: "
            "combined = %.0f%% VAE + %.0f%% GAN",
            VAE_WEIGHT * 100, GAN_WEIGHT * 100,
        )
        return gan.discriminator
    except Exception as exc:
        log.warning("Failed to load GAN discriminator (%s). VAE-only.", exc)
        return None


def _fuse_scores(
    vae_scores: np.ndarray,
    gan_discriminator,
    tensor: torch.Tensor,
) -> np.ndarray:
    """
    Fuse VAE reconstruction error with GAN anomaly score.
    If GAN is not available, returns normalized VAE scores unchanged.
    """
    # Normalize VAE scores to [0, 1]
    vae_max = vae_scores.max() + 1e-9
    vae_norm = vae_scores / vae_max

    if gan_discriminator is None:
        return vae_norm

    with torch.no_grad():
        # GAN anomaly score = 1 - P(real) ∈ [0, 1]
        gan_scores = gan_discriminator.anomaly_score(tensor).cpu().numpy()

    combined = VAE_WEIGHT * vae_norm + GAN_WEIGHT * gan_scores
    log.debug(
        "Score fusion stats – VAE norm: mean=%.4f  GAN: mean=%.4f  Combined: mean=%.4f",
        vae_norm.mean(), gan_scores.mean(), combined.mean(),
    )
    return combined


# ---------------------------------------------------------------------------
# SHAP explainability – fully wired into the training run
# ---------------------------------------------------------------------------
def _run_shap_analysis(
    model: VAE,
    X_train_scaled: np.ndarray,
    X_flagged_scaled: np.ndarray,
    y_train: np.ndarray,
    feature_cols: List[str],
) -> Optional[pd.DataFrame]:
    """
    Run SHAP KernelExplainer on flagged transactions to explain WHY
    each was scored as anomalous.

    Outputs
    -------
    reports/shap_feature_importance.csv  – global mean |SHAP| per feature
    reports/shap_values.parquet          – per-sample SHAP matrix
    reports/shap_summary.png             – beeswarm plot (if matplotlib available)

    Returns global importance DataFrame, or None if shap not installed.
    """
    try:
        import shap
    except ImportError:
        log.warning(
            "shap not installed – SHAP analysis skipped. "
            "Install with: pip install shap"
        )
        return None

    if len(X_flagged_scaled) == 0:
        log.warning("No flagged samples to explain.")
        return None

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Wrap VAE as a plain numpy → numpy callable
    model.eval()
    def _score_fn(X_np: np.ndarray) -> np.ndarray:
        t = torch.tensor(X_np.astype(np.float32), device=DEVICE)
        with torch.no_grad():
            return model.anomaly_score(t).cpu().numpy()

    # Background dataset: k-means medoids of 200 NORMAL training samples
    X_normal = X_train_scaled[~y_train.astype(bool)]
    n_bg = min(200, len(X_normal))
    log.info("Building SHAP background from %d normal samples (k-means) …", n_bg)
    background = shap.kmeans(X_normal, n_bg)

    explainer = shap.KernelExplainer(_score_fn, background)

    # Explain up to 300 flagged samples (KernelExplainer scales as O(n * nsamples))
    n_explain = min(300, len(X_flagged_scaled))
    X_explain = X_flagged_scaled[:n_explain]
    log.info("Running SHAP on %d flagged transactions (nsamples=100) …", n_explain)
    shap_values = explainer.shap_values(X_explain, nsamples=100, silent=True)

    # Global importance: mean absolute SHAP per feature
    importance = np.abs(shap_values).mean(axis=0)
    importance_df = (
        pd.DataFrame({"feature": feature_cols, "shap_importance": importance})
        .sort_values("shap_importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(REPORTS_DIR / "shap_feature_importance.csv", index=False)

    # Save raw SHAP matrix for dashboard waterfall plots
    shap_df = pd.DataFrame(shap_values, columns=feature_cols)
    shap_df.to_parquet(REPORTS_DIR / "shap_values.parquet", index=False)

    # Try to save summary beeswarm plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        shap.summary_plot(
            shap_values, X_explain,
            feature_names=feature_cols,
            show=False,
            max_display=10,
        )
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "shap_summary.png", dpi=120, bbox_inches="tight")
        plt.close()
        log.info("SHAP beeswarm plot saved → reports/shap_summary.png")
    except Exception as exc:
        log.debug("Could not save SHAP plot: %s", exc)

    log.info(
        "SHAP complete. Top-3 features: %s",
        importance_df["feature"].head(3).tolist(),
    )
    return importance_df


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------
class AnomalyDetectorAgent:
    """
    Analyst Agent – VAE anomaly detection with GAN score fusion and SHAP explainability.
    All three subsystems are fully wired: train → fuse → explain → persist.
    """

    def __init__(self) -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        self.scaler:    Optional[StandardScaler] = None
        self.model:     Optional[VAE] = None
        self.threshold: float = 0.0

    # ------------------------------------------------------------------
    def run(self) -> None:
        log.info("=== AnomalyDetectorAgent starting ===")

        # 1. Load data
        txn_df = pd.read_parquet(DATA_DIR / "transactions.parquet")
        txn_df["timestamp"] = pd.to_datetime(txn_df["timestamp"])
        log.info("Loaded %d transactions", len(txn_df))

        # 2. Feature engineering
        features = derive_transaction_features(txn_df)
        feature_cols = list(features.columns)
        X = features.values.astype(np.float32)
        labels = txn_df["is_suspicious"].values.astype(bool)

        # 3. Stratified train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.15, random_state=42, stratify=labels
        )

        # 4. Fit scaler on NORMAL train samples only
        normal_mask = ~y_train
        self.scaler = StandardScaler()
        X_train_normal_scaled = self.scaler.fit_transform(X_train[normal_mask])
        X_train_scaled        = self.scaler.transform(X_train)
        X_test_scaled         = self.scaler.transform(X_test)

        # 5. Build VAE
        input_dim = X.shape[1]
        self.model = VAE(
            input_dim=input_dim, latent_dim=LATENT_DIM,
            hidden_dims=HIDDEN_DIMS, beta=BETA,
        ).to(DEVICE)
        log.info("VAE parameters: %d", sum(p.numel() for p in self.model.parameters()))

        # 6. DataLoader (normal samples only)
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train_normal_scaled)),
            batch_size=BATCH_SIZE, shuffle=True,
            num_workers=0, pin_memory=(DEVICE.type == "cuda"),
        )
        optimizer  = optim.AdamW(self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        # 7. Training loop with early stopping
        history, best_loss, patience_count = [], float("inf"), 0
        PATIENCE = 7

        for epoch in range(1, EPOCHS + 1):
            m = _train_epoch(self.model, train_loader, optimizer)
            scheduler.step()
            history.append(m)
            if epoch % 5 == 0 or epoch == 1:
                log.info("Epoch %3d/%d  loss=%.6f  recon=%.6f  kl=%.6f",
                         epoch, EPOCHS, m["loss"], m["recon"], m["kl"])
            if m["loss"] < best_loss:
                best_loss = m["loss"]; patience_count = 0; self._save_model()
            else:
                patience_count += 1
                if patience_count >= PATIENCE:
                    log.info("Early stopping at epoch %d", epoch); break

        self._load_model(MODEL_DIR / "vae_model.pth")

        # ── 8. LOAD GAN DISCRIMINATOR (score fusion) ─────────────────────────
        gan_disc = _load_gan_discriminator(input_dim)

        # ── 9. Compute combined scores on train set → threshold ──────────────
        train_tensor = torch.from_numpy(X_train_scaled.astype(np.float32)).to(DEVICE)
        vae_train    = self.model.anomaly_score(train_tensor).cpu().numpy()
        combined_train = _fuse_scores(vae_train, gan_disc, train_tensor)

        normal_combined   = combined_train[normal_mask]
        self.threshold    = float(np.percentile(normal_combined, THRESHOLD_PERCENTILE))
        log.info("Combined threshold (p%d): %.6f", THRESHOLD_PERCENTILE, self.threshold)

        # ── 10. Evaluate on test set ──────────────────────────────────────────
        test_tensor  = torch.from_numpy(X_test_scaled.astype(np.float32)).to(DEVICE)
        vae_test     = self.model.anomaly_score(test_tensor).cpu().numpy()
        combined_test = _fuse_scores(vae_test, gan_disc, test_tensor)

        y_pred = (combined_test >= self.threshold).astype(int)
        report = classification_report(y_test.astype(int), y_pred, digits=4)
        cm     = confusion_matrix(y_test.astype(int), y_pred)
        log.info("\nClassification Report (VAE+GAN fusion):\n%s", report)

        # ── 11. Save vae_alerts.parquet (feeds orchestrator + report writer) ─
        all_scaled = self.scaler.transform(X).astype(np.float32)
        all_tensor = torch.from_numpy(all_scaled).to(DEVICE)
        all_vae    = self.model.anomaly_score(all_tensor).cpu().numpy()
        all_comb   = _fuse_scores(all_vae, gan_disc, all_tensor)

        alerts_df = pd.DataFrame({
            "customer_id":    txn_df["sender_id"].values,
            "vae_score":      all_vae,
            "combined_score": all_comb,
            "is_flagged":     (all_comb >= self.threshold).astype(int),
        })
        # Aggregate to customer level (max score per customer)
        alerts_agg = (
            alerts_df.groupby("customer_id")
            .agg(vae_score=("vae_score", "mean"),
                 combined_score=("combined_score", "mean"),
                 is_flagged=("is_flagged", "max"))
            .reset_index()
        )
        alerts_agg.to_parquet(REPORTS_DIR / "vae_alerts.parquet", index=False)
        n_flagged = int(alerts_agg["is_flagged"].sum())
        log.info("VAE alerts saved (%d customers flagged out of %d).",
                 n_flagged, len(alerts_agg))

        # ── 12. SHAP EXPLAINABILITY (fully wired) ────────────────────────────
        flagged_idx    = all_comb >= self.threshold
        X_flagged_scaled = all_scaled[flagged_idx]
        shap_importance = _run_shap_analysis(
            self.model,
            X_train_scaled,
            X_flagged_scaled,
            y_train,
            feature_cols,
        )
        if shap_importance is not None:
            log.info("SHAP top feature: %s", shap_importance.iloc[0]["feature"])

        # ── 13. Persist artefacts ─────────────────────────────────────────────
        self._save_scaler()
        meta = {
            "threshold":      self.threshold,
            "input_dim":      int(input_dim),
            "latent_dim":     LATENT_DIM,
            "hidden_dims":    list(HIDDEN_DIMS),
            "beta":           BETA,
            "feature_cols":   feature_cols,
            "best_train_loss": best_loss,
            "confusion_matrix": cm.tolist(),
            "gan_fusion_active": gan_disc is not None,
            "vae_weight":     VAE_WEIGHT,
            "gan_weight":     GAN_WEIGHT,
            "shap_computed":  shap_importance is not None,
        }
        (MODEL_DIR / "vae_meta.json").write_text(json.dumps(meta, indent=2))
        pd.DataFrame(history).to_csv(REPORTS_DIR / "vae_training_history.csv", index=False)
        log.info("=== AnomalyDetectorAgent finished ===")

    # ------------------------------------------------------------------
    def _save_model(self) -> None:
        torch.save(self.model.state_dict(), MODEL_DIR / "vae_model.pth")

    def _load_model(self, path: Path) -> None:
        self.model.load_state_dict(torch.load(path, map_location=DEVICE))
        log.info("Best VAE checkpoint loaded ← %s", path)

    def _save_scaler(self) -> None:
        path = MODEL_DIR / "scaler.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("Scaler saved → %s", path)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    AnomalyDetectorAgent().run()