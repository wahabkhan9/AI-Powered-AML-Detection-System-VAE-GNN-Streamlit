"""
agents/gan_trainer.py
=====================
GAN Training Agent – trains the TransactionGAN on normal transaction features.

The trained Discriminator is used as a second anomaly signal alongside the VAE.
The Generator is used to augment the minority (suspicious) class for GNN training.

Output
------
models/gan_generator.pth
models/gan_discriminator.pth
models/gan_meta.json
data/synthetic_suspicious.parquet   (augmented suspicious samples)
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from features.feature_engineering import derive_transaction_features
from models.gan import TransactionGAN

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR   = Path("data")
MODEL_DIR  = Path("models")

BATCH_SIZE   = 512
EPOCHS       = 60
LR_G         = 2e-4
LR_D         = 1e-4
LATENT_DIM   = 32
N_CRITIC     = 2          # discriminator steps per generator step
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SYNTHETIC  = 5_000      # synthetic suspicious samples to generate


# ---------------------------------------------------------------------------
class GANTrainerAgent:
    """Train TransactionGAN and produce synthetic suspicious samples."""

    def run(self) -> None:
        log.info("=== GANTrainerAgent starting ===")
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # ── Load data ────────────────────────────────────────────────────────
        txn_df = pd.read_parquet(DATA_DIR / "transactions.parquet")
        txn_df["timestamp"] = pd.to_datetime(txn_df["timestamp"])

        # ── Features ─────────────────────────────────────────────────────────
        feat_df = derive_transaction_features(txn_df)

        # Load scaler from VAE training (ensures feature alignment)
        scaler_path = MODEL_DIR / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            X_all = scaler.transform(feat_df.values).astype(np.float32)
        else:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_all = scaler.fit_transform(feat_df.values).astype(np.float32)

        # Train GAN only on NORMAL transactions
        normal_mask = ~txn_df["is_suspicious"].values
        X_normal = X_all[normal_mask]

        feature_dim = X_normal.shape[1]
        log.info("Training on %d normal transactions, feature_dim=%d", len(X_normal), feature_dim)

        # ── Model ────────────────────────────────────────────────────────────
        gan = TransactionGAN(
            feature_dim=feature_dim,
            latent_dim=LATENT_DIM,
        ).to(DEVICE)

        opt_g = optim.Adam(gan.generator.parameters(),     lr=LR_G, betas=(0.5, 0.999))
        opt_d = optim.Adam(gan.discriminator.parameters(), lr=LR_D, betas=(0.5, 0.999))

        train_tensor = torch.tensor(X_normal)
        loader = DataLoader(
            TensorDataset(train_tensor),
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True,
        )

        history: list[Dict] = []

        # ── Training loop ────────────────────────────────────────────────────
        for epoch in range(1, EPOCHS + 1):
            d_losses, g_losses = [], []

            for step, (real_batch,) in enumerate(loader):
                real_batch = real_batch.to(DEVICE)
                z = torch.randn(real_batch.size(0), LATENT_DIM, device=DEVICE)
                fake_batch = gan.generator(z)

                # ── Discriminator step ────────────────────────────────────
                d_loss = gan.discriminator_loss(real_batch, fake_batch)
                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(gan.discriminator.parameters(), 1.0)
                opt_d.step()
                d_losses.append(d_loss.item())

                # ── Generator step (every N_CRITIC discriminator steps) ───
                if step % N_CRITIC == 0:
                    z2 = torch.randn(real_batch.size(0), LATENT_DIM, device=DEVICE)
                    fake2 = gan.generator(z2)
                    g_loss = gan.generator_loss(fake2)
                    opt_g.zero_grad(set_to_none=True)
                    g_loss.backward()
                    torch.nn.utils.clip_grad_norm_(gan.generator.parameters(), 1.0)
                    opt_g.step()
                    g_losses.append(g_loss.item())

            avg_d = float(np.mean(d_losses))
            avg_g = float(np.mean(g_losses)) if g_losses else 0.0
            history.append({"epoch": epoch, "d_loss": avg_d, "g_loss": avg_g})

            if epoch % 10 == 0 or epoch == 1:
                log.info("Epoch %3d/%d  D_loss=%.4f  G_loss=%.4f", epoch, EPOCHS, avg_d, avg_g)

        # ── Save models ──────────────────────────────────────────────────────
        torch.save(gan.generator.state_dict(),     MODEL_DIR / "gan_generator.pth")
        torch.save(gan.discriminator.state_dict(), MODEL_DIR / "gan_discriminator.pth")

        meta = {
            "feature_dim": feature_dim,
            "latent_dim": LATENT_DIM,
            "epochs": EPOCHS,
        }
        (MODEL_DIR / "gan_meta.json").write_text(json.dumps(meta, indent=2))
        pd.DataFrame(history).to_csv(Path("reports") / "gan_training_history.csv", index=False)

        # ── Generate synthetic suspicious samples for augmentation ───────────
        log.info("Generating %d synthetic suspicious samples …", N_SYNTHETIC)
        synthetic_scaled = gan.generate(N_SYNTHETIC, device=DEVICE).cpu().numpy()
        synthetic_original = scaler.inverse_transform(synthetic_scaled)

        feat_cols = list(feat_df.columns)
        synth_df = pd.DataFrame(synthetic_original, columns=feat_cols)
        synth_df["is_suspicious"] = True
        synth_df["label"] = "gan_synthetic"
        synth_df.to_parquet(DATA_DIR / "synthetic_suspicious.parquet", index=False)

        log.info("Synthetic suspicious data saved → data/synthetic_suspicious.parquet")
        log.info("=== GANTrainerAgent finished ===")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    GANTrainerAgent().run()