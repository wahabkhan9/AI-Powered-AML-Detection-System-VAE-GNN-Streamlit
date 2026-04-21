"""
models/gan.py
=============
Generative Adversarial Network (GAN) for financial transaction anomaly detection.

Two roles:
  1. Generator  – produces synthetic suspicious transactions for data augmentation,
                  helping address class imbalance in training.
  2. Discriminator – distinguishes real-normal from generated/anomalous transactions;
                     its confidence score becomes a second anomaly signal.

Architecture
------------
Generator  : latent_dim → FC(128) → FC(256) → FC(feature_dim)  [tanh output]
Discriminator : feature_dim → FC(256) → FC(128) → FC(1)          [sigmoid output]

Training
--------
Standard min-max GAN loss:
  D: max  E[log D(x)] + E[log(1 - D(G(z)))]
  G: min  E[log(1 - D(G(z)))]   (or non-saturating: max E[log D(G(z))])
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
class Generator(nn.Module):
    """Maps a latent noise vector z ∈ R^latent_dim to a synthetic transaction."""

    def __init__(self, latent_dim: int, feature_dim: int, hidden_dims: Tuple[int, ...] = (128, 256)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = latent_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_dim = h
        layers += [nn.Linear(in_dim, feature_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------
class Discriminator(nn.Module):
    """
    Binary classifier: real-normal (1) vs generated/suspicious (0).
    The output probability P(real) serves as an inverse anomaly score:
    low P(real) → high anomaly likelihood.
    """

    def __init__(self, feature_dim: int, hidden_dims: Tuple[int, ...] = (256, 128)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = feature_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))   # raw logit; use BCEWithLogitsLoss
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (shape [N, 1])."""
        return self.net(x)

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Anomaly score = 1 - P(real).
        Range [0, 1].  Higher → more anomalous.
        """
        self.eval()
        logits = self.forward(x).squeeze(1)
        p_real = torch.sigmoid(logits)
        return 1.0 - p_real


# ---------------------------------------------------------------------------
# GAN wrapper
# ---------------------------------------------------------------------------
class TransactionGAN(nn.Module):
    """
    Wraps Generator + Discriminator with training utilities.

    Parameters
    ----------
    feature_dim  : number of input features (must match VAE feature_dim)
    latent_dim   : noise vector dimensionality
    hidden_dims_g: generator hidden layer sizes
    hidden_dims_d: discriminator hidden layer sizes
    """

    def __init__(
        self,
        feature_dim: int,
        latent_dim: int = 32,
        hidden_dims_g: Tuple[int, ...] = (128, 256),
        hidden_dims_d: Tuple[int, ...] = (256, 128),
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim, feature_dim, hidden_dims_g)
        self.discriminator = Discriminator(feature_dim, hidden_dims_d)
        log.debug(
            "TransactionGAN  feature_dim=%d  latent_dim=%d",
            feature_dim, latent_dim,
        )

    # ------------------------------------------------------------------
    def generate(self, n: int, device: torch.device | None = None) -> torch.Tensor:
        """Sample n synthetic transactions."""
        device = device or next(self.generator.parameters()).device
        z = torch.randn(n, self.latent_dim, device=device)
        self.generator.eval()
        with torch.no_grad():
            return self.generator(z)

    # ------------------------------------------------------------------
    def discriminator_loss(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
    ) -> torch.Tensor:
        """
        Standard non-saturating GAN discriminator loss with label smoothing.
        Real labels → 0.9 (smoothed), Fake labels → 0.0.
        """
        real_labels = torch.full((real.size(0), 1), 0.9, device=real.device)
        fake_labels = torch.zeros(fake.size(0), 1, device=fake.device)

        real_logits = self.discriminator(real)
        fake_logits = self.discriminator(fake.detach())

        d_real = F.binary_cross_entropy_with_logits(real_logits, real_labels)
        d_fake = F.binary_cross_entropy_with_logits(fake_logits, fake_labels)
        return (d_real + d_fake) * 0.5

    # ------------------------------------------------------------------
    def generator_loss(self, fake: torch.Tensor) -> torch.Tensor:
        """Non-saturating generator loss: max log D(G(z))."""
        real_labels = torch.ones(fake.size(0), 1, device=fake.device)
        fake_logits = self.discriminator(fake)
        return F.binary_cross_entropy_with_logits(fake_logits, real_labels)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-sample anomaly scores in [0, 1]."""
        return self.discriminator.anomaly_score(x)
