"""
models/vae.py
=============
Variational Autoencoder (VAE) for unsupervised transaction anomaly detection.

Architecture
------------
Encoder  : FC(input_dim → 256) → FC(256 → 128) → μ, log_σ²  (latent_dim)
Decoder  : FC(latent_dim → 128) → FC(128 → 256) → FC(256 → input_dim)

Anomaly score = mean squared reconstruction error per sample.
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class Encoder(nn.Module):
    """Probabilistic encoder: maps input x → (μ, log_σ²)."""

    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...], latent_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.LeakyReLU(0.2)]
            in_dim = h
        self.net = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_log_var = nn.Linear(in_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.fc_mu(h), self.fc_log_var(h)


class Decoder(nn.Module):
    """Deterministic decoder: maps z → x̂."""

    def __init__(self, latent_dim: int, hidden_dims: Tuple[int, ...], output_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = latent_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.LeakyReLU(0.2)]
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))  # no activation – raw logits
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VAE(nn.Module):
    """
    Variational Autoencoder for transaction anomaly detection.

    Parameters
    ----------
    input_dim   : dimensionality of scaled feature vector
    latent_dim  : size of latent code z  (default 16)
    hidden_dims : encoder/decoder hidden layer sizes
    beta        : KL-divergence weight (β-VAE). β > 1 encourages disentanglement.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        hidden_dims: Tuple[int, ...] = (256, 128),
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, tuple(reversed(hidden_dims)), input_dim)

        self._init_weights()
        log.debug(
            "VAE initialised  input_dim=%d  latent_dim=%d  hidden=%s  β=%.2f",
            input_dim, latent_dim, hidden_dims, beta,
        )

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    @staticmethod
    def reparameterise(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample z = μ + ε·σ  (ε ~ N(0,1)) during training; return μ at eval."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        x_hat   : reconstructed input
        mu      : latent mean
        log_var : latent log-variance
        z       : sampled latent code
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterise(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var, z

    # ------------------------------------------------------------------
    def loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ELBO loss = MSE reconstruction + β·KL divergence.

        Returns
        -------
        total_loss : scalar
        recon_loss : scalar
        kl_loss    : scalar
        """
        recon_loss = F.mse_loss(x_hat, x, reduction="mean")
        # Analytical KL: -½ Σ(1 + log_var - μ² - exp(log_var))
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        total_loss = recon_loss + self.beta * kl_loss
        return total_loss, recon_loss, kl_loss

    # ------------------------------------------------------------------
    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample anomaly score = mean squared reconstruction error.
        Higher score → more anomalous.
        """
        self.eval()
        mu, _ = self.encoder(x)           # use μ (no noise at inference)
        x_hat = self.decoder(mu)
        scores = F.mse_loss(x_hat, x, reduction="none").mean(dim=1)
        return scores

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience accessor for the encoder."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Convenience accessor for the decoder."""
        return self.decoder(z)
