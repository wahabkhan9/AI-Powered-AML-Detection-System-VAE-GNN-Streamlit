"""
tests/test_vae.py
=================
Unit and integration tests for the VAE anomaly detector.
"""

from __future__ import annotations

import pytest
import torch
import numpy as np
from models.vae import VAE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def vae() -> VAE:
    return VAE(input_dim=9, latent_dim=8, hidden_dims=(32, 16), beta=1.0)


@pytest.fixture(scope="module")
def sample_batch() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(64, 9)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestVAEArchitecture:
    def test_forward_output_shapes(self, vae, sample_batch):
        x_hat, mu, log_var, z = vae(sample_batch)
        assert x_hat.shape == sample_batch.shape, "Reconstruction shape mismatch"
        assert mu.shape == (64, 8), "μ shape mismatch"
        assert log_var.shape == (64, 8), "log_var shape mismatch"
        assert z.shape == (64, 8), "z shape mismatch"

    def test_loss_is_positive(self, vae, sample_batch):
        x_hat, mu, log_var, _ = vae(sample_batch)
        loss, recon, kl = vae.loss(sample_batch, x_hat, mu, log_var)
        assert loss.item() > 0, "Total loss must be positive"
        assert recon.item() >= 0, "Reconstruction loss must be non-negative"
        assert kl.item() >= 0, "KL loss must be non-negative"

    def test_anomaly_score_shape(self, vae, sample_batch):
        scores = vae.anomaly_score(sample_batch)
        assert scores.shape == (64,), "Anomaly scores must be 1-D per sample"

    def test_anomaly_score_non_negative(self, vae, sample_batch):
        scores = vae.anomaly_score(sample_batch)
        assert (scores >= 0).all(), "Anomaly scores must be non-negative"

    def test_reparameterise_uses_mu_at_eval(self, vae, sample_batch):
        """In eval mode, anomaly_score uses μ deterministically."""
        vae.eval()
        s1 = vae.anomaly_score(sample_batch)
        s2 = vae.anomaly_score(sample_batch)
        assert torch.allclose(s1, s2), "Scores should be deterministic at eval time"

    def test_parameter_count(self, vae):
        n_params = sum(p.numel() for p in vae.parameters())
        assert n_params > 0, "VAE should have trainable parameters"

    def test_gradient_flow(self, vae, sample_batch):
        vae.train()
        x_hat, mu, log_var, _ = vae(sample_batch)
        loss, _, _ = vae.loss(sample_batch, x_hat, mu, log_var)
        loss.backward()
        for name, p in vae.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"


class TestVAEAnomalyDetection:
    def test_anomalous_samples_score_higher(self):
        """Samples far from the training distribution should have higher scores."""
        torch.manual_seed(42)
        vae = VAE(input_dim=4, latent_dim=4, hidden_dims=(16, 8), beta=1.0)
        # Simulate training on near-zero data
        optimizer = torch.optim.Adam(vae.parameters(), lr=1e-2)
        for _ in range(100):
            x = torch.randn(128, 4) * 0.1          # normal = near zero
            x_hat, mu, lv, _ = vae(x)
            loss, _, _ = vae.loss(x, x_hat, mu, lv)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        normal = torch.randn(100, 4) * 0.1
        anomalous = torch.randn(100, 4) * 10.0     # far out-of-distribution

        normal_scores = vae.anomaly_score(normal).mean().item()
        anomaly_scores = vae.anomaly_score(anomalous).mean().item()
        assert anomaly_scores > normal_scores, (
            f"Anomalous samples should score higher: {anomaly_scores:.4f} vs {normal_scores:.4f}"
        )

    def test_encode_decode_roundtrip(self):
        vae = VAE(input_dim=6, latent_dim=3, hidden_dims=(16,), beta=0.0)
        x = torch.randn(32, 6)
        mu, _ = vae.encode(x)
        x_hat = vae.decode(mu)
        assert x_hat.shape == x.shape
