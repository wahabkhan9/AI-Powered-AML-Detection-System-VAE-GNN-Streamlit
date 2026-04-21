"""
tests/test_gan.py
=================
Unit tests for the TransactionGAN model.
"""

from __future__ import annotations

import pytest
import torch
import numpy as np
from models.gan import TransactionGAN, Generator, Discriminator


@pytest.fixture(scope="module")
def gan() -> TransactionGAN:
    return TransactionGAN(feature_dim=9, latent_dim=16, hidden_dims_g=(32, 64), hidden_dims_d=(64, 32))


@pytest.fixture(scope="module")
def real_batch() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(32, 9)


class TestGenerator:
    def test_output_shape(self, gan, real_batch):
        z = torch.randn(32, 16)
        fake = gan.generator(z)
        assert fake.shape == (32, 9)

    def test_output_in_tanh_range(self, gan):
        z = torch.randn(64, 16)
        fake = gan.generator(z)
        assert fake.min() >= -1.01 and fake.max() <= 1.01


class TestDiscriminator:
    def test_output_shape(self, gan, real_batch):
        logits = gan.discriminator(real_batch)
        assert logits.shape == (32, 1)

    def test_anomaly_score_range(self, gan, real_batch):
        scores = gan.discriminator.anomaly_score(real_batch)
        assert scores.shape == (32,)
        assert (scores >= 0).all() and (scores <= 1).all()


class TestTransactionGAN:
    def test_generate_shape(self, gan):
        samples = gan.generate(50)
        assert samples.shape == (50, 9)

    def test_discriminator_loss_positive(self, gan, real_batch):
        z = torch.randn(32, 16)
        fake = gan.generator(z)
        loss = gan.discriminator_loss(real_batch, fake)
        assert loss.item() > 0

    def test_generator_loss_positive(self, gan):
        z = torch.randn(32, 16)
        fake = gan.generator(z)
        loss = gan.generator_loss(fake)
        assert loss.item() > 0

    def test_gradient_flows_discriminator(self, gan, real_batch):
        z = torch.randn(32, 16)
        fake = gan.generator(z)
        loss = gan.discriminator_loss(real_batch, fake)
        loss.backward()
        for name, p in gan.discriminator.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"

    def test_gradient_flows_generator(self, gan):
        z = torch.randn(32, 16)
        fake = gan.generator(z)
        loss = gan.generator_loss(fake)
        loss.backward()
        for name, p in gan.generator.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"

    def test_anomaly_score_deterministic_at_eval(self, gan, real_batch):
        s1 = gan.anomaly_score(real_batch)
        s2 = gan.anomaly_score(real_batch)
        assert torch.allclose(s1, s2)
