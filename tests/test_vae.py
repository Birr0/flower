"""Tests for the VAE model — BetaVAE and LightningVAE."""
from __future__ import annotations

import warnings
import pytest
import torch

from flower.models.dsprites import BetaVAE, LightningBetaVAE
from flower.models.rgbmnist import LightningVAE, VAE


class TestBetaVAE:
    """Tests for the BetaVAE architecture."""

    @pytest.fixture()
    def model(self):
        return BetaVAE(latent_dim=10)

    def test_encode_returns_shapes(self, model):
        x = torch.randn(4, 1, 64, 64)
        z, mu, logvar = model.encode(x)
        assert z.shape == (4, 10)
        assert mu.shape == (4, 10)
        assert logvar.shape == (4, 10)

    def test_forward_returns_dict(self, model):
        x = torch.randn(4, 1, 64, 64)
        out = model(x)
        # BetaVAE.forward returns (reconstruction, z, mu, logvar)
        recon, z, mu, logvar = out
        assert isinstance(recon, torch.Tensor)
        assert z.shape == (4, 10)
        assert mu.shape == (4, 10)

    def test_reconstruction_shape(self, model):
        x = torch.randn(4, 1, 64, 64)
        recon, _, _, _ = model(x)
        assert recon.shape == (4, 1, 64, 64)


class TestVAE:
    """Tests for the VAE model (older version)."""

    @pytest.fixture()
    def model(self):
        return VAE(hidden_dim=64)

    def test_forward_returns_dict(self, model):
        x = torch.randn(4, 3, 28, 28)
        out = model(x)
        assert "z" in out
        assert "recon" in out
        assert "mu" in out
        assert "log_var" in out

    def test_z_shape(self, model):
        x = torch.randn(4, 3, 28, 28)
        out = model(x)
        assert out["z"].shape == (4, 64)

    def test_recon_shape(self, model):
        x = torch.randn(4, 3, 28, 28)
        out = model(x)
        assert out["recon"].shape == (4, 3, 28, 28)


class TestLightningVAE:
    """Tests for LightningVAE — one training step only."""

    @pytest.fixture()
    def vae_model(self):
        return BetaVAE(latent_dim=10)

    @pytest.fixture()
    def lightning_vae(self, vae_model):
        return LightningVAE(vae_model, lr=1e-3, batch_size=4, beta=1.0)

    def test_configure_optimizers(self, lightning_vae):
        opt = lightning_vae.configure_optimizers()
        assert opt is not None
        # Should be an optimizer instance
        assert hasattr(opt, "param_groups")

    @pytest.fixture()
    def vae_batch(self):
        """BetaVAE expects 64x64 grayscale images."""
        return {
            "X": torch.randn(2, 1, 64, 64),
            "y": torch.randn(2, 3),
            "catalog": {
                "r": torch.tensor([0.3, 0.3]),
                "g": torch.tensor([0.5, 0.5]),
                "b": torch.tensor([0.7, 0.7]),
                "digit": torch.tensor([0, 1]).unsqueeze(-1),
            },
        }

    def test_training_step(self, lightning_vae, vae_batch):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loss = lightning_vae.training_step(vae_batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_validation_step(self, lightning_vae, vae_batch):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loss = lightning_vae.validation_step(vae_batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_loss_values_reachable(self, lightning_vae, vae_batch):
        """Loss should be a finite positive number."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loss = lightning_vae.training_step(vae_batch, 0)
        assert loss > 0.0
        assert torch.isfinite(loss)

    def test_test_step(self, lightning_vae, vae_batch):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loss = lightning_vae.test_step(vae_batch)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)


class TestLightningBetaVAE:
    """Tests for LightningBetaVAE."""

    @pytest.fixture()
    def vae_model(self):
        return BetaVAE(latent_dim=10)

    @pytest.fixture()
    def lightning_vae(self, vae_model):
        return LightningBetaVAE(vae_model, lr=1e-3, beta=1.0)

    @pytest.fixture()
    def vae_batch(self):
        return {
            "X": torch.randn(2, 1, 64, 64),
            "y": torch.randn(2, 3),
            "catalog": {
                "r": torch.tensor([0.3, 0.3]),
                "g": torch.tensor([0.5, 0.5]),
                "b": torch.tensor([0.7, 0.7]),
                "digit": torch.tensor([0, 1]).unsqueeze(-1),
            },
        }

    def test_training_step(self, lightning_vae, vae_batch):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loss = lightning_vae.training_step(vae_batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)