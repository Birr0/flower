"""Tests for the shared Flow model — LightningFlowMatching and VelocityField.

VelocityField and LightningFlowMatching now live in flower.models.modules and
are shared by every dataset (dsprites, rgbmnist, ...) via subclassing. These
tests exercise that shared implementation directly instead of duplicating the
same checks per dataset subclass.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn
from flow_matching.solver import ODESolver

from flower.models.modules import LightningFlowMatching, VelocityField, WrappedModel

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def catalog():
    return {
        "variables": {
            "x": {"size": 3, "continuous": 1},
            "y": {"size": 1, "continuous": 1},
        },
        "drop_variables": [],
    }


@pytest.fixture
def base_model():
    """A minimal model with an encode method for LightningFlowMatching."""

    class MockBaseModel(nn.Module):
        def encode(self, x):
            b = x.shape[0]
            return {
                "z": x.clone(),
                "mu": torch.zeros(b, 64),
                "logvar": torch.zeros(b, 64),
            }

    return MockBaseModel()


# ---------------------------------------------------------------------------
# VelocityField (unit)
# ---------------------------------------------------------------------------


class TestVelocityField:
    """Tests for the shared VelocityField neural network."""

    @pytest.fixture
    def vf(self):
        return VelocityField(code_dim=64, hidden_dim=64, conditional_dim=12, n_layers=2)

    def test_forward_shapes(self, vf):
        x_t = torch.randn(4, 64)
        t = torch.tensor([0.5, 0.3, 0.7, 0.1])
        y = torch.randn(4, 12)
        out = vf(x_t=x_t, t=t, y=y)
        assert out.shape == (4, 64)

    def test_forward_deterministic(self, vf):
        x_t = torch.randn(4, 64)
        t = torch.tensor([0.5, 0.3, 0.7, 0.1])
        y = torch.randn(4, 12)
        vf.eval()
        with torch.no_grad():
            out1 = vf(x_t=x_t.clone(), t=t.clone(), y=y.clone())
            out2 = vf(x_t=x_t.clone(), t=t.clone(), y=y.clone())
        torch.testing.assert_close(out1, out2)

    def test_different_batch_sizes(self, vf):
        for batch in [1, 8, 32]:
            x_t = torch.randn(batch, 64)
            t = torch.rand(batch)
            y = torch.randn(batch, 12)
            out = vf(x_t=x_t, t=t, y=y)
            assert out.shape == (batch, 64)


# ---------------------------------------------------------------------------
# LightningFlowMatching — integration / smoke tests
# ---------------------------------------------------------------------------


class TestLightningFlowMatching:
    """Smoke tests for the shared LightningFlowMatching module."""

    @pytest.fixture
    def config(self, catalog, base_model):
        return {
            "base_model": base_model,
            "lr": 1e-3,
            "batch_size": 4,
            "code_dim": 64,
            "hidden_dim": 64,
            "catalog": catalog,
            "n_steps": 20,
            "n_layers": 2,
            "beta_start_step": 0,
            "beta_warmup_steps": 100,
            "max_beta": 1.0,
        }

    @pytest.fixture
    def flow(self, config):
        flow = LightningFlowMatching(**config)
        # predict_step requires solver, which is normally created with ckpt_path
        flow.wrapped_vf = WrappedModel(flow.vf)
        flow.solver = ODESolver(velocity_model=flow.wrapped_vf)
        return flow

    def _make_batch(self, flow):
        """Create a batch matching the catalog's conditional dimension."""
        X = torch.randn(4, 64)
        cond_dim = flow.cond_dim  # Derived from catalog
        y = torch.randn(4, cond_dim)
        return {"X": X, "y": y}

    def test_configure_optimizers(self, flow):
        opt = flow.configure_optimizers()
        assert opt is not None
        assert hasattr(opt, "param_groups")

    def test_training_step(self, flow):
        batch = self._make_batch(flow)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loss = flow.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_validation_step(self, flow):
        batch = self._make_batch(flow)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loss = flow.validation_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_loss_values_reachable(self, flow):
        batch = self._make_batch(flow)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loss = flow.training_step(batch, 0)
        assert loss > 0.0
        assert torch.isfinite(loss)

    def test_predict_step(self, flow):
        flow.eval()
        batch = self._make_batch(flow)
        with torch.no_grad():
            out = flow.predict_step(batch["X"], batch["y"], embed_opt=["orig"])
        assert "orig" in out

    def test_predict_step_cond(self, flow):
        flow.eval()
        batch = self._make_batch(flow)
        with torch.no_grad():
            out = flow.predict_step(batch["X"], batch["y"], embed_opt=["cond"])
        assert "cond" in out

    def test_predict_step_uncond(self, flow):
        flow.eval()
        batch = self._make_batch(flow)
        with torch.no_grad():
            out = flow.predict_step(batch["X"], batch["y"], embed_opt=["uncond"])
        assert "uncond" in out

    def test_predict_step_multiple(self, flow):
        flow.eval()
        batch = self._make_batch(flow)
        with torch.no_grad():
            out = flow.predict_step(
                batch["X"], batch["y"], embed_opt=["orig", "cond", "uncond"]
            )
        assert set(out.keys()) == {"orig", "cond", "uncond"}


class TestLightningFlowMatchingZOnlyEncoder:
    """Smoke tests for the shared base paired with a PretrainedSpender-shaped
    encoder (spectra's base model) — no "mu"/"logvar" keys, only "z". Tests
    against flower.models.modules directly rather than flower.models.spectra,
    since spectra.LightningFlowMatching is a bare `pass` subclass of the base.
    """

    @pytest.fixture
    def base_model(self):
        class MockSpenderBaseModel(nn.Module):
            """Mimics PretrainedSpender.encode: only a "z" key, no mu/logvar."""

            def encode(self, x):
                return {"z": x.clone()}

        return MockSpenderBaseModel()

    @pytest.fixture
    def config(self, catalog, base_model):
        return {
            "base_model": base_model,
            "lr": 1e-3,
            "batch_size": 4,
            "code_dim": 64,
            "hidden_dim": 64,
            "catalog": catalog,
            "n_steps": 20,
            "n_layers": 3,
            "beta_start_step": 0,
            "beta_warmup_steps": 100,
            "max_beta": 1.0,
        }

    @pytest.fixture
    def flow(self, config):
        flow = LightningFlowMatching(**config)
        flow.wrapped_vf = WrappedModel(flow.vf)
        flow.solver = ODESolver(velocity_model=flow.wrapped_vf)
        return flow

    def _make_batch(self, flow):
        X = torch.randn(4, 64)
        y = torch.randn(4, flow.cond_dim)
        return {"X": X, "y": y}

    def test_configure_optimizers(self, flow):
        opt = flow.configure_optimizers()
        assert opt is not None
        assert hasattr(opt, "param_groups")

    def test_training_step(self, flow):
        batch = self._make_batch(flow)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loss = flow.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_validation_step(self, flow):
        batch = self._make_batch(flow)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loss = flow.validation_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_test_step(self, flow):
        batch = self._make_batch(flow)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loss = flow.test_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_predict_step_multiple(self, flow):
        flow.eval()
        batch = self._make_batch(flow)
        with torch.no_grad():
            out = flow.predict_step(
                batch["X"], batch["y"], embed_opt=["orig", "cond", "uncond"]
            )
        assert set(out.keys()) == {"orig", "cond", "uncond"}
