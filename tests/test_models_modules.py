"""Tests for flower.models.modules — model utility classes and functions."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from flower.models.modules import (
    AdaLN,
    BaseModel,
    ConditionEmbedder,
    ConditionalPrior,
    TimestepEmbedder,
    WrappedModel,
    get_conditional_len,
    get_no_of_continuous_variables,
)


# ---------------------------------------------------------------------------
# Catalog helpers
# ---------------------------------------------------------------------------


class TestGetConditionalLen:
    def test_rgbmnist(self, rgbmnist_catalog):
        # total = 1+1+1+10 = 13, dropped b (1) -> 12
        result = get_conditional_len(rgbmnist_catalog)
        assert result == 12

    def test_dsprites(self, dsprites_catalog):
        # total = 1+1+1+1+1+1 = 6, nothing dropped -> 6
        result = get_conditional_len(dsprites_catalog)
        assert result == 6

    def test_empty_drop(self):
        catalog = {
            "variables": {"x": {"size": 5, "continuous": 1}},
            "drop_variables": [],
        }
        assert get_conditional_len(catalog) == 5

    def test_all_dropped(self):
        catalog = {
            "variables": {"x": {"size": 5, "continuous": 1}},
            "drop_variables": ["x"],
        }
        assert get_conditional_len(catalog) == 0


class TestGetNoOfContinuousVariables:
    def test_dsprites(self, dsprites_catalog):
        # continuous: 0 + 1 + 1 + 1 + 1 + 1 = 5, nothing dropped -> 5
        assert get_no_of_continuous_variables(dsprites_catalog) == 5

    def test_rgbmnist(self, rgbmnist_catalog):
        # continuous: 1+1+1+0 = 3, dropped b (1) -> 2
        assert get_no_of_continuous_variables(rgbmnist_catalog) == 2

    def test_all_discrete(self):
        catalog = {
            "variables": {"shape": {"size": 1, "continuous": 0}},
            "drop_variables": [],
        }
        assert get_no_of_continuous_variables(catalog) == 0


# ---------------------------------------------------------------------------
# TimestepEmbedder
# ---------------------------------------------------------------------------


class TestTimestepEmbedder:
    def test_output_size(self):
        embedder = TimestepEmbedder(hidden_size=64, frequency_embedding_size=16)
        t = torch.tensor([0.0])
        out = embedder(t)
        assert out.shape == (1, 64)

    def test_batched(self):
        embedder = TimestepEmbedder(hidden_size=32, frequency_embedding_size=16)
        t = torch.tensor([0.0, 0.5, 1.0])
        out = embedder(t)
        assert out.shape == (3, 32)

    def test_deterministic(self):
        embedder = TimestepEmbedder(hidden_size=32, frequency_embedding_size=16)
        t = torch.tensor([0.5])
        out1 = embedder(t)
        out2 = embedder(t)
        torch.testing.assert_close(out1, out2)


# ---------------------------------------------------------------------------
# ConditionEmbedder
# ---------------------------------------------------------------------------


class TestConditionEmbedder:
    def test_output_size(self):
        embedder = ConditionEmbedder(cond_dim=12, hidden_size=64)
        y = torch.randn(4, 12)
        out = embedder(y)
        assert out.shape == (4, 64)

    def test_different_batch(self):
        embedder = ConditionEmbedder(cond_dim=6, hidden_size=32)
        y = torch.randn(1, 6)
        out = embedder(y)
        assert out.shape == (1, 32)


# ---------------------------------------------------------------------------
# ConditionalPrior
# ---------------------------------------------------------------------------


class TestConditionalPrior:
    def test_output_shapes(self):
        prior = ConditionalPrior(cond_dim=12, hidden_dim=64, code_dim=64)
        y = torch.randn(4, 12)
        mu, logvar = prior(y)
        assert mu.shape == (4, 64)
        assert logvar.shape == (4, 64)

    def test_initialization_zeros(self):
        prior = ConditionalPrior(cond_dim=12, hidden_dim=64, code_dim=64)
        y = torch.randn(4, 12)
        mu, logvar = prior(y)
        # Final layer is zero-initialized, so output should be near zero
        assert torch.allclose(mu, torch.zeros_like(mu), atol=1e-4)
        assert torch.allclose(logvar, torch.zeros_like(logvar), atol=1e-4)


# ---------------------------------------------------------------------------
# AdaLN
# ---------------------------------------------------------------------------


class TestAdaLN:
    def test_output_shapes(self):
        adaln = AdaLN(hidden_dim=64, cond_dim=12)
        x = torch.randn(4, 64)
        cond = torch.randn(4, 12)
        out, gate = adaln(x, cond)
        assert out.shape == (4, 64)
        assert gate.shape == (4, 64)

    def test_different_sizes(self):
        adaln = AdaLN(hidden_dim=32, cond_dim=6)
        x = torch.randn(2, 32)
        cond = torch.randn(2, 6)
        out, gate = adaln(x, cond)
        assert out.shape == (2, 32)
        assert gate.shape == (2, 32)


# ---------------------------------------------------------------------------
# WrappedModel
# ---------------------------------------------------------------------------


class TestWrappedModel:
    """Test classifier-free guidance wrapper."""

    @pytest.fixture()
    def mock_velocity_model(self):
        """A minimal nn.Module that accepts x_t, t, y kwargs like the real velocity models.

        The forward method combines x_t and y so CFG scaling actually changes outputs.
        """
        class MockVelocityModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.net_cond = nn.Sequential(
                    nn.Linear(64 + 12, 64),
                    nn.SiLU(),
                    nn.Linear(64, 64),
                )
                self.net_uncond = nn.Sequential(
                    nn.Linear(64, 64),
                    nn.SiLU(),
                    nn.Linear(64, 64),
                )
                self.null_y = nn.Embedding(1, 12)

            def forward(self, x_t, t, y=None):
                if y is not None:
                    return self.net_cond(torch.cat([x_t, y], dim=1))
                return self.net_uncond(x_t)

        return MockVelocityModel()

    def test_cfg_scale_1_returns_cond_only(self, mock_velocity_model):
        wrapped = WrappedModel(mock_velocity_model)
        wrapped.eval()
        x = torch.randn(4, 64)
        t = torch.tensor([0.5] * 4)
        y = torch.randn(4, 12)
        with torch.no_grad():
            out = wrapped(x, t, y=y)
        assert out.shape == (4, 64)

    def test_cfg_scale_gt1_guidance(self, mock_velocity_model):
        """With cfg_scale > 1, output should differ from plain conditional."""
        wrapped = WrappedModel(mock_velocity_model)
        wrapped.eval()
        x = torch.randn(4, 64)
        t = torch.tensor([0.5] * 4)
        y = torch.randn(4, 12)
        with torch.no_grad():
            out_1 = wrapped(x.clone(), t.clone(), y=y.clone(), cfg_scale=1.0)
            out_5 = wrapped(x.clone(), t.clone(), y=y.clone(), cfg_scale=5.0)
        # They should be different (not all close) when guidance is active
        assert not torch.allclose(out_1, out_5, atol=1e-4)

    def test_cfg_scale_0(self, mock_velocity_model):
        """cfg_scale=0 means only unconditional prediction."""
        wrapped = WrappedModel(mock_velocity_model)
        wrapped.eval()
        x = torch.randn(4, 64)
        t = torch.tensor([0.5] * 4)
        y = torch.randn(4, 12)
        with torch.no_grad():
            out = wrapped(x.clone(), t.clone(), y=y.clone(), cfg_scale=0.0)
        assert out.shape == (4, 64)