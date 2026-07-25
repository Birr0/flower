"""Tests for the iVAE nonlinear-ICA baseline (issue #20 / E2).

The model and synthetic benchmark follow the reference implementation
https://github.com/ilkhem/iVAE (Khemakhem et al., AISTATS 2020). Two tiers,
mirroring the rest of the suite:

1. Fast contract tests — the ``{z, mu, logvar}`` encoder contract, tensor shapes,
   the conditional-prior parameters, the closed-form KL (checked against
   ``torch.distributions``), a single Lightning step, and ``predict_step`` keys.
2. The headline ``slow`` identifiability test — generate synthetic *nonlinear*-ICA
   data (xtanh mixing, segment-modulated sources) with auxiliary ``u``, train the
   iVAE, and confirm it recovers the true sources (high MCC) where a *linear*
   FastICA floor cannot. This is the core nonlinear-ICA identifiability result.

The identifiability test uses a plain torch loop (not a Lightning ``Trainer``) to
stay fast and avoid the trainer warning-storm under ``filterwarnings = error`` —
but it reuses the shipped loss (``LightningIVAE._losses``).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch
import torch.distributions as dist
from sklearn.decomposition import FastICA

from flower.evaluation.ica import compute_mcc, sample_synthetic_nonlinear_ica
from flower.models.ivae import IVAE, LightningIVAE


# ---------------------------------------------------------------------------
# IVAE module — contract and shapes
# ---------------------------------------------------------------------------
class TestIVAE:
    @pytest.fixture
    def model(self):
        return IVAE(data_dim=8, aux_dim=3, hidden_dim=16, n_layers=2)

    def test_encode_contract(self, model):
        x = torch.randn(5, 8)
        u = torch.randn(5, 3)
        out = model.encode(x, u)
        assert set(out) == {"z", "mu", "logvar"}
        for k in out:
            assert out[k].shape == (5, 8)

    def test_latent_dim_defaults_to_data_dim(self, model):
        assert model.latent_dim == 8

    def test_custom_latent_dim(self):
        model = IVAE(data_dim=8, aux_dim=3, latent_dim=4, hidden_dim=16, n_layers=2)
        out = model.encode(torch.randn(5, 8), torch.randn(5, 3))
        assert out["z"].shape == (5, 4)

    def test_prior_params_shapes(self, model):
        u = torch.randn(5, 3)
        lam_mu, lam_logvar = model.prior_params(u)
        assert lam_mu.shape == (5, 8)
        assert lam_logvar.shape == (5, 8)

    def test_fixed_prior_mean_is_zero(self):
        model = IVAE(
            data_dim=8, aux_dim=3, hidden_dim=16, n_layers=2, learn_prior_mean=False
        )
        lam_mu, _ = model.prior_params(torch.randn(5, 3))
        assert torch.count_nonzero(lam_mu) == 0

    def test_fair_encoder_ignores_u(self):
        # condition_encoder=False: encoder sees only x (input dim = data_dim),
        # and its output is invariant to u.
        model = IVAE(
            data_dim=8, aux_dim=3, hidden_dim=16, n_layers=2, condition_encoder=False
        )
        assert model.g.fc[0].in_features == 8
        x = torch.randn(5, 8)
        torch.manual_seed(0)
        mu1 = model.encode(x, torch.zeros(5, 3))["mu"]
        mu2 = model.encode(x, torch.ones(5, 3))["mu"]
        assert torch.equal(mu1, mu2)

    def test_forward_keys_and_shapes(self, model):
        x = torch.randn(5, 8)
        u = torch.randn(5, 3)
        out = model(x, u)
        assert set(out) == {"recon", "z", "mu", "logvar", "prior_mu", "prior_logvar"}
        assert out["recon"].shape == x.shape
        assert out["prior_mu"].shape == (5, 8)

    def test_logvar_is_clamped(self, model):
        out = model.encode(1e6 * torch.randn(5, 8), torch.randn(5, 3))
        assert torch.isfinite(out["logvar"]).all()
        assert (out["logvar"] <= 20.0 + 1e-4).all()
        assert (out["logvar"] >= -30.0 - 1e-4).all()


# ---------------------------------------------------------------------------
# Closed-form KL to the conditional prior
# ---------------------------------------------------------------------------
class TestKLToPrior:
    def test_matches_torch_distributions(self):
        torch.manual_seed(0)
        mu = torch.randn(7, 5)
        logvar = torch.randn(7, 5)
        prior_mu = torch.randn(7, 5)
        prior_logvar = torch.randn(7, 5)

        q = dist.Normal(mu, (0.5 * logvar).exp())
        p = dist.Normal(prior_mu, (0.5 * prior_logvar).exp())
        expected = dist.kl_divergence(q, p).sum(dim=-1).mean()

        got = LightningIVAE._kl_to_prior(mu, logvar, prior_mu, prior_logvar)
        assert torch.allclose(got, expected, atol=1e-5)

    def test_zero_when_q_equals_prior(self):
        mu = torch.randn(4, 6)
        logvar = torch.randn(4, 6)
        kl = LightningIVAE._kl_to_prior(mu, logvar, mu.clone(), logvar.clone())
        assert torch.allclose(kl, torch.zeros(()), atol=1e-6)

    def test_non_negative(self):
        torch.manual_seed(1)
        kl = LightningIVAE._kl_to_prior(
            torch.randn(4, 6), torch.randn(4, 6), torch.randn(4, 6), torch.randn(4, 6)
        )
        assert kl.item() >= 0.0


# ---------------------------------------------------------------------------
# LightningIVAE — one step and predict
# ---------------------------------------------------------------------------
class TestLightningIVAE:
    @pytest.fixture
    def lit(self):
        ivae = IVAE(data_dim=8, aux_dim=3, hidden_dim=16, n_layers=2)
        return LightningIVAE(ivae, lr=1e-2, batch_size=4, beta=1.0)

    @pytest.fixture
    def batch(self):
        return {"X": torch.randn(4, 8), "u": torch.randn(4, 3), "catalog": {}}

    def test_training_step_returns_scalar_loss(self, lit, batch):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # self.log() without a Trainer
            loss = lit.training_step(batch, 0)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        loss.backward()  # loss must be differentiable w.r.t. params

    def test_configure_optimizers(self, lit):
        cfg = lit.configure_optimizers()
        assert isinstance(cfg["optimizer"], torch.optim.Optimizer)
        assert "lr_scheduler" in cfg

    def test_predict_step_keys(self, lit, batch):
        out = lit.predict_step(batch)
        assert {"X", "z", "prior_mu"} <= set(out)
        assert out["z"].shape == (4, 8)


# ---------------------------------------------------------------------------
# MCC — the identifiability metric
# ---------------------------------------------------------------------------
class TestMCC:
    def test_identical_sources_give_one(self):
        rng = np.random.default_rng(0)
        s = rng.standard_normal((500, 4))
        assert compute_mcc(s, s.copy()) == pytest.approx(1.0, abs=1e-6)

    def test_invariant_to_permutation_and_sign_scale(self):
        rng = np.random.default_rng(1)
        s = rng.standard_normal((1000, 4))
        # Permute columns, flip signs, rescale — MCC must be unchanged (~1).
        perm = [2, 0, 3, 1]
        recovered = s[:, perm] * np.array([-1.0, 3.0, 0.5, -2.0])
        assert compute_mcc(s, recovered) == pytest.approx(1.0, abs=1e-6)

    def test_independent_sources_give_low_mcc(self):
        rng = np.random.default_rng(2)
        s = rng.standard_normal((2000, 4))
        r = rng.standard_normal((2000, 4))
        assert compute_mcc(s, r) < 0.3


# ---------------------------------------------------------------------------
# Headline: recover the sources under nonlinear mixing (nonlinear-ICA result)
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestIdentifiability:
    """iVAE recovers auxiliary-modulated sources under nonlinear (xtanh) mixing;
    a linear FastICA floor does not. The core nonlinear-ICA identifiability claim.
    """

    def _train_ivae(self, x, u, latent_dim, *, epochs, seed):
        torch.manual_seed(seed)
        x_t = torch.as_tensor(x)
        u_t = torch.as_tensor(u)
        ivae = IVAE(
            data_dim=x.shape[1],
            aux_dim=u.shape[1],
            latent_dim=latent_dim,
            hidden_dim=100,
            n_layers=3,
            activation="xtanh",
            learn_prior_mean=False,  # zero-mean, variance-modulated benchmark
        )
        lit = LightningIVAE(ivae, lr=1e-2, batch_size=64, beta=1.0)
        opt = torch.optim.Adam(ivae.parameters(), lr=lit.lr)
        # StepLR is the manual-loop equivalent of the shipped ReduceLROnPlateau;
        # the LR decay is what lets the encoder converge to an identifying solution.
        sched = torch.optim.lr_scheduler.StepLR(
            opt, step_size=max(1, epochs // 3), gamma=0.3
        )
        n, bs = x.shape[0], 64
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i : i + bs]
                loss, _, _ = lit._losses(ivae(x_t[idx], u_t[idx]), x_t[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
        with torch.no_grad():
            return ivae.encode(x_t, u_t)["mu"].numpy()

    def test_ivae_beats_linear_ica_floor(self):
        # Canonical regime: 2 sources mixed nonlinearly (xtanh) into 6-dim data.
        # The extra mixing dimensions defeat a linear unmixing while the low
        # source count keeps iVAE recovery tractable in a test budget.
        d_sources, d_data = 2, 6
        x, u, s_true = sample_synthetic_nonlinear_ica(
            n_per_seg=500,
            n_seg=40,
            d_sources=d_sources,
            d_data=d_data,
            n_layers=3,
            prior="gauss",
            activation="xtanh",
            seed=0,
        )

        s_ivae = self._train_ivae(x, u, d_sources, epochs=30, seed=0)
        mcc_ivae = compute_mcc(s_true, s_ivae)

        ica = FastICA(
            n_components=d_sources,
            random_state=0,
            max_iter=1000,
            whiten="unit-variance",
        )
        s_lin = ica.fit_transform(x)
        mcc_lin = compute_mcc(s_true, s_lin)

        # The nonlinear-ICA result: iVAE recovers the sources (calibrated ~0.94),
        # the linear floor cannot (~0.65).
        assert mcc_ivae > 0.85, f"iVAE MCC too low: {mcc_ivae:.3f}"
        assert mcc_ivae > mcc_lin + 0.15, (
            f"iVAE ({mcc_ivae:.3f}) should clearly beat linear floor ({mcc_lin:.3f})"
        )
