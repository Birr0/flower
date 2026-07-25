"""Tests for the iVAE baseline on the 2D-Gaussians toy (issue #20 / E2).

The toy (``examples/2d_gaussians/data.py``) is a 4-mode GMM at (+/-3, +/-3) with
isotropic covariance 0.5: ``X = center(y) + diff`` where ``y`` is the mode
(discrete condition) and ``diff ~ N(0, 0.5 I)`` is the condition-independent
"seed". The iVAE baseline conditions on ``u = one-hot(y)`` and must recover a
representation with the mode suppressed but ``diff`` preserved.

Two residual constructions are exercised (both requested for the baseline):
- **method B** — the conditional-prior residual ``s - lambda_mu(u)`` (the natural
  one here, since the mode lives entirely in the mean),
- **method A** — drop the sources most dependent on ``y``.

Fast tests cover the two pure residual functions; the ``slow`` test trains the
iVAE and checks it recovers ``diff`` while suppressing the mode.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression

from flower.evaluation.ica import (
    compute_mcc,
    conditional_mean_residual,
    conditional_prior_residual,
    drop_top_k_dependent,
    regression_residual,
)
from flower.models.ivae import IVAE, LightningIVAE


def _make_gaussian_data(n_per_mode=2000, seed=0):
    """Additive 4-mode Gaussian toy: X = center[mode] + diff."""
    rng = np.random.default_rng(seed)
    centers = np.array([[3, 3], [-3, 3], [-3, -3], [3, -3]], dtype=np.float32)
    modes = np.repeat(np.arange(4), n_per_mode)
    rng.shuffle(modes)
    diff = rng.normal(0.0, np.sqrt(0.5), (len(modes), 2)).astype(np.float32)
    x = centers[modes] + diff
    u = np.eye(4, dtype=np.float32)[modes]
    return x, u, diff, modes


# ---------------------------------------------------------------------------
# Method B — conditional-prior residual
# ---------------------------------------------------------------------------
class TestConditionalPriorResidual:
    def test_subtracts_prior_mean(self):
        s = np.array([[1.0, 2.0], [3.0, 4.0]])
        prior_mu = np.array([[0.5, 1.0], [1.0, 1.0]])
        res = conditional_prior_residual(s, prior_mu)
        assert np.allclose(res, s - prior_mu)

    def test_shape_preserved(self):
        s = np.random.default_rng(0).standard_normal((10, 3))
        res = conditional_prior_residual(s, np.zeros((10, 3)))
        assert res.shape == (10, 3)


# ---------------------------------------------------------------------------
# Method B (linear analog) — subtract the empirical per-condition mean
# ---------------------------------------------------------------------------
class TestConditionalMeanResidual:
    def test_subtracts_group_means(self):
        y = np.array([0, 0, 1, 1])
        # group 0 mean = 2, group 1 mean = 11 (first column)
        s = np.array([[1.0, 0.0], [3.0, 0.0], [10.0, 0.0], [12.0, 0.0]])
        res, means = conditional_mean_residual(s, y)
        assert np.allclose(res[:, 0], [-1.0, 1.0, -1.0, 1.0])
        assert np.allclose(means[0], [2.0, 0.0])
        assert np.allclose(means[1], [11.0, 0.0])

    def test_reuses_provided_means(self):
        # Means fit on train are applied verbatim to a (test) split.
        y = np.array([0, 1])
        s = np.array([[5.0], [5.0]])
        means = {0: np.array([1.0]), 1: np.array([2.0])}
        res, out_means = conditional_mean_residual(s, y, means)
        assert np.allclose(res[:, 0], [4.0, 3.0])
        assert out_means is means

    def test_shape_preserved(self):
        rng = np.random.default_rng(0)
        s = rng.standard_normal((20, 3))
        y = rng.integers(0, 4, 20)
        res, _ = conditional_mean_residual(s, y)
        assert res.shape == (20, 3)


# ---------------------------------------------------------------------------
# Method B (continuous) — residualise against a continuous condition
# ---------------------------------------------------------------------------
class TestRegressionResidual:
    def test_removes_linear_dependence(self):
        rng = np.random.default_rng(0)
        y = rng.standard_normal(600)
        # col 0 is linear in y; col 1 is independent.
        s = np.stack(
            [2.0 * y + 0.01 * rng.standard_normal(600), rng.standard_normal(600)],
            axis=1,
        )
        res, _coef = regression_residual(s, y)
        assert abs(np.corrcoef(res[:, 0], y)[0, 1]) < 0.1  # linear part removed
        assert res.shape == s.shape

    def test_reuses_fitted_coef(self):
        rng = np.random.default_rng(1)
        y_tr = rng.standard_normal(400)
        s_tr = np.stack(
            [3.0 * y_tr + 0.05 * rng.standard_normal(400), rng.standard_normal(400)],
            axis=1,
        )
        _, coef = regression_residual(s_tr, y_tr)
        # Apply train coefficients to a test split.
        y_te = rng.standard_normal(100)
        s_te = np.stack(
            [3.0 * y_te + 0.05 * rng.standard_normal(100), rng.standard_normal(100)],
            axis=1,
        )
        res_te, coef2 = regression_residual(s_te, y_te, coef)
        assert coef2 is coef
        # Training fit still removes the y-dependence on the held-out split.
        assert abs(np.corrcoef(res_te[:, 0], y_te)[0, 1]) < 0.2


# ---------------------------------------------------------------------------
# Method A — drop sources most dependent on the condition
# ---------------------------------------------------------------------------
class TestDropTopKDependent:
    @pytest.fixture
    def sources_and_labels(self):
        rng = np.random.default_rng(0)
        y = np.repeat(np.arange(4), 250)
        # col 0 strongly depends on y (per-group means), col 1 is independent.
        col0 = y.astype(float) * 5.0 + rng.standard_normal(len(y))
        col1 = rng.standard_normal(len(y))
        sources = np.stack([col0, col1], axis=1)
        return sources, y

    def test_drops_the_dependent_source(self, sources_and_labels):
        sources, y = sources_and_labels
        res, dropped = drop_top_k_dependent(sources, y, k=1)
        assert list(dropped) == [0]
        assert res.shape == (len(y), 1)
        assert np.allclose(res[:, 0], sources[:, 1])

    def test_k_zero_keeps_all(self, sources_and_labels):
        sources, y = sources_and_labels
        res, dropped = drop_top_k_dependent(sources, y, k=0)
        assert res.shape == sources.shape
        assert len(dropped) == 0

    def test_continuous_dependence(self):
        rng = np.random.default_rng(3)
        y = rng.standard_normal(1000)  # continuous condition
        col0 = 2.0 * y + rng.standard_normal(1000)  # correlated with y
        col1 = rng.standard_normal(1000)  # independent
        sources = np.stack([col0, col1], axis=1)
        res, dropped = drop_top_k_dependent(sources, y, k=1, dependence="continuous")
        assert list(dropped) == [0]
        assert np.allclose(res[:, 0], sources[:, 1])


# ---------------------------------------------------------------------------
# Headline: recover the seed and suppress the mode (2D-Gaussians)
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestGaussiansIdentifiability:
    def _train(self, x, u, *, epochs, seed):
        torch.manual_seed(seed)
        x_t = torch.as_tensor(x)
        u_t = torch.as_tensor(u)
        ivae = IVAE(
            data_dim=2,
            aux_dim=u.shape[1],
            latent_dim=2,
            hidden_dim=64,
            n_layers=3,
            activation="xtanh",
            learn_prior_mean=True,  # the mode lives in the mean
        )
        lit = LightningIVAE(ivae, lr=1e-2, batch_size=64, beta=1.0)
        opt = torch.optim.Adam(ivae.parameters(), lr=lit.lr)
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
            out = ivae(x_t, u_t)
        return out["mu"].numpy(), out["prior_mu"].numpy()

    @staticmethod
    def _mode_accuracy(features, modes):
        n = len(modes)
        split = n // 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(max_iter=1000)
            clf.fit(features[:split], modes[:split])
            return clf.score(features[split:], modes[split:])

    def test_recovers_seed_and_suppresses_mode(self):
        x, u, diff, modes = _make_gaussian_data(n_per_mode=1500, seed=0)

        s_hat, prior_mu = self._train(x, u, epochs=30, seed=0)
        residual_b = conditional_prior_residual(s_hat, prior_mu)

        # (1) The conditional-prior residual recovers the seed `diff` (calibrated
        # ~0.89), far above the raw sources which still carry the mode mean.
        assert compute_mcc(diff, residual_b) > 0.85
        assert compute_mcc(diff, residual_b) > compute_mcc(diff, s_hat) + 0.3

        # (2) The mode is trivial from raw X but suppressed to ~chance (0.25) in
        # the residual.
        acc_raw = self._mode_accuracy(x, modes)
        acc_res = self._mode_accuracy(residual_b, modes)
        assert acc_raw > 0.95, f"mode should be trivial from X: {acc_raw:.3f}"
        assert acc_res < 0.45, f"mode should be suppressed in residual: {acc_res:.3f}"
