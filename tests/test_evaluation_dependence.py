"""Tests for the probe-free dependence metrics (issue #20 / E2).

These metrics sit alongside the probe scores (``flower.evaluation.metrics``) in
the removal/preservation tables: per-dimension, closed-form, hyperparameter-free
measures of how much a residual embedding still tracks a condition.

The tests pin down four things the results depend on:

1. **Scale.** Every metric is in ``[0, 1]``, and the categorical metric (``eta``)
   is put on the ``|Pearson r|`` scale so a table can mix variable types.
2. **What each metric can and cannot see** — including ``eta``'s blind spot for
   label-dependent *variance*, which is the caveat the write-up has to carry.
3. **Confound removal**, the property the spectra table leans on: redshift is
   entangled with the physical targets, so a raw correlation re-measures it.
4. **Backwards compatibility** with the private helpers in
   ``flower.evaluation.ica``, so ``drop_top_k_dependent`` keeps ranking sources
   exactly as it did and previously reported numbers stay reproducible.
"""

from __future__ import annotations

import numpy as np
import pytest

from flower.evaluation.dependence import (
    abs_pearson,
    abs_spearman,
    correlation_ratio,
    dependence_report,
    distance_correlation,
    multiple_correlation,
    partial_correlation,
)
from flower.evaluation.ica import _abs_correlation, _correlation_ratio


@pytest.fixture
def rng():
    return np.random.default_rng(0)


class TestAbsPearson:
    def test_perfect_linear_relation_scores_one(self):
        y = np.linspace(0, 1, 50)
        x = np.column_stack([2 * y + 1, -3 * y])
        assert np.allclose(abs_pearson(x, y), [1.0, 1.0])

    def test_is_sign_invariant(self, rng):
        y = rng.normal(size=200)
        x = rng.normal(size=(200, 3))
        assert np.allclose(abs_pearson(x, y), abs_pearson(-x, y))

    def test_matches_numpy_corrcoef(self, rng):
        y = rng.normal(size=300)
        x = rng.normal(size=(300, 4))
        expected = [abs(np.corrcoef(x[:, j], y)[0, 1]) for j in range(4)]
        assert np.allclose(abs_pearson(x, y), expected)

    def test_constant_column_scores_zero(self, rng):
        y = rng.normal(size=100)
        x = np.column_stack([np.ones(100), y])
        scores = abs_pearson(x, y)
        assert scores[0] == 0.0
        assert np.isfinite(scores).all()

    def test_one_dimensional_x_is_a_single_feature(self, rng):
        y = rng.normal(size=64)
        assert abs_pearson(y, y).shape == (1,)


class TestAbsSpearman:
    def test_monotone_nonlinear_relation_beats_pearson(self):
        t = np.linspace(0.01, 1.0, 400)
        y = t**5
        assert abs_spearman(t, y)[0] == pytest.approx(1.0)
        assert abs_pearson(t, y)[0] < 0.95

    def test_handles_ties_without_nan(self, rng):
        y = rng.integers(0, 3, size=200).astype(float)
        x = rng.integers(0, 2, size=(200, 2)).astype(float)
        scores = abs_spearman(x, y)
        assert np.isfinite(scores).all()
        assert ((scores >= 0) & (scores <= 1)).all()


class TestCorrelationRatio:
    def test_perfect_group_separation_scores_one(self):
        labels = np.repeat([0, 1, 2], 10)
        x = labels.astype(float).reshape(-1, 1)
        assert correlation_ratio(x, labels) == pytest.approx(1.0)

    def test_unsquared_is_the_square_root(self, rng):
        labels = rng.integers(0, 4, size=300)
        x = rng.normal(size=(300, 5)) + labels[:, None]
        eta_sq = correlation_ratio(x, labels, squared=True)
        assert np.allclose(correlation_ratio(x, labels, squared=False), np.sqrt(eta_sq))

    def test_binary_eta_equals_point_biserial_pearson(self, rng):
        labels = rng.integers(0, 2, size=400)
        x = rng.normal(size=(400, 3)) + 0.7 * labels[:, None]
        eta = correlation_ratio(x, labels, squared=False)
        assert np.allclose(eta, abs_pearson(x, labels.astype(float)))

    def test_blind_to_variance_only_dependence(self, rng):
        """Documents the caveat: eta sees group *means*, not group spreads."""
        labels = np.repeat([0, 1], 2000)
        scale = np.where(labels == 0, 1.0, 5.0)
        x = (rng.normal(size=4000) * scale).reshape(-1, 1)
        assert correlation_ratio(x, labels)[0] < 0.02

    def test_constant_column_scores_zero(self, rng):
        labels = rng.integers(0, 3, size=90)
        assert correlation_ratio(np.ones((90, 1)), labels)[0] == 0.0


class TestMultipleCorrelation:
    def test_reduces_to_pearson_for_a_single_column(self, rng):
        y = rng.normal(size=250)
        x = rng.normal(size=(250, 3))
        assert np.allclose(multiple_correlation(x, y), abs_pearson(x, y))

    def test_recovers_a_circular_variable_from_its_sin_cos_pair(self, rng):
        theta = rng.uniform(0, 2 * np.pi, size=500)
        components = np.column_stack([np.sin(theta), np.cos(theta)])
        x = np.sin(theta).reshape(-1, 1)
        assert multiple_correlation(x, components)[0] == pytest.approx(1.0)
        # The naive alternative — correlating against the raw angle — understates
        # a relation that is in fact exact, and its value depends on where the
        # angle happens to wrap.
        assert abs_pearson(x, theta)[0] < 0.9

    def test_scores_stay_in_unit_interval(self, rng):
        x = rng.normal(size=(120, 6))
        y = rng.normal(size=(120, 3))
        scores = multiple_correlation(x, y)
        assert ((scores >= 0) & (scores <= 1)).all()


class TestPartialCorrelation:
    def test_removes_a_shared_confound(self, rng):
        """The spectra case: x and y are correlated only through z."""
        z = rng.normal(size=2000)
        x = (z + 0.1 * rng.normal(size=2000)).reshape(-1, 1)
        y = z + 0.1 * rng.normal(size=2000)
        assert abs_pearson(x, y)[0] > 0.95
        assert partial_correlation(x, y, z)[0] < 0.1

    def test_keeps_structure_not_explained_by_the_control(self, rng):
        z = rng.normal(size=2000)
        shared = rng.normal(size=2000)
        x = (z + shared).reshape(-1, 1)
        y = z + 2 * shared
        assert partial_correlation(x, y, z)[0] > 0.9

    def test_rank_variant_is_required_when_the_confound_acts_nonlinearly(self, rng):
        """The spectra case in miniature: the control enters through a curved map.

        Removing ``z`` *linearly* barely dents a ``z**3`` confound, so the linear
        partial still reads ~1.0 and would be reported as preserved structure.
        The rank variant removes most of it. This is why the spectra table has to
        default to ``partial_spearman``.
        """
        z = rng.uniform(0.1, 1.0, size=1500)
        x = (z**3 + 0.01 * rng.normal(size=1500)).reshape(-1, 1)
        y = z**3 + 0.01 * rng.normal(size=1500)

        raw = abs_pearson(x, y)[0]
        linear = partial_correlation(x, y, z)[0]
        ranked = partial_correlation(x, y, z, rank=True)[0]

        assert raw > 0.95
        assert linear > 0.9
        assert ranked < 0.25
        assert ranked < linear / 3

    def test_accepts_multiple_controls(self, rng):
        control = rng.normal(size=(800, 2))
        x = (control.sum(axis=1) + 0.05 * rng.normal(size=800)).reshape(-1, 1)
        y = control.sum(axis=1) + 0.05 * rng.normal(size=800)
        assert partial_correlation(x, y, control)[0] < 0.2


class TestDistanceCorrelation:
    """Distance correlation — the any-dependence metric, subsampled.

    dCor is zero *only* under independence, so it sees the two failure modes that
    defeat the linear/monotone family (documented in the spectra and 2D-Gaussians
    notes): a condition spread thinly across coordinates, and a nonlinear
    symmetric function of coordinates. It costs O(n^2) memory, so it is always
    computed on a capped subsample and averaged over replicates. These tests pin
    down both the detection property and the cost control.
    """

    def test_perfect_dependence_scores_one(self, rng):
        x = rng.normal(size=(300, 2))
        scores = distance_correlation(x, x[:, 0], n_subsample=200, random_state=0)
        assert scores[0] == pytest.approx(1.0)

    def test_detects_a_relation_pearson_cannot_see(self, rng):
        """y = x**2 on symmetric x: Pearson and Spearman ~ 0, dCor clearly not."""
        x = rng.normal(size=(2000, 1))
        y = x[:, 0] ** 2 + 0.01 * rng.normal(size=2000)
        assert abs_pearson(x, y)[0] < 0.1
        assert abs_spearman(x, y)[0] < 0.1
        assert distance_correlation(x, y, n_subsample=500, random_state=0)[0] > 0.3

    def test_detects_the_radial_distance_blind_spot(self, rng):
        """The 2D-Gaussians case: a norm is invisible to per-coordinate |rho|."""
        x = rng.normal(size=(2000, 2))
        dist = np.linalg.norm(x, axis=1)
        assert abs_spearman(x, dist).max() < 0.15
        scores = distance_correlation(x, dist, n_subsample=500, random_state=0)
        assert scores.max() > 0.3

    def test_independent_data_scores_near_zero(self, rng):
        x = rng.normal(size=(2000, 3))
        y = rng.normal(size=2000)
        assert distance_correlation(x, y, n_subsample=500, random_state=0).max() < 0.2

    def test_subsampling_caps_the_cost(self, rng):
        """A full O(n^2) pass at this n would not fit in memory; the cap must bind."""
        x = rng.normal(size=(200_000, 2))
        y = rng.normal(size=200_000)
        scores = distance_correlation(x, y, n_subsample=200, n_replicates=2)
        assert scores.shape == (2,)
        assert np.isfinite(scores).all()

    def test_is_deterministic_given_a_random_state(self, rng):
        x = rng.normal(size=(1000, 2))
        y = x[:, 0] ** 2 + rng.normal(size=1000)
        a = distance_correlation(x, y, n_subsample=300, random_state=7)
        b = distance_correlation(x, y, n_subsample=300, random_state=7)
        assert np.array_equal(a, b)

    def test_accepts_a_multi_column_y(self, rng):
        x = rng.normal(size=(600, 2))
        y = np.column_stack([x[:, 0], rng.normal(size=600)])
        scores = distance_correlation(x, y, n_subsample=300, random_state=0)
        assert scores.shape == (2,)
        assert scores[0] > scores[1]

    def test_constant_column_scores_zero(self, rng):
        x = np.column_stack([np.ones(400), rng.normal(size=400)])
        y = rng.normal(size=400)
        assert distance_correlation(x, y, n_subsample=200, random_state=0)[0] == 0.0

    def test_scores_stay_in_unit_interval(self, rng):
        x = rng.normal(size=(500, 3))
        y = rng.normal(size=500)
        scores = distance_correlation(x, y, n_subsample=250, random_state=0)
        assert ((scores >= 0) & (scores <= 1)).all()

    def test_subsample_larger_than_n_uses_every_row(self, rng):
        x = rng.normal(size=(120, 2))
        y = x[:, 0] ** 2
        scores = distance_correlation(x, y, n_subsample=10_000, random_state=0)
        assert np.isfinite(scores).all()

    def test_more_replicates_reduce_run_to_run_spread(self, rng):
        """Subsampling is a variance source; replicate averaging must damp it."""
        x = rng.normal(size=(4000, 1))
        y = x[:, 0] ** 2

        def spread(n_replicates):
            vals = [
                distance_correlation(
                    x, y, n_subsample=150, n_replicates=n_replicates, random_state=s
                )[0]
                for s in range(6)
            ]
            return float(np.std(vals))

        assert spread(8) < spread(1)


class TestDependenceReportDcor:
    """``dependence_report`` must expose dCor with its subsampling controls."""

    def test_reports_dcor_with_a_chance_floor(self, rng):
        x = rng.normal(size=(1500, 3))
        y = rng.normal(size=1500)
        rep = dependence_report(x, y, "dcor", n_subsample=250, random_state=0)
        assert rep["metric"] == "dcor"
        assert rep["per_dim"].shape == (3,)
        # dCor is positively biased at small n, so the null floor must be well
        # above zero and reported — otherwise 0.15 reads as real dependence.
        assert rep["null_level"] > 0.0
        assert rep["max"] < 3 * rep["null_level"]

    def test_null_floor_shrinks_as_the_subsample_grows(self, rng):
        x = rng.normal(size=(4000, 2))
        y = rng.normal(size=4000)
        small = dependence_report(x, y, "dcor", n_subsample=100, random_state=0)
        large = dependence_report(x, y, "dcor", n_subsample=600, random_state=0)
        assert large["null_level"] < small["null_level"]

    def test_records_the_subsample_actually_used(self, rng):
        x = rng.normal(size=(5000, 2))
        y = rng.normal(size=5000)
        rep = dependence_report(x, y, "dcor", n_subsample=400, random_state=0)
        assert rep["n_subsample"] == 400
        assert rep["n_samples"] == 5000


class TestDependenceReport:
    def test_summary_is_consistent_with_per_dim(self, rng):
        y = rng.normal(size=300)
        x = np.column_stack([y, rng.normal(size=300), 0.5 * y])
        report = dependence_report(x, y, "pearson", threshold=0.4)
        per_dim = report["per_dim"]
        assert report["max"] == pytest.approx(per_dim.max())
        assert report["mean"] == pytest.approx(per_dim.mean())
        assert report["argmax"] == int(per_dim.argmax())
        assert report["n_above"] == int((per_dim > 0.4).sum())
        assert report["n_samples"] == 300

    def test_eta_metric_reports_unsquared_scale(self, rng):
        labels = rng.integers(0, 3, size=300)
        x = rng.normal(size=(300, 4)) + labels[:, None]
        report = dependence_report(x, labels, "eta")
        expected = correlation_ratio(x, labels, squared=False)
        assert np.allclose(report["per_dim"], expected)

    @pytest.mark.parametrize("metric", ["partial_pearson", "partial_spearman"])
    def test_partial_metrics_use_the_control(self, rng, metric):
        z = rng.normal(size=1000)
        x = (z + 0.1 * rng.normal(size=1000)).reshape(-1, 1)
        y = z + 0.1 * rng.normal(size=1000)
        report = dependence_report(x, y, metric, control=z)
        assert report["max"] < 0.15

    def test_null_level_shrinks_with_sample_size(self, rng):
        small = dependence_report(rng.normal(size=(50, 3)), rng.normal(size=50))
        large = dependence_report(rng.normal(size=(5000, 3)), rng.normal(size=5000))
        assert 0 < large["null_level"] < small["null_level"] < 1

    def test_rejects_an_unknown_metric(self, rng):
        with pytest.raises(ValueError, match="invalid metric"):
            dependence_report(rng.normal(size=(10, 2)), rng.normal(size=10), "kendall")

    def test_partial_metric_requires_a_control(self, rng):
        with pytest.raises(ValueError, match="requires `control`"):
            dependence_report(
                rng.normal(size=(10, 2)), rng.normal(size=10), "partial_pearson"
            )


class TestIcaHelpersStayEquivalent:
    """``drop_top_k_dependent`` must keep ranking sources exactly as before."""

    def test_abs_correlation_matches(self, rng):
        y = rng.normal(size=200)
        x = rng.normal(size=(200, 5))
        assert np.allclose(_abs_correlation(x, y), abs_pearson(x, y))

    def test_correlation_ratio_matches(self, rng):
        labels = rng.integers(0, 4, size=200)
        x = rng.normal(size=(200, 5)) + labels[:, None]
        assert np.allclose(
            _correlation_ratio(x, labels), correlation_ratio(x, labels, squared=True)
        )
