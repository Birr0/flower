"""Tests for flower.evaluation.metrics."""

from __future__ import annotations

import numpy as np
import pytest
from datasets import Dataset, DatasetDict
from sklearn.linear_model import LinearRegression, LogisticRegression

from flower.evaluation.metrics import (
    bootstrap_summary,
    evaluate_embedding_classifier,
    evaluate_embedding_regressor,
    prepare_data,
    print_bootstrap_stats,
)


class TestBootstrapSummary:
    def test_returns_expected_keys(self):
        result = bootstrap_summary([1.0, 2.0, 3.0])
        assert set(result.keys()) == {
            "mean",
            "median",
            "ci_95",
            "ci_68",
            "err_95",
            "scores",
        }

    def test_mean_is_correct(self):
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = bootstrap_summary(scores)
        assert result["mean"] == 3.0

    def test_median_is_correct(self):
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = bootstrap_summary(scores)
        assert result["median"] == 3.0

    def test_ci_95_is_valid_interval(self):
        rng = np.random.RandomState(42)
        scores = list(rng.randn(200))
        result = bootstrap_summary(scores)
        ci = result["ci_95"]
        assert ci[0] < ci[1]
        # The interval should bracket the mean roughly
        assert ci[0] <= result["mean"] <= ci[1]

    def test_ci_68_narrower_than_ci_95(self):
        rng = np.random.RandomState(42)
        scores = list(rng.randn(500))
        result = bootstrap_summary(scores)
        ci_95_width = result["ci_95"][1] - result["ci_95"][0]
        ci_68_width = result["ci_68"][1] - result["ci_68"][0]
        assert ci_68_width < ci_95_width

    def test_err_95_is_half_ci_width(self):
        result = bootstrap_summary([1, 2, 3, 4, 5])
        ci_95_width = result["ci_95"][1] - result["ci_95"][0]
        assert abs(result["err_95"] - ci_95_width / 2) < 1e-10

    def test_scores_preserved(self):
        scores = [1.0, 2.0, 3.0]
        result = bootstrap_summary(scores)
        np.testing.assert_array_equal(result["scores"], np.array(scores))

    def test_single_value(self):
        result = bootstrap_summary([42.0])
        assert result["mean"] == 42.0
        assert result["median"] == 42.0
        # With one value, percentile is the value itself
        assert result["ci_95"] == (42.0, 42.0)

    def test_empty_list(self):
        # filterwarnings="error" in pytest config converts warnings to exceptions,
        # so we must use a broad Exception rather than warns()
        with pytest.raises(Exception):  # noqa: B017, PT011
            bootstrap_summary([])


class TestPrintBootstrapStats:
    def test_prints_without_error(self, capsys):
        stats = bootstrap_summary([1.0, 2.0, 3.0])
        # Should not raise
        print_bootstrap_stats("TestMetric", stats)
        captured = capsys.readouterr()
        assert "TestMetric Mean:" in captured.out
        assert "95% CI:" in captured.out

    def test_output_format(self, capsys):
        stats = bootstrap_summary([0.8, 0.85, 0.9, 0.95])
        print_bootstrap_stats("Accuracy", stats)
        captured = capsys.readouterr()
        assert "Accuracy" in captured.out
        assert "Median:" in captured.out
        assert "Err 95:" in captured.out


class TestEvaluateEmbeddingClassifier:
    @pytest.fixture
    def dummy_sklearn_lr(self):
        """Lighter than MLPClassifier; avoids multi-label issues."""

        return LogisticRegression(max_iter=200, random_state=42)

    def test_returns_expected_keys(self, dummy_classification_data, dummy_sklearn_lr):
        X_train, y_train, X_test, y_test = dummy_classification_data
        result = evaluate_embedding_classifier(
            X_train,
            y_train,
            X_test,
            y_test,
            dummy_sklearn_lr,
            n_iterations=10,
            random_state=42,
        )
        assert set(result.keys()) == {
            "test_f1",
            "test_accuracy",
            "bootstrap_f1",
            "bootstrap_accuracy",
            "pipeline",
        }

    def test_f1_is_valid_range(self, dummy_classification_data, dummy_sklearn_lr):
        X_train, y_train, X_test, y_test = dummy_classification_data
        result = evaluate_embedding_classifier(
            X_train,
            y_train,
            X_test,
            y_test,
            dummy_sklearn_lr,
            n_iterations=10,
            random_state=42,
        )
        assert 0.0 <= result["test_f1"] <= 1.0

    def test_accuracy_is_valid_range(self, dummy_classification_data, dummy_sklearn_lr):
        X_train, y_train, X_test, y_test = dummy_classification_data
        result = evaluate_embedding_classifier(
            X_train,
            y_train,
            X_test,
            y_test,
            dummy_sklearn_lr,
            n_iterations=10,
            random_state=42,
        )
        assert 0.0 <= result["test_accuracy"] <= 1.0

    def test_bootstrap_f1_has_ci(self, dummy_classification_data, dummy_sklearn_lr):
        X_train, y_train, X_test, y_test = dummy_classification_data
        result = evaluate_embedding_classifier(
            X_train,
            y_train,
            X_test,
            y_test,
            dummy_sklearn_lr,
            n_iterations=10,
            random_state=42,
        )
        assert "ci_95" in result["bootstrap_f1"]

    def test_1d_input_reshaped(self, dummy_sklearn_lr):
        """1D X arrays should be reshaped to 2D."""
        rng = np.random.RandomState(42)
        X_train = rng.randn(20)
        y_train = rng.randint(0, 2, size=20)
        X_test = rng.randn(10)
        y_test = rng.randint(0, 2, size=10)
        result = evaluate_embedding_classifier(
            X_train,
            y_train,
            X_test,
            y_test,
            dummy_sklearn_lr,
            n_iterations=5,
            random_state=42,
        )
        assert "test_f1" in result

    def test_pipeline_saved(self, dummy_classification_data, dummy_sklearn_lr):
        X_train, y_train, X_test, y_test = dummy_classification_data
        result = evaluate_embedding_classifier(
            X_train,
            y_train,
            X_test,
            y_test,
            dummy_sklearn_lr,
            n_iterations=10,
            random_state=42,
        )
        assert result["pipeline"] is not None
        assert hasattr(result["pipeline"], "predict")


class TestEvaluateEmbeddingRegressor:
    @pytest.fixture
    def dummy_sklearn_lr_regressor(self):
        """LogisticRegression-based regressor won't work; use LinearRegression."""

        return LinearRegression()

    def test_returns_expected_keys(
        self,
        dummy_regression_data,
        dummy_sklearn_lr_regressor,
    ):
        X_train, y_train, X_test, y_test = dummy_regression_data
        result = evaluate_embedding_regressor(
            X_train,
            y_train,
            X_test,
            y_test,
            dummy_sklearn_lr_regressor,
            n_iterations=10,
            random_state=42,
        )
        assert set(result.keys()) == {"test_r2", "bootstrap", "pipeline"}

    def test_r2_is_finite(self, dummy_regression_data, dummy_sklearn_lr_regressor):
        X_train, y_train, X_test, y_test = dummy_regression_data
        result = evaluate_embedding_regressor(
            X_train,
            y_train,
            X_test,
            y_test,
            dummy_sklearn_lr_regressor,
            n_iterations=10,
            random_state=42,
        )
        assert np.isfinite(result["test_r2"])

    def test_bootstrap_has_ci(self, dummy_regression_data, dummy_sklearn_lr_regressor):
        X_train, y_train, X_test, y_test = dummy_regression_data
        result = evaluate_embedding_regressor(
            X_train,
            y_train,
            X_test,
            y_test,
            dummy_sklearn_lr_regressor,
            n_iterations=10,
            random_state=42,
        )
        assert "ci_95" in result["bootstrap"]

    def test_1d_input_reshaped(self, dummy_sklearn_lr_regressor):
        """1D X arrays should be reshaped to 2D."""
        rng = np.random.RandomState(42)
        X_train = rng.randn(20)
        y_train = rng.randn(20)
        X_test = rng.randn(10)
        y_test = rng.randn(10)
        result = evaluate_embedding_regressor(
            X_train,
            y_train,
            X_test,
            y_test,
            dummy_sklearn_lr_regressor,
            n_iterations=5,
            random_state=42,
        )
        assert "test_r2" in result

    def test_pipeline_saved(self, dummy_regression_data, dummy_sklearn_lr_regressor):
        X_train, y_train, X_test, y_test = dummy_regression_data
        result = evaluate_embedding_regressor(
            X_train,
            y_train,
            X_test,
            y_test,
            dummy_sklearn_lr_regressor,
            n_iterations=10,
            random_state=42,
        )
        assert result["pipeline"] is not None


class TestPrepareData:
    def test_returns_four_arrays(self):
        """prepare_data should return X_train, y_train, X_test, y_test."""

        train_data = Dataset.from_dict(
            {"embed": [[1.0, 2.0], [3.0, 4.0]], "factor": [1, 2]}
        )
        test_data = Dataset.from_dict({"embed": [[5.0, 6.0]], "factor": [3]})

        ds = DatasetDict({"train": train_data, "test": test_data})

        X_train, y_train, X_test, y_test = prepare_data(ds, "embed", "factor")

        assert X_train.shape == (2, 2)
        assert y_train.shape == (2,)
        assert X_test.shape == (1, 2)
        assert y_test.shape == (1,)

    def test_values_correct(self):
        train_data = Dataset.from_dict({"embed": [[1.0, 2.0]], "factor": [42]})
        test_data = Dataset.from_dict({"embed": [[3.0, 4.0]], "factor": [99]})

        ds = DatasetDict({"train": train_data, "test": test_data})

        X_train, y_train, X_test, y_test = prepare_data(ds, "embed", "factor")

        np.testing.assert_array_equal(X_train, [[1.0, 2.0]])
        np.testing.assert_array_equal(y_train, [42])
        np.testing.assert_array_equal(X_test, [[3.0, 4.0]])
        np.testing.assert_array_equal(y_test, [99])
