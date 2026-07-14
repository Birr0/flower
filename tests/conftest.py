"""Shared pytest fixtures for the flower test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributions as d
from sklearn.neural_network import MLPClassifier, MLPRegressor

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="session")
def device():
    """Always use CPU for tests."""
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Minimal y_catalog dicts (used by model module helpers)
# ---------------------------------------------------------------------------


@pytest.fixture
def rgbmnist_catalog():
    """y_catalog matching src/conf/data/rgbmnist.yaml."""
    return {
        "variables": {
            "r": {"name": "r", "size": 1, "processing_fn": None, "continuous": 1},
            "g": {"name": "g", "size": 1, "processing_fn": None, "continuous": 1},
            "b": {"name": "b", "size": 1, "processing_fn": None, "continuous": 1},
            "digit": {
                "name": "digit",
                "size": 10,
                "processing_fn": None,
                "continuous": 0,
            },
        },
        "drop_variables": ["b"],
    }


@pytest.fixture
def dsprites_catalog():
    """y_catalog matching src/conf/data/dsprites.yaml (no drops)."""
    return {
        "variables": {
            "label_shape": {
                "name": "label_shape",
                "size": 1,
                "processing_fn": None,
                "continuous": 0,
            },
            "value_x_position": {
                "name": "value_x_position",
                "size": 1,
                "processing_fn": None,
                "continuous": 1,
            },
            "value_y_position": {
                "name": "value_y_position",
                "size": 1,
                "processing_fn": None,
                "continuous": 1,
            },
            "value_orientation_sin": {
                "name": "value_orientation_sin",
                "size": 1,
                "processing_fn": None,
                "continuous": 1,
            },
            "value_orientation_cos": {
                "name": "value_orientation_cos",
                "size": 1,
                "processing_fn": None,
                "continuous": 1,
            },
            "value_scale": {
                "name": "value_scale",
                "size": 1,
                "processing_fn": None,
                "continuous": 1,
            },
        },
        "drop_variables": [],
        "continuous": [
            "label_shape",
            "value_x_position",
            "value_y_position",
            "value_orientation_sin",
            "value_orientation_cos",
            "value_scale",
        ],
        "discrete": [],
    }


# ---------------------------------------------------------------------------
# Dummy dataset for FlowerDataset / FlowerDataLoader
# ---------------------------------------------------------------------------


class _DummyDataset:
    """Tiny synthetic dataset that returns X (tensor) and catalog (dict)."""

    def __init__(self, n: int = 32, img_size: int = 8, channels: int = 1):
        self.n = n
        self.img_size = img_size
        self.channels = channels
        self.y_catalog = {
            "variables": {
                "r": {"size": 1, "processing_fn": None},
                "g": {"size": 1, "processing_fn": None},
                "b": {"size": 1, "processing_fn": None},
                "digit": {"size": 10, "processing_fn": None},
            },
            "join_method": "concat",
            "drop_variables": ["b"],
        }

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        X = torch.randn(self.channels, self.img_size, self.img_size)
        catalog = {
            "r": torch.tensor(0.3),
            "g": torch.tensor(0.5),
            "b": torch.tensor(0.7),
            "digit": torch.tensor(idx % 10),
        }
        return {"X": X, "catalog": catalog}


@pytest.fixture
def dummy_dataset():
    return _DummyDataset(n=16, img_size=8, channels=1)


@pytest.fixture
def dummy_batch():
    """A single batch dict as returned by a FlowerDataLoader."""
    return {
        "X": torch.randn(4, 1, 8, 8),  # 1-channel (grayscale like MNIST)
        "y": torch.randn(4, 3),
        "catalog": {
            "r": torch.tensor([0.3, 0.3, 0.3, 0.3]),
            "g": torch.tensor([0.5, 0.5, 0.5, 0.5]),
            "b": torch.tensor([0.7, 0.7, 0.7, 0.7]),
            "digit": torch.tensor([0, 1, 2, 3]).unsqueeze(-1),
        },
    }


# ---------------------------------------------------------------------------
# Minimal Hydra-style config objects for model instantiation
# ---------------------------------------------------------------------------


@pytest.fixture
def vae_config():
    """Config dict for creating a VAE via hydra.utils.instantiate."""
    return {
        "_target_": "flower.models.rgbmnist.VAE",
        "hidden_dim": 64,
    }


@pytest.fixture
def dummy_lightning_vae_config(vae_config):
    """LightningVAE config."""
    return {
        "_target_": "flower.training.lightning_loaders.LightningVAE",
        "vae": vae_config,
        "lr": 1e-3,
        "batch_size": 4,
        "beta": 1.0,
    }


@pytest.fixture
def tmp_checkpoint_dir(tmp_path: Path) -> Path:
    """Create a temp directory with a fake .ckpt file for get_ckpt_files tests."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "123456_0.ckpt").touch()
    return ckpt_dir


# ---------------------------------------------------------------------------
# Torch distributions used by augmentations tests
# ---------------------------------------------------------------------------


@pytest.fixture
def uniform_dist():
    return d.Uniform(low=0.05, high=0.95)


# ---------------------------------------------------------------------------
# Evaluation metrics fixtures (sklearn pipelines, numpy arrays)
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_sklearn_mlp():
    """A small MLPClassifier for classification evaluation tests."""
    return MLPClassifier(
        hidden_layer_sizes=(16, 8),
        max_iter=10,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2,
    )


@pytest.fixture
def dummy_sklearn_mlp_regressor():
    """A small MLPRegressor for regression evaluation tests."""
    return MLPRegressor(
        hidden_layer_sizes=(16, 8),
        max_iter=10,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2,
    )


@pytest.fixture
def dummy_classification_data():
    """Simple classification data (X, y) for metrics tests."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(32, 8)
    y_train = rng.randint(0, 3, size=32)
    X_test = rng.randn(16, 8)
    y_test = rng.randint(0, 3, size=16)
    return X_train, y_train, X_test, y_test


@pytest.fixture
def dummy_regression_data():
    """Simple regression data (X, y) for metrics tests."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(32, 8)
    y_train = rng.randn(32)
    X_test = rng.randn(16, 8)
    y_test = rng.randn(16)
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Inference helper fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_prediction_dict():
    """A minimal predictions dict as returned by a Lightning step."""
    return {
        "X": torch.randn(4, 1, 8, 8),
        "y": torch.randn(4, 12),
        "z": torch.randn(4, 64),
        "catalog": {
            "r": torch.tensor([0.3, 0.3, 0.3, 0.3]),
            "g": torch.tensor([0.5, 0.5, 0.5, 0.5]),
            "b": torch.tensor([0.7, 0.7, 0.7, 0.7]),
            "digit": torch.tensor([0, 1, 2, 3]).unsqueeze(-1),
        },
    }


@pytest.fixture
def dummy_samples_dict():
    """A minimal samples dict as returned by a sampling step (includes z_prime)."""
    return {
        "X": torch.randn(4, 1, 8, 8),
        "y": torch.randn(4, 12),
        "z": torch.randn(4, 64),
        "z_prime": torch.randn(4, 64),
        "catalog": {
            "r": torch.tensor([0.3, 0.3, 0.3, 0.3]),
            "g": torch.tensor([0.5, 0.5, 0.5, 0.5]),
            "b": torch.tensor([0.7, 0.7, 0.7, 0.7]),
            "digit": torch.tensor([0, 1, 2, 3]).unsqueeze(-1),
        },
    }
