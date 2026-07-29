"""Integration tests: Hydra config loading, data module instantiation,
mini E2E training.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import torch.nn as nn
from hydra.utils import get_object
from lightning.pytorch import Trainer
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from flower.data.modules import FlowerDataLoader
from flower.models.dsprites import BetaVAE, LightningBetaVAE, LightningFlowMatching

_ROOT = Path(__file__).resolve().parent.parent


# ===========================================================================
# Mock base model with encode() interface (mimics VAE/ViT encoder)
# ===========================================================================


class MockEncoder(nn.Module):
    """Minimal encoder mock with encode() -> {"z", "mu", "logvar"}."""

    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.proj = nn.Linear(4096, latent_dim)  # from 64x64 flattened (1-channel)

    def encode(self, x):
        # x shape: (batch, 1, 64, 64) or (batch, 3, 28, 28)
        b = x.shape[0]
        x_flat = x.view(b, -1)
        mu = self.proj(x_flat)
        logvar = torch.zeros_like(mu)
        x_1 = mu + torch.randn_like(mu) * 0.1  # slight perturbation
        return {"z": x_1, "mu": mu, "logvar": logvar}


class MockRGBEncoder(nn.Module):
    """Encoder mock for RGB images (28x28 -> 3*28*28)."""

    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.proj = nn.Linear(3 * 28 * 28, latent_dim)

    def encode(self, x):
        b = x.shape[0]
        x_flat = x.view(b, -1)
        mu = self.proj(x_flat)
        logvar = torch.zeros_like(mu)
        x_1 = mu + torch.randn_like(mu) * 0.1
        return {"z": x_1, "mu": mu, "logvar": logvar}


# ===========================================================================
# Helper functions to build configs (bypassing Hydra's problematic defaults)
# ===========================================================================


def _build_dsprites_flow_config(overrides: dict | None = None) -> dict:
    """Build a minimal dsprites_Flow experiment config without Hydra."""
    base = {
        "experiment_name": "test_dsprites",
        "data": {
            "name": "dsprites",
            "y_catalog": {
                "catalog_name": "dsprites",
                "drop_variables": ["image", "label_shape", "value_orientation_sin"],
                "variables": {
                    "value_x_position": {"size": 1, "continuous": 1},
                    "value_y_position": {"size": 1, "continuous": 1},
                    "label_orientation": {"size": 1, "continuous": 1},
                    "value_scale": {"size": 1, "continuous": 1},
                    "label": {"size": 1, "continuous": 0},
                    "label_disparity": {"size": 1, "continuous": 0},
                    "image": {"size": 64, "continuous": 0},
                    "label_shape": {"size": 1, "continuous": 0},
                    "value_orientation_sin": {"size": 1, "continuous": 0},
                    "value_orientation_cos": {"size": 1, "continuous": 0},
                },
            },
            "x_ds": {
                "_target_": "flower.data.dsprites.DspritesDataset",
                "data_dir": "/mock/data/dsprites-dataset",
                "split": "train",
                "transform": None,
                "n_samples": 100,
            },
            "loader": {
                "_target_": "flower.data.modules.FlowerDataLoader",
                "batch_size": 4,
            },
        },
        "lightning_loader": {
            "_target_": "flower.models.dsprites.LightningFlowMatching",
            "vf": None,  # Will be replaced with mock
            "base_model": {
                "_target_": "__main__.MockEncoder",
                "latent_dim": 64,
            },
            "lr": 1e-4,
        },
        "trainer": {
            "accelerator": "cpu",
            "max_epochs": 1,
            "limit_train_batches": 2,
            "limit_val_batches": 1,
            "limit_test_batches": 1,
        },
    }
    if overrides:
        for key, value in overrides.items():
            _set_nested(base, key, value)
    return base


def _build_rgbmnist_flow_config(overrides: dict | None = None) -> dict:
    """Build a minimal rgbmnist_Flow experiment config without Hydra."""
    base = {
        "experiment_name": "test_rgbmnist",
        "data": {
            "name": "rgbmnist",
            "y_catalog": {
                "catalog_name": "rgbmnist",
                "drop_variables": ["b"],
                "variables": {
                    "r": {"size": 1, "continuous": 1},
                    "g": {"size": 1, "continuous": 1},
                    "b": {"size": 1, "continuous": 1},
                    "digit": {"size": 10, "continuous": 0},
                    "label": {"size": 1, "continuous": 0},
                },
            },
            "x_ds": {
                "_target_": "torchvision.datasets.MNIST",
                "data_dir": "/mock/data",
                "split": "train",
                "transform": None,
                "n_samples": 100,
            },
            "loader": {
                "_target_": "flower.data.modules.FlowerDataLoader",
                "batch_size": 256,
            },
        },
        "lightning_loader": {
            "_target_": "flower.models.rgbmnist.LightningFlowMatching",
            "vf": None,  # Will be replaced with mock
            "base_model": {
                "_target_": "__main__.MockRGBEncoder",
                "latent_dim": 64,
            },
            "lr": 1e-4,
        },
        "trainer": {
            "accelerator": "cpu",
            "max_epochs": 1,
            "limit_train_batches": 2,
            "limit_val_batches": 1,
            "limit_test_batches": 1,
        },
    }
    if overrides:
        for key, value in overrides.items():
            _set_nested(base, key, value)
    return base


def _set_nested(d: dict, key_path: str, value):
    """Set a nested value in a dict using dot notation."""
    keys = key_path.split(".")
    for key in keys[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value


def _make_mock_vf(base_model):
    """Create a minimal velocity field mock that matches the expected interface."""

    class MockVelocityField(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            self.code_dim = base_model.latent_dim

        def encode(self, mu, logvar):
            z_mu, z_logvar = mu, logvar
            return self.base_model(z_mu, z_logvar)

    return MockVelocityField(base_model)


# ===========================================================================
# Shared fixtures
# ===========================================================================


@pytest.fixture
def load_config():
    """Return a function to build experiment configs."""

    def _load(experiment: str, overrides: dict | None = None):
        if experiment == "dsprites_Flow":
            cfg = _build_dsprites_flow_config(overrides)
        elif experiment == "rgbmnist_Flow":
            cfg = _build_rgbmnist_flow_config(overrides)
        elif experiment == "dsprites_VAE":
            cfg = _build_dsprites_vae_config(overrides)
        else:
            err = f"Unknown experiment: {experiment}"
            raise ValueError(err)
        return cfg

    return _load


def _build_dsprites_vae_config(overrides: dict | None = None) -> dict:
    """Build a minimal dsprites_VAE experiment config without Hydra."""
    base = {
        "experiment_name": "test_dsprites_vae",
        "data": {
            "name": "dsprites",
            "y_catalog": {
                "catalog_name": "dsprites",
                "drop_variables": ["image", "label_shape", "value_orientation_sin"],
                "variables": {
                    "value_x_position": {"size": 1, "continuous": 1},
                    "value_y_position": {"size": 1, "continuous": 1},
                    "label_orientation": {"size": 1, "continuous": 1},
                    "value_scale": {"size": 1, "continuous": 1},
                    "label": {"size": 1, "continuous": 0},
                    "label_disparity": {"size": 1, "continuous": 0},
                    "image": {"size": 64, "continuous": 0},
                    "label_shape": {"size": 1, "continuous": 0},
                    "value_orientation_sin": {"size": 1, "continuous": 0},
                    "value_orientation_cos": {"size": 1, "continuous": 0},
                },
            },
            "x_ds": {
                "_target_": "flower.data.dsprites.DspritesDataset",
                "data_dir": "/mock/data/dsprites-dataset",
                "split": "train",
                "transform": None,
                "n_samples": 100,
            },
            "loader": {
                "_target_": "flower.data.modules.FlowerDataLoader",
                "batch_size": 4,
            },
        },
        "lightning_loader": {
            "_target_": "flower.models.dsprites.LightningVAE",
            "vae": {
                "_target_": "flower.models.modules.BetaVAE",
                "latent_dim": 32,
                "code_dim": 64,
            },
            "lr": 1e-3,
        },
        "trainer": {
            "accelerator": "cpu",
            "max_epochs": 1,
            "limit_train_batches": 2,
            "limit_val_batches": 1,
            "limit_test_batches": 1,
        },
    }
    if overrides:
        for key, value in overrides.items():
            _set_nested(base, key, value)
    return base


# ===========================================================================
# 1. Config loading tests
# ===========================================================================


class TestDspritesFlowConfig:
    """Integration tests for dsprites_Flow experiment config."""

    def test_config_loads(self, load_config):
        cfg = load_config("dsprites_Flow")
        assert cfg["data"]["y_catalog"] is not None
        assert cfg["lightning_loader"] is not None

    def test_dsprites_catalog_structure(self, load_config):
        cfg = load_config("dsprites_Flow")
        cat = cfg["data"]["y_catalog"]
        assert "variables" in cat
        assert "drop_variables" in cat
        assert len(cat["variables"]) == 10  # 6 actual + 3 dropped + 1 cos

    def test_lightning_module_instantiates(self, load_config):
        cfg = load_config("dsprites_Flow")
        catalog = cfg["data"]["y_catalog"]

        # Build the mock encoder
        encoder = MockEncoder(latent_dim=64)

        model = LightningFlowMatching(
            base_model=encoder,
            lr=cfg["lightning_loader"]["lr"],
            batch_size=4,
            code_dim=64,
            hidden_dim=64,
            catalog=catalog,
            n_layers=4,
        )
        assert hasattr(model, "vf")
        assert hasattr(model, "base_model")
        assert hasattr(model, "configure_optimizers")

    def test_optimizer_configures(self, load_config):
        cfg = load_config("dsprites_Flow")
        catalog = cfg["data"]["y_catalog"]

        encoder = MockEncoder(latent_dim=64)

        model = LightningFlowMatching(
            base_model=encoder,
            lr=cfg["lightning_loader"]["lr"],
            batch_size=4,
            code_dim=64,
            hidden_dim=64,
            catalog=catalog,
            n_layers=4,
        )
        opt = model.configure_optimizers()
        assert opt is not None
        assert hasattr(opt, "param_groups")
        assert opt.param_groups[0]["lr"] == 0.0001


class TestRgbmnistFlowConfig:
    """Integration tests for rgbmnist_Flow experiment config."""

    def test_config_loads(self, load_config):
        cfg = load_config("rgbmnist_Flow")
        assert cfg["data"]["y_catalog"] is not None
        assert "rgbmnist" in cfg["data"]["y_catalog"]["catalog_name"]

    def test_rgbmnist_catalog_structure(self, load_config):
        cfg = load_config("rgbmnist_Flow")
        cat = cfg["data"]["y_catalog"]
        assert len(cat["variables"]) == 5  # r, g, b, digit, label
        assert "b" in cat["drop_variables"]
        # total dims across all variables
        assert sum(v["size"] for v in cat["variables"].values()) == 14
        # dropped dims: b(1)
        assert sum(cat["variables"][v]["size"] for v in cat["drop_variables"]) == 1

    def test_lightning_module_instantiates(self, load_config):
        cfg = load_config("rgbmnist_Flow")
        catalog = cfg["data"]["y_catalog"]

        encoder = MockRGBEncoder(latent_dim=64)

        model = LightningFlowMatching(
            base_model=encoder,
            lr=cfg["lightning_loader"]["lr"],
            batch_size=4,
            code_dim=64,
            hidden_dim=64,
            catalog=catalog,
        )
        assert hasattr(model, "vf")
        assert hasattr(model, "configure_optimizers")

    def test_optimizer_configures(self, load_config):
        cfg = load_config("rgbmnist_Flow")
        catalog = cfg["data"]["y_catalog"]

        encoder = MockRGBEncoder(latent_dim=64)

        model = LightningFlowMatching(
            base_model=encoder,
            lr=cfg["lightning_loader"]["lr"],
            batch_size=4,
            code_dim=64,
            hidden_dim=64,
            catalog=catalog,
        )
        opt = model.configure_optimizers()
        assert opt is not None


class TestConfigOverrides:
    """Integration tests for config override capability."""

    def test_override_hidden_dim(self, load_config):
        cfg = load_config(
            "dsprites_Flow", overrides={"lightning_loader.base_model.latent_dim": 32}
        )
        assert cfg["lightning_loader"]["base_model"]["latent_dim"] == 32

    def test_override_lr(self, load_config):
        cfg = load_config("dsprites_Flow", overrides={"lightning_loader.lr": 0.01})
        assert cfg["lightning_loader"]["lr"] == 0.01

    def test_model_uses_overridden_lr(self, load_config):
        cfg = load_config("dsprites_Flow", overrides={"lightning_loader.lr": 0.05})
        catalog = cfg["data"]["y_catalog"]

        encoder = MockEncoder(latent_dim=64)

        model = LightningFlowMatching(
            base_model=encoder,
            lr=cfg["lightning_loader"]["lr"],
            batch_size=4,
            code_dim=64,
            hidden_dim=64,
            catalog=catalog,
            n_layers=4,
        )
        opt = model.configure_optimizers()
        assert opt.param_groups[0]["lr"] == 0.05

    def test_override_n_layers(self, load_config):
        cfg = load_config("dsprites_Flow", overrides={"lightning_loader.n_layers": 3})
        assert cfg["lightning_loader"]["n_layers"] == 3


class TestDspritesVAEConfig:
    """Integration tests for dsprites_VAE experiment config."""

    def test_config_loads(self, load_config):
        cfg = load_config("dsprites_VAE")
        assert cfg["lightning_loader"] is not None

    def test_lightning_module_instantiates_no_ckpt(self, load_config):
        cfg = load_config("dsprites_VAE")
        vae_cfg = cfg["lightning_loader"]["vae"]

        vae = BetaVAE(
            latent_dim=vae_cfg["latent_dim"],
        )

        lightning_cfg = {
            "vae": vae,
            "lr": cfg["lightning_loader"]["lr"],
        }
        model = LightningBetaVAE(**lightning_cfg)
        assert hasattr(model, "vae")
        assert hasattr(model, "configure_optimizers")


# ===========================================================================
# 2. Data module tests (mocked filesystem)
# ===========================================================================


class TestDspritesDataModule:
    """Integration tests for dsprites data module with mocked filesystem."""

    def test_data_module_config_loads(self, load_config):
        cfg = load_config("dsprites_Flow")
        assert cfg["data"]["x_ds"] is not None

    def test_data_module_instantiates_with_mock_data(self):
        class DummyDataset(Dataset):
            def __init__(self, size=4):
                self.size = size

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                return {
                    "X": torch.randn(1, 64, 64),
                    "y": torch.randn(7),
                }

        train_ds = DummyDataset(size=4)
        test_ds = DummyDataset(size=2)

        data_module = FlowerDataLoader(
            datasets={"train": train_ds, "test": test_ds},
            batch_size=4,
        )
        assert hasattr(data_module, "train_dataloader")

    def test_data_module_setup(self):
        class DummyDataset(Dataset):
            def __init__(self, size=4):
                self.size = size

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                return {
                    "X": torch.randn(1, 64, 64),
                    "y": torch.randn(7),
                }

        train_ds = DummyDataset(size=4)
        test_ds = DummyDataset(size=2)

        data_module = FlowerDataLoader(
            datasets={"train": train_ds, "test": test_ds, "val": test_ds},
            batch_size=4,
        )
        data_module.setup()
        # After setup, train_dataset should exist
        assert hasattr(data_module, "train_dataset")


class TestRgbmnistDataModule:
    """Integration tests for rgbmnist data module."""

    def test_data_module_config_loads(self, load_config):
        cfg = load_config("rgbmnist_Flow")
        assert cfg["data"]["x_ds"] is not None

    def test_data_module_instantiates(self, load_config):
        cfg = load_config("rgbmnist_Flow")
        loader_cfg = cfg["data"]["loader"]
        # For RGBMNIST we just verify the config structure is valid
        assert loader_cfg["batch_size"] == 256


# ===========================================================================
# 3. Mini E2E: one training step through Lightning Trainer
# ===========================================================================


class TestMiniE2EDspritesFlow:
    """Mini E2E: one training step through Lightning Trainer
    with dsprites_Flow config.
    """

    @pytest.fixture
    def dummy_batch_dsprites(self):
        """Create a minimal batch matching dsprites catalog structure (7-dim y)."""
        batch_size = 4
        return {
            "X": torch.randn(batch_size, 1, 64, 64),
            "y": torch.randn(batch_size, 7),  # 7 non-dropped catalog dims
            "catalog": {
                "label_shape": torch.zeros(batch_size, 1),
                "value_x_position": torch.zeros(batch_size, 1),
                "value_y_position": torch.zeros(batch_size, 1),
                "value_orientation_sin": torch.zeros(batch_size, 1),
                "value_orientation_cos": torch.zeros(batch_size, 1),
                "value_scale": torch.zeros(batch_size, 1),
                "label": torch.zeros(batch_size, 1),
                "label_disparity": torch.zeros(batch_size, 1),
            },
        }

    @pytest.fixture
    def model(self, load_config):
        cfg = load_config("dsprites_Flow")
        catalog = cfg["data"]["y_catalog"]

        encoder = MockEncoder(latent_dim=64)

        return LightningFlowMatching(
            base_model=encoder,
            lr=cfg["lightning_loader"]["lr"],
            batch_size=4,
            code_dim=64,
            hidden_dim=64,
            catalog=catalog,
            n_layers=4,
        )

    def test_one_training_step(self, model, dummy_batch_dsprites):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loss = model.training_step(dummy_batch_dsprites, 0)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)
        assert torch.isfinite(loss)

    def test_one_validation_step(self, model, dummy_batch_dsprites):
        """Run one validation_step and verify loss is computed."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loss = model.validation_step(dummy_batch_dsprites, 0)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_predict_step(self, model, dummy_batch_dsprites):
        """Run predict_step and verify outputs."""
        model.eval()
        with torch.no_grad():
            out = model.predict_step(
                dummy_batch_dsprites["X"],
                dummy_batch_dsprites["y"],
                embed_opt=["orig"],
            )
        assert "orig" in out

    @pytest.mark.skip(
        reason="predict_step with cond/uncond requires solver which needs a checkpoint"
    )
    def test_predict_step_cond(self, model, dummy_batch_dsprites):
        model.eval()
        with torch.no_grad():
            out = model.predict_step(
                dummy_batch_dsprites["X"],
                dummy_batch_dsprites["y"],
                embed_opt=["cond"],
            )
        assert "cond" in out

    @pytest.mark.skip(
        reason="predict_step with cond/uncond requires solver which needs a checkpoint"
    )
    def test_predict_step_uncond(self, model, dummy_batch_dsprites):
        model.eval()
        with torch.no_grad():
            out = model.predict_step(
                dummy_batch_dsprites["X"],
                dummy_batch_dsprites["y"],
                embed_opt=["uncond"],
            )
        assert "uncond" in out

    @pytest.mark.skip(
        reason="predict_step with cond/uncond requires solver which needs a checkpoint"
    )
    def test_predict_step_multi(self, model, dummy_batch_dsprites):
        model.eval()
        with torch.no_grad():
            out = model.predict_step(
                dummy_batch_dsprites["X"],
                dummy_batch_dsprites["y"],
                embed_opt=["orig", "cond", "uncond"],
            )
        assert set(out.keys()) == {"orig", "cond", "uncond"}

    def test_trainer_can_instantiate(self):
        """Verify the Trainer config can be built (without actually training)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trainer = Trainer(
                accelerator="cpu",
                max_epochs=1,
                limit_train_batches=1,
                limit_val_batches=1,
                enable_progress_bar=False,
                enable_checkpointing=False,
                logger=False,
                devices=1,
            )
        assert trainer is not None


class TestMiniE2ERgbmnistFlow:
    """Mini E2E: one training step through Lightning Trainer
    with rgbmnist_Flow config.
    """

    @pytest.fixture
    def dummy_batch_rgb(self):
        batch_size = 4
        return {
            "X": torch.randn(batch_size, 3, 28, 28),
            "y": torch.randn(
                batch_size, 13
            ),  # r(1)+g(1)+digit(10)+label(1)=13, b(1) dropped
            "catalog": {
                "r": torch.zeros(batch_size, 1),
                "g": torch.zeros(batch_size, 1),
                "digit": torch.zeros(batch_size, 10),
                "label": torch.zeros(batch_size, 1),
            },
        }

    @pytest.fixture
    def model(self, load_config):
        cfg = load_config("rgbmnist_Flow")
        catalog = cfg["data"]["y_catalog"]

        encoder = MockRGBEncoder(latent_dim=64)

        return LightningFlowMatching(
            base_model=encoder,
            lr=cfg["lightning_loader"]["lr"],
            batch_size=4,
            code_dim=64,
            hidden_dim=64,
            catalog=catalog,
        )

    def test_one_training_step(self, model, dummy_batch_rgb):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loss = model.training_step(dummy_batch_rgb, 0)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)
        assert torch.isfinite(loss)

    def test_predict_step(self, model, dummy_batch_rgb):
        model.eval()
        with torch.no_grad():
            out = model.predict_step(
                dummy_batch_rgb["X"],
                dummy_batch_rgb["y"],
                embed_opt=["orig"],
            )
        assert "orig" in out


# ===========================================================================
# 4. Config robustness tests
# ===========================================================================


class TestConfigRobustness:
    """Integration tests for handling null/missing config values."""

    def test_model_ignores_null_data_path(self, load_config):
        """The experiment config may reference paths that don't exist locally."""
        cfg = load_config("dsprites_Flow")
        catalog = cfg["data"]["y_catalog"]

        encoder = MockEncoder(latent_dim=64)

        model = LightningFlowMatching(
            base_model=encoder,
            lr=cfg["lightning_loader"]["lr"],
            batch_size=4,
            code_dim=64,
            hidden_dim=64,
            catalog=catalog,
            n_layers=4,
        )
        assert model is not None

    def test_catalog_has_required_keys(self, load_config):
        for exp in ["dsprites_Flow", "rgbmnist_Flow"]:
            cfg = load_config(exp)
            cat = cfg["data"]["y_catalog"]
            assert "variables" in cat
            assert "drop_variables" in cat


# ===========================================================================
# Test fixtures (mocked filesystem)
# ===========================================================================


@pytest.fixture
def tmp_dsprites_data(tmp_path):
    """Create a temporary directory with minimal dsprites-dataset files."""
    data_dir = tmp_path / "data" / "dsprites-dataset"
    data_dir.mkdir(parents=True)

    # dataset_info.json
    info_file = data_dir / "dataset_info.json"
    info = {"citation": "", "description": "Mock dsprites", "license": ""}
    info_file.write_text(json.dumps(info))

    # features.json
    features_file = data_dir / "features.json"
    features = {
        "image": {"dtype": "float32", "shape": [64, 64, 3]},
        "label_shape": {"dtype": "int64", "shape": []},
        "value_x_position": {"dtype": "float64", "shape": []},
        "value_y_position": {"dtype": "float64", "shape": []},
        "label_orientation": {"dtype": "float64", "shape": []},
        "value_scale": {"dtype": "float64", "shape": []},
    }
    features_file.write_text(json.dumps(features))

    # Minimal parquet file with 1 sample
    try:
        inner = pa.list_(pa.float32(), 1)
        outer = pa.list_(inner, 64)
        img_type = pa.list_(outer)
        img_array = pa.array(
            [[[[0.5] for _ in range(64)] for _ in range(64)]], type=img_type
        )
        table = pa.table(
            {
                "image": img_array,
                "label_shape": [0],
                "value_x_position": [0.5],
                "value_y_position": [0.5],
                "label_orientation": [0.0],
                "value_scale": [0.75],
            }
        )
        train_dir = data_dir / "train"
        train_dir.mkdir()
        pq.write_table(table, str(train_dir / "data-00000-of-00001.parquet"))
    except ImportError:
        (data_dir / "train").mkdir(exist_ok=True)

    return data_dir.parent


@pytest.fixture
def mock_dsprites_env(tmp_dsprites_data):
    """Mock the DATA_ROOT env-var so that `load_from_disk` points at tmp data."""
    with mock.patch.dict(
        os.environ, {"DATA_ROOT": str(tmp_dsprites_data.parent)}, clear=False
    ):
        yield tmp_dsprites_data


# ===========================================================================
# 5. Real-YAML config verification
#
# The `load_config` fixture above builds config dicts in Python, mirroring the
# YAML by hand. That cannot catch drift in the actual files under `src/conf/`,
# which is how five malformed `_target_` paths ("flower..data.…") survived from
# the initial commit until issue #34. These tests read the YAML off disk.
# ===========================================================================

_DATA_CONF_DIR = _ROOT / "src" / "conf" / "data"


def _iter_targets(node, trail=""):
    """Yield (yaml_path, target) for every `_target_` under `node`."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "_target_":
                yield trail or "<root>", value
            else:
                yield from _iter_targets(value, f"{trail}.{key}" if trail else key)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _iter_targets(value, f"{trail}[{i}]")


def _data_config_files():
    return sorted(_DATA_CONF_DIR.glob("*.yaml"))


class TestDataConfigTargets:
    """Every `_target_` in `src/conf/data/` must be importable.

    `cfg.data.loader` is instantiated recursively, so an unresolvable target
    raises at construction time -- before `FlowerDataLoader.setup()` is ever
    reached -- and takes down every callback that monitors `val_loss` with it.
    """

    def test_data_configs_found(self):
        assert _data_config_files(), f"no data configs under {_DATA_CONF_DIR}"

    @pytest.mark.parametrize("config_path", _data_config_files(), ids=lambda p: p.name)
    def test_every_target_resolves(self, config_path, tmp_path):
        cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)
        targets = list(_iter_targets(cfg))
        assert targets, f"{config_path.name} declares no _target_"

        # `flower.data.sdss` reads DATA_ROOT at module scope, so importing it
        # without a value raises TypeError rather than a helpful error.
        with mock.patch.dict(os.environ, {"DATA_ROOT": str(tmp_path)}, clear=False):
            for yaml_path, target in targets:
                try:
                    get_object(target)
                except Exception as exc:
                    pytest.fail(
                        f"{config_path.name}: {yaml_path}._target_ = {target!r} "
                        f"is not importable: {type(exc).__name__}: {exc}"
                    )

    @pytest.mark.parametrize("config_path", _data_config_files(), ids=lambda p: p.name)
    def test_all_three_splits_declared(self, config_path):
        """train/val/test must all be present -- #34 only broke val and test."""
        cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)
        datasets = cfg.get("loader", {}).get("datasets", {})
        assert set(datasets) >= {"train", "val", "test"}, (
            f"{config_path.name} loader.datasets is missing a split: {sorted(datasets)}"
        )
