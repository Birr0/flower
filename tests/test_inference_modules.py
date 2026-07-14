"""Tests for flower.inference.modules — inference helper functions."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from hydra.core.global_hydra import GlobalHydra

from flower.inference.modules import (
    convert_to_np,
    create_embeddings,
    create_lightning_loader,
    create_samples,
    wandb_format,
)  # type: ignore[import-untyped]


class TestConvertToNp:
    def test_single_tensor(self):
        t = torch.randn(4, 8)
        result = convert_to_np([t])
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], np.ndarray)
        assert result[0].shape == (4, 8)

    def test_multiple_tensors(self):
        t1 = torch.randn(4, 8)
        t2 = torch.randn(4, 12)
        result = convert_to_np([t1, t2])
        assert len(result) == 2
        assert result[0].shape == (4, 8)
        assert result[1].shape == (4, 12)

    def test_cpu_device(self):
        t = torch.randn(2, 3)
        result = convert_to_np([t])
        assert isinstance(result[0], np.ndarray)
        assert result[0].dtype in (np.float32, np.float64)

    def test_preserves_values(self):
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = convert_to_np([t])
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_almost_equal(result[0], expected)


class TestCreateEmbeddings:
    def test_returns_hf_dataset(self, dummy_prediction_dict):
        result = create_embeddings(dummy_prediction_dict, "test")
        assert result is not None
        # HuggingFace Dataset objects have a __len__ and __getitem__
        assert len(result) == 4

    def test_contains_expected_columns(self, dummy_prediction_dict):
        result = create_embeddings(dummy_prediction_dict, "test")
        column_names = result.column_names
        # catalog keys are converted to columns
        assert "r" in column_names
        assert "g" in column_names
        assert "b" in column_names
        assert "digit" in column_names
        # z is added
        assert "z" in column_names
        # X and y are preserved from predictions
        assert "X" in column_names
        assert "y" in column_names

    def test_catalog_removed_from_dict(self, dummy_prediction_dict):
        """Remove catalog key, replace with individual columns."""
        # NOTE: copy is imported below where needed

        pred_copy = copy.deepcopy(dummy_prediction_dict)
        assert "catalog" in pred_copy
        create_embeddings(pred_copy, "test")
        assert "catalog" not in pred_copy

    def test_single_item(self):
        t = torch.randn(1, 8)
        pred = {
            "X": t,
            "y": torch.randn(1, 4),
            "z": torch.randn(1, 16),
            "catalog": {"a": torch.tensor([1.0])},
        }
        result = create_embeddings(pred, "train")
        assert len(result) == 1


class TestCreateSamples:
    def test_returns_hf_dataset(self, dummy_samples_dict):
        result = create_samples(dummy_samples_dict, "test")
        assert result is not None
        assert len(result) == 4

    def test_contains_z_prime(self, dummy_samples_dict):
        result = create_samples(dummy_samples_dict, "test")
        column_names = result.column_names
        assert "z_prime" in column_names

    def test_catalog_removed_from_dict(self, dummy_samples_dict):
        # NOTE: copy is imported below where needed

        samples_copy = copy.deepcopy(dummy_samples_dict)
        assert "catalog" in samples_copy
        create_samples(samples_copy, "test")
        assert "catalog" not in samples_copy

    def test_single_item(self):
        t = torch.randn(1, 8)
        samples = {
            "X": t,
            "y": torch.randn(1, 4),
            "z": torch.randn(1, 16),
            "z_prime": torch.randn(1, 16),
            "catalog": {"a": torch.tensor([1.0])},
        }
        result = create_samples(samples, "train")
        assert len(result) == 1


class TestWandbFormat:
    def test_returns_dataframe(self, dummy_prediction_dict):
        ds = create_embeddings(dummy_prediction_dict, "test")
        result = wandb_format(ds, {"type": "image"})
        assert result is not None
        assert hasattr(result, "columns")

    def test_has_columns(self, dummy_prediction_dict):
        ds = create_embeddings(dummy_prediction_dict, "test")
        result = wandb_format(ds, {"type": "image"})
        assert len(result.columns) > 0

    def test_contains_z_column(self, dummy_prediction_dict):
        ds = create_embeddings(dummy_prediction_dict, "test")
        result = wandb_format(ds, {"type": "image"})
        assert "z" in result.columns


class TestCreateLightningLoader:
    """create_lightning_loader requires Hydra config;
    test that it raises on missing config."""

    def test_raises_without_hydra_config(self):
        """Should raise when Hydra is not configured."""

        GlobalHydra.instance().clear()
        with pytest.raises(AttributeError):
            create_lightning_loader(None)
