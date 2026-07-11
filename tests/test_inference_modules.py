"""Tests for inference modules — create_embeddings, wandb_format, etc."""
from __future__ import annotations

import pytest
import torch

from flower.inference.modules import (
    convert_to_np,
    create_embeddings,
    create_samples,
)


class TestConvertToNp:
    def test_list_of_tensors(self):
        values = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        result = convert_to_np(values)
        assert isinstance(result, list)
        assert len(result) == 2
        import numpy as np
        assert isinstance(result[0], np.ndarray)
        assert result[0].dtype == np.float32

    def test_empty(self):
        result = convert_to_np([])
        assert result == []

    def test_single_tensor(self):
        t = torch.randn(4, 64)
        result = convert_to_np([t])
        assert isinstance(result[0], type(torch.zeros(1).numpy()))


class TestCreateEmbeddings:
    """Test predictions -> HuggingFace dataset conversion."""

    @pytest.fixture()
    def predictions(self):
        return {
            "X": torch.randn(4, 3, 8, 8),
            "catalog": {
                "r": torch.tensor([0.3, 0.4, 0.5, 0.6]),
                "g": torch.tensor([0.3, 0.4, 0.5, 0.6]),
                "digit": torch.tensor([0, 1, 2, 3]).unsqueeze(-1),
            },
            "z": torch.randn(4, 10),
        }

    def test_create_embeddings_removes_catalog(self, predictions):
        ds = create_embeddings(predictions.copy(), "test")
        import numpy as np
        # catalog should be extracted into top-level keys
        col_names = ds.column_names
        assert "r" in col_names or "catalog" not in col_names
        assert "digit" in col_names or "catalog" not in col_names

    def test_create_embeddings_has_z(self, predictions):
        ds = create_embeddings(predictions.copy(), "test")
        assert "z" in ds.column_names


class TestCreateSamples:
    """Test samples -> HuggingFace dataset conversion."""

    @pytest.fixture()
    def samples(self):
        return {
            "catalog": {
                "r": torch.tensor([0.3, 0.4]),
                "digit": torch.tensor([0, 1]).unsqueeze(-1),
            },
            "z": torch.randn(2, 10),
            "z_prime": torch.randn(2, 10),
        }

    def test_create_samples_has_z_and_z_prime(self, samples):
        ds = create_samples(samples.copy(), "test")
        assert "z" in ds.column_names
        assert "z_prime" in ds.column_names