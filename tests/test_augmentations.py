"""Tests for flower.data.augmentations — ImageAugmentation."""
from __future__ import annotations

import torch
import torch.distributions as d

from flower.data.augmentations import ImageAugmentation


class TestImageAugmentation:
    """Unit tests for ImageAugmentation."""

    def _make_affine(self, sampler=None, value=1.0):
        """Helper to build a minimal affine config dict."""
        return {
            "scale": {"sampler": sampler, "value": value},
            "angle": {"sampler": sampler, "value": 0.0},
            "translate": {"sampler": sampler, "value": [0, 0]},
            "shear": {"sampler": sampler, "value": [0, 0]},
        }

    def test_apply_transformations_returns_dict(self):
        """Affine config is truthy → transforms applied, returns dict with keys."""
        sampler = d.Uniform(low=0.8, high=1.2)
        affine = self._make_affine(sampler=sampler)
        aug = ImageAugmentation(
            affine=affine,
            image_size=(8, 8),
            random_colouring=False,
        )
        img = torch.ones(3, 16, 16)
        result = aug.apply_transformations(img)
        assert isinstance(result, dict)
        assert "img" in result
        assert "colour_factors" in result
        assert "zero_channel_idxs" in result
        assert "affine" in result

    def test_apply_transformations_output_shape(self):
        """Output image shape matches target image_size."""
        affine = self._make_affine(sampler=None, value=1.0)
        aug = ImageAugmentation(
            affine=affine,
            image_size=(8, 8),
            random_colouring=False,
        )
        img = torch.ones(3, 16, 16)
        result = aug.apply_transformations(img)
        assert result["img"].shape == (3, 8, 8)

    def test_apply_transformations_no_coloring(self):
        """When random_colouring is False, colour_factors is None."""
        affine = self._make_affine(sampler=None, value=1.0)
        aug = ImageAugmentation(
            affine=affine,
            image_size=(8, 8),
            random_colouring=False,
        )
        img = torch.ones(3, 8, 8)
        result = aug.apply_transformations(img)
        assert result["colour_factors"] is None

    def test_apply_transformations_with_coloring(self):
        """When random_colouring is a distribution, it generates factors."""
        sampler = d.Uniform(low=0.5, high=1.0)
        affine = self._make_affine(sampler=None, value=1.0)
        aug = ImageAugmentation(
            affine=affine,
            image_size=(8, 8),
            random_colouring=sampler,
        )
        # ImageAugmentation.random_colour repeats to 3 channels
        img = torch.ones(1, 8, 8)
        result = aug.apply_transformations(img)
        assert result["colour_factors"] is not None
        assert result["img"].shape == (3, 8, 8)  # Becomes 3 channels after colouring

    def test_apply_transformations_no_affine_uses_values(self):
        """When sampler is None, uses value field from config."""
        affine = self._make_affine(sampler=None, value=1.0)
        aug = ImageAugmentation(
            affine=affine,
            image_size=(8, 8),
            random_colouring=False,
        )
        img = torch.ones(3, 8, 8)
        result = aug.apply_transformations(img)
        assert result["img"].shape == (3, 8, 8)
        # Affine dict should be populated with value fields
        assert "affine" in result

    def test_resize_different_size(self):
        """Setting image_size to a different value resizes correctly."""
        affine = self._make_affine(sampler=None, value=1.0)
        aug = ImageAugmentation(
            affine=affine,
            image_size=(16, 16),
            random_colouring=False,
        )
        img = torch.ones(3, 8, 8)
        result = aug.apply_transformations(img)
        assert result["img"].shape == (3, 16, 16)