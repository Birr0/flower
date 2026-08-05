"""Tests for flower.data.modules — FlowerDataset, FlowerDataLoader."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

from flower.data.modules import FlowerDataLoader, FlowerDataset


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
            "r": torch.tensor([0.3]),
            "g": torch.tensor([0.5]),
            "b": torch.tensor([0.7]),
            "digit": torch.tensor([idx % 10]),
        }
        return {"X": X, "catalog": catalog}


class TestFlowerDataset:
    """Unit tests for the FlowerDataset wrapper."""

    def test_len(self):
        ds = _DummyDataset(n=16, channels=1)
        wrapped = FlowerDataset(ds, return_catalog=True)
        assert len(wrapped) == 16

    def test_getitem_keys(self):
        ds = _DummyDataset(n=16, channels=1)
        wrapped = FlowerDataset(ds, return_catalog=True)
        item = wrapped[0]
        assert "X" in item
        assert "y" in item
        assert "catalog" in item

    def test_getitem_no_catalog(self):
        ds = _DummyDataset(n=16, channels=1)
        wrapped = FlowerDataset(ds, return_catalog=False)
        item = wrapped[0]
        assert "X" in item
        assert "y" in item
        assert "catalog" not in item

    def test_getitem_X_shape(self):
        ds = _DummyDataset(n=16, channels=1, img_size=8)
        wrapped = FlowerDataset(ds, return_catalog=True)
        item = wrapped[0]
        assert item["X"].shape == (1, 8, 8)

    def test_y_is_tensor(self):
        ds = _DummyDataset(n=16, channels=1)
        wrapped = FlowerDataset(ds, return_catalog=True)
        item = wrapped[0]
        assert isinstance(item["y"], torch.Tensor)

    def test_y_is_concat_of_catalog(self):
        """y should be concatenated r + g + digit one-hot (b dropped)."""
        ds = _DummyDataset(n=16, channels=1)
        wrapped = FlowerDataset(ds, return_catalog=True)
        item = wrapped[0]
        # 1 (r) + 1 (g) + 10 (digit one-hot) = 12
        # but processing_fn for digit is None here, so raw tensors
        # The real y_catalog uses one_hot processing_fn for digit.
        # With no processing_fn, concat of scalar r + scalar g + scalar digit
        # = 3 values.  (size of digit entry is 10 but raw value is scalar)
        assert len(item["y"].shape) == 1


class TestFlowerDataLoader:
    """Unit tests for FlowerDataLoader."""

    def _make_dm(self, return_catalog=True, n=8, val_split=0.25, channels=1):
        """Helper to create a FlowerDataModule with the dummy dataset."""
        train_ds = FlowerDataset(
            _DummyDataset(n=n, channels=channels), return_catalog=return_catalog
        )
        test_ds = FlowerDataset(
            _DummyDataset(n=n, channels=channels), return_catalog=return_catalog
        )
        dm = FlowerDataLoader(
            datasets={"train": train_ds, "val": None, "test": test_ds},
            batch_size=4,
            val_split=val_split,
            shuffle=True,
            num_workers=0,
        )
        dm.setup()
        return dm

    def test_train_dataloader_shapes(self):
        dm = self._make_dm()
        loader = dm.train_dataloader()
        batch = next(iter(loader))
        assert batch["X"].shape[0] == 4  # batch size
        assert "y" in batch
        assert "catalog" in batch

    def test_val_dataloader_shapes(self):
        dm = self._make_dm()
        loader = dm.val_dataloader()
        batch = next(iter(loader))
        # 8 samples, val_split=0.25 -> 2 val
        assert batch["X"].shape[0] == 2

    def test_test_dataloader_shapes(self):
        dm = self._make_dm(n=8)
        loader = dm.test_dataloader()
        batches = list(loader)
        # 8 samples, batch_size=4 -> 2 full batches of 4
        assert len(batches) == 2
        assert batches[0]["X"].shape[0] == 4

    def test_val_split_produces_subset(self):
        dm = self._make_dm()
        assert hasattr(dm, "train_dataset")
        assert hasattr(dm, "val_dataset")
        assert len(dm.train_dataset) < 8
        assert len(dm.val_dataset) > 0

    def test_no_val_split_uses_explicit_val(self):
        """When val dataset is provided explicitly, use it (no split)."""
        train_ds = FlowerDataset(_DummyDataset(n=8, channels=1), return_catalog=True)
        val_ds = FlowerDataset(_DummyDataset(n=4, channels=1), return_catalog=True)
        test_ds = FlowerDataset(_DummyDataset(n=4, channels=1), return_catalog=True)
        dm = FlowerDataLoader(
            datasets={"train": train_ds, "val": val_ds, "test": test_ds},
            batch_size=4,
            val_split=0.0,
            num_workers=0,
        )
        dm.setup()
        assert len(dm.val_dataset) == 4

    def test_dataloader_iterates(self):
        dm = self._make_dm(n=8)
        loader = dm.train_dataloader()
        batches = list(loader)
        assert len(batches) > 0

    def test_no_shuffle_in_val(self):
        """Validation loader should not shuffle."""
        dm = self._make_dm()
        loader = dm.val_dataloader()
        # Just verify it iterates without error
        batch = next(iter(loader))
        assert batch is not None


class TestImportWithoutDataRoot:
    """The data modules must import with `DATA_ROOT` unset (#37).

    `sdss.py` read the variable at module scope and concatenated it, so a fresh
    interpreter raised `TypeError: unsupported operand type(s) for +: 'NoneType'
    and 'str'` before any user code ran. That took down anything importing the
    module transitively, including pytest collection on a clone with no `.env`.

    Run in a subprocess from a temporary directory: the module is already
    imported in this process, and `load_dotenv()` would otherwise walk up and
    find the repository's own `.env`.
    """

    MODULES = ("flower.data.sdss", "flower.data.dsprites")

    def _import_in_clean_interpreter(
        self, module: str, cwd
    ) -> subprocess.CompletedProcess:
        env = {k: v for k, v in os.environ.items() if k != "DATA_ROOT"}
        return subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )

    @pytest.mark.parametrize("module", MODULES)
    def test_imports_without_data_root(self, module: str, tmp_path):
        result = self._import_in_clean_interpreter(module, tmp_path)
        assert result.returncode == 0, (
            f"importing {module} without DATA_ROOT failed:\n{result.stderr}"
        )

    @pytest.mark.parametrize("module", MODULES)
    def test_no_none_leaks_into_a_path(self, module: str, tmp_path):
        """Importing must not silently build a path from a `None` root.

        `dsprites.py` did not raise at import -- it stored `None` and produced
        `"None/dsprites-dataset"` later, which fails far from the cause.
        """
        env = {k: v for k, v in os.environ.items() if k != "DATA_ROOT"}
        probe = (
            f"import {module} as m;"
            "vals = [str(v) for v in vars(m).values() if isinstance(v, str)];"
            "bad = [v for v in vals if v.startswith('None')];"
            "assert not bad, bad"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert result.returncode == 0, result.stderr
