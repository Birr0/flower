"""Tests for flower.utilities."""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from flower.utilities import data_root, find_project_root


class TestDataRoot:
    """`DATA_ROOT` must be resolved when asked for, not when a module is imported.

    Reading it at module scope (the old behaviour, #37) meant importing
    `flower.data.sdss` raised before any user code ran. Resolution is now
    deferred to call time so import stays side-effect free.
    """

    def test_returns_the_value_when_set(self, tmp_path: Path):
        with mock.patch.dict(os.environ, {"DATA_ROOT": str(tmp_path)}, clear=False):
            assert data_root() == str(tmp_path)

    def test_raises_a_named_error_when_unset(self, monkeypatch: pytest.MonkeyPatch):
        """Unset must be a clear failure, not `None` propagated into a path.

        `dsprites.py` previously built `"None/dsprites-dataset"` and failed later
        with a confusing FileNotFoundError; `sdss.py` raised TypeError at import.
        """
        monkeypatch.delenv("DATA_ROOT", raising=False)
        monkeypatch.setattr("flower.utilities.load_dotenv", lambda *a, **k: None)

        with pytest.raises(RuntimeError, match="DATA_ROOT"):
            data_root()

    def test_trailing_separator_is_stripped(self, monkeypatch: pytest.MonkeyPatch):
        """So callers can join without producing a doubled separator."""
        monkeypatch.setenv("DATA_ROOT", "/some/root/")
        assert data_root() == "/some/root"


class TestFindProjectRoot:
    def test_finds_pyproject_in_self(self, tmp_path: Path):
        """When starting at the repo root, should find pyproject.toml."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[project]\nname = 'test'\n")

        result = find_project_root(str(tmp_path))
        assert Path(result) == tmp_path

    def test_finds_pyproject_in_subdir(self, tmp_path: Path):
        """When starting in a subdirectory, should traverse up."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[project]\nname = 'test'\n")
        subdir = tmp_path / "a" / "b" / "c"
        subdir.mkdir(parents=True)

        result = find_project_root(str(subdir))
        assert Path(result) == tmp_path

    def test_raises_when_not_found(self, tmp_path: Path):
        """Should raise FileNotFoundError when no pyproject.toml exists."""
        # Create a deep subdirectory with no pyproject.toml above it
        deep = tmp_path / "x" / "y" / "z"
        deep.mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match=r"pyproject.toml"):
            find_project_root(str(deep))
