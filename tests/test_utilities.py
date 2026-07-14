"""Tests for flower.utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from flower.utilities import find_project_root


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
