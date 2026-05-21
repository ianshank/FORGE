"""Tests for forge.utils.weight_loader module."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from forge.utils.weight_loader import (
    _HF_MISSING_MSG,
    _TORCH_MISSING_MSG,
    DEFAULT_REPO_ID,
    DEFAULT_REVISION,
    WeightLoader,
    WeightLoaderConfig,
)

# ---------------------------------------------------------------------------
# WeightLoaderConfig
# ---------------------------------------------------------------------------


class TestWeightLoaderConfig:
    """Tests for WeightLoaderConfig defaults and customisation."""

    def test_defaults(self) -> None:
        """Default config should use mousedroid repo and main revision."""
        cfg = WeightLoaderConfig()
        assert cfg.repo_id == DEFAULT_REPO_ID
        assert cfg.revision == DEFAULT_REVISION
        assert cfg.cache_dir is None
        assert cfg.force_download is False

    def test_custom_values(self) -> None:
        """All fields should accept custom values."""
        cfg = WeightLoaderConfig(
            repo_id="org/model",
            revision="v1.0",
            cache_dir="/tmp/cache",
            force_download=True,
        )
        assert cfg.repo_id == "org/model"
        assert cfg.revision == "v1.0"
        assert cfg.cache_dir == "/tmp/cache"
        assert cfg.force_download is True

    def test_is_dataclass(self) -> None:
        """WeightLoaderConfig should be a proper dataclass."""
        cfg = WeightLoaderConfig()
        field_names = {f.name for f in fields(cfg)}
        assert field_names == {"repo_id", "revision", "cache_dir", "force_download"}


# ---------------------------------------------------------------------------
# WeightLoader construction
# ---------------------------------------------------------------------------


class TestWeightLoaderInit:
    """Tests for WeightLoader initialisation."""

    def test_default_config(self) -> None:
        """WeightLoader with no args should use default config."""
        loader = WeightLoader()
        assert loader.config.repo_id == DEFAULT_REPO_ID

    def test_custom_config(self) -> None:
        """WeightLoader should accept a custom config."""
        cfg = WeightLoaderConfig(repo_id="custom/repo")
        loader = WeightLoader(cfg)
        assert loader.config.repo_id == "custom/repo"

    def test_config_property(self) -> None:
        """config property should return the stored configuration."""
        cfg = WeightLoaderConfig(revision="dev")
        loader = WeightLoader(cfg)
        assert loader.config is cfg


# ---------------------------------------------------------------------------
# resolve_path
# ---------------------------------------------------------------------------


class TestWeightLoaderResolvePath:
    """Tests for WeightLoader.resolve_path()."""

    @patch("forge.utils.weight_loader._require_huggingface_hub")
    def test_resolve_path_calls_hf_hub_download(self, mock_require: MagicMock) -> None:
        """resolve_path should call hf_hub_download with correct args."""
        mock_hf = MagicMock()
        mock_hf.hf_hub_download.return_value = "/cache/file.npz"
        mock_require.return_value = mock_hf

        cfg = WeightLoaderConfig(
            repo_id="user/repo", revision="v2", cache_dir="/my/cache", force_download=True
        )
        loader = WeightLoader(cfg)
        result = loader.resolve_path("weights/model.npz")

        mock_hf.hf_hub_download.assert_called_once_with(
            repo_id="user/repo",
            filename="weights/model.npz",
            revision="v2",
            cache_dir="/my/cache",
            force_download=True,
        )
        assert result == Path("/cache/file.npz")

    @patch("forge.utils.weight_loader._require_huggingface_hub")
    def test_resolve_path_returns_path_object(self, mock_require: MagicMock) -> None:
        """resolve_path should return a Path, not a string."""
        mock_hf = MagicMock()
        mock_hf.hf_hub_download.return_value = "/some/path/file.pt"
        mock_require.return_value = mock_hf

        loader = WeightLoader()
        result = loader.resolve_path("file.pt")
        assert isinstance(result, Path)


# ---------------------------------------------------------------------------
# load_npz
# ---------------------------------------------------------------------------


class TestWeightLoaderLoadNpz:
    """Tests for WeightLoader.load_npz()."""

    def test_load_npz_returns_dict(self, tmp_path: Path) -> None:
        """load_npz should return a dict of numpy arrays."""
        # Create a real .npz file
        arr_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        arr_b = np.array([[4.0, 5.0]], dtype=np.float32)
        npz_path = tmp_path / "test.npz"
        np.savez(str(npz_path), weight_a=arr_a, weight_b=arr_b)

        with patch.object(WeightLoader, "resolve_path", return_value=npz_path):
            loader = WeightLoader()
            data = loader.load_npz("test.npz")

        assert isinstance(data, dict)
        assert set(data.keys()) == {"weight_a", "weight_b"}
        np.testing.assert_array_equal(data["weight_a"], arr_a)
        np.testing.assert_array_equal(data["weight_b"], arr_b)

    def test_load_npz_preserves_dtype(self, tmp_path: Path) -> None:
        """load_npz should preserve the original array dtypes."""
        arr = np.array([1, 2, 3], dtype=np.int16)
        npz_path = tmp_path / "int16.npz"
        np.savez(str(npz_path), data=arr)

        with patch.object(WeightLoader, "resolve_path", return_value=npz_path):
            loader = WeightLoader()
            data = loader.load_npz("int16.npz")

        assert data["data"].dtype == np.int16


# ---------------------------------------------------------------------------
# load_pt
# ---------------------------------------------------------------------------


class TestWeightLoaderLoadPt:
    """Tests for WeightLoader.load_pt()."""

    def test_load_pt_returns_checkpoint(self, tmp_path: Path) -> None:
        """load_pt should return the deserialized checkpoint dict."""
        torch = pytest.importorskip("torch")

        checkpoint = {"weight": torch.tensor([1.0, 2.0]), "epoch": 10}
        pt_path = tmp_path / "model.pt"
        torch.save(checkpoint, str(pt_path))

        with patch.object(WeightLoader, "resolve_path", return_value=pt_path):
            loader = WeightLoader()
            loaded = loader.load_pt("model.pt", device="cpu")

        assert torch.equal(loaded["weight"], checkpoint["weight"])
        assert loaded["epoch"] == 10

    def test_load_pt_raises_without_torch(self) -> None:
        """load_pt should raise ImportError with a helpful message when torch is missing."""
        with (
            patch.object(WeightLoader, "resolve_path", return_value=Path("/fake.pt")),
            patch.dict("sys.modules", {"torch": None}),
        ):
            loader = WeightLoader()
            with pytest.raises(ImportError, match="torch is required"):
                loader.load_pt("fake.pt")


# ---------------------------------------------------------------------------
# list_files
# ---------------------------------------------------------------------------


class TestWeightLoaderListFiles:
    """Tests for WeightLoader.list_files()."""

    @patch("forge.utils.weight_loader._require_huggingface_hub")
    def test_list_files_returns_sorted_filenames(self, mock_require: MagicMock) -> None:
        """list_files should return sorted filenames from model_info."""
        sibling_b = SimpleNamespace(rfilename="bdi/belief.npz")
        sibling_a = SimpleNamespace(rfilename="README.md")
        sibling_c = SimpleNamespace(rfilename="rssm/final.pt")

        mock_hf = MagicMock()
        mock_hf.model_info.return_value = SimpleNamespace(
            siblings=[sibling_b, sibling_a, sibling_c]
        )
        mock_require.return_value = mock_hf

        loader = WeightLoader()
        files = loader.list_files()

        assert files == ["README.md", "bdi/belief.npz", "rssm/final.pt"]

    @patch("forge.utils.weight_loader._require_huggingface_hub")
    def test_list_files_empty_repo(self, mock_require: MagicMock) -> None:
        """list_files should handle repos with no siblings gracefully."""
        mock_hf = MagicMock()
        mock_hf.model_info.return_value = SimpleNamespace(siblings=None)
        mock_require.return_value = mock_hf

        loader = WeightLoader()
        files = loader.list_files()
        assert files == []


# ---------------------------------------------------------------------------
# Missing dependency handling
# ---------------------------------------------------------------------------


class TestWeightLoaderMissingDeps:
    """Tests for error handling when optional dependencies are missing."""

    def test_resolve_path_raises_without_huggingface_hub(self) -> None:
        """resolve_path should raise ImportError when huggingface_hub is missing."""
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            loader = WeightLoader()
            with pytest.raises(ImportError, match="huggingface_hub is required"):
                loader.resolve_path("file.npz")

    def test_list_files_raises_without_huggingface_hub(self) -> None:
        """list_files should raise ImportError when huggingface_hub is missing."""
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            loader = WeightLoader()
            with pytest.raises(ImportError, match="huggingface_hub is required"):
                loader.list_files()

    def test_error_messages_are_informative(self) -> None:
        """Error messages should include install instructions."""
        assert "pip install" in _HF_MISSING_MSG
        assert "pip install" in _TORCH_MISSING_MSG
