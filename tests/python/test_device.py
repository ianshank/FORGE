"""Tests for forge.utils.device — compute device detection."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from forge.utils.device import ensure_device_available, get_device, is_gpu_available


class TestGetDevice:
    """Tests for get_device().

    ``get_device`` performs ``import torch`` inside the function body, so
    patching ``sys.modules["torch"]`` controls which object is returned
    without needing to reload the module.
    """

    def test_returns_cuda_when_cuda_available(self) -> None:
        """get_device() returns 'cuda' when torch.cuda.is_available() is True."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = get_device()
        assert result == "cuda"

    def test_returns_mps_when_mps_available(self) -> None:
        """get_device() returns 'mps' when CUDA unavailable but MPS available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = get_device()
        assert result == "mps"

    def test_returns_cpu_when_no_accelerator(self) -> None:
        """get_device() returns 'cpu' when no GPU accelerator is available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = get_device()
        assert result == "cpu"

    def test_returns_cpu_when_torch_not_installed(self) -> None:
        """get_device() returns 'cpu' when torch raises ImportError."""
        # Setting sys.modules["torch"] = None causes `import torch` to raise ImportError
        with patch.dict(sys.modules, {"torch": None}):
            result = get_device()
        assert result == "cpu"


class TestIsGpuAvailable:
    """Tests for is_gpu_available()."""

    def test_true_when_cuda(self) -> None:
        """is_gpu_available() is True when get_device returns 'cuda'."""
        with patch("forge.utils.device.get_device", return_value="cuda"):
            assert is_gpu_available() is True

    def test_true_when_mps(self) -> None:
        """is_gpu_available() is True when get_device returns 'mps'."""
        with patch("forge.utils.device.get_device", return_value="mps"):
            assert is_gpu_available() is True

    def test_false_when_cpu(self) -> None:
        """is_gpu_available() is False when get_device returns 'cpu'."""
        with patch("forge.utils.device.get_device", return_value="cpu"):
            assert is_gpu_available() is False


class TestEnsureDeviceAvailable:
    """Tests for ensure_device_available() — the train.py fail-fast gate."""

    @staticmethod
    def _torch(*, cuda: bool, count: int = 1, mps: bool = False, cuda_build: str | None = "12.1"):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = cuda
        mock_torch.cuda.device_count.return_value = count
        mock_torch.backends.mps.is_available.return_value = mps
        mock_torch.version.cuda = cuda_build
        return mock_torch

    def test_auto_and_cpu_pass_without_torch(self) -> None:
        """auto/cpu never touch torch, so they work on torch-less installs."""
        with patch.dict(sys.modules, {"torch": None}):
            assert ensure_device_available("auto") == "auto"
            assert ensure_device_available("cpu") == "cpu"

    def test_cuda_passes_when_available(self) -> None:
        with patch.dict(sys.modules, {"torch": self._torch(cuda=True, count=2)}):
            assert ensure_device_available("cuda") == "cuda"
            assert ensure_device_available("cuda:1") == "cuda:1"

    def test_cuda_unavailable_on_cpu_only_build_names_the_wheel(self) -> None:
        with (
            patch.dict(sys.modules, {"torch": self._torch(cuda=False, cuda_build=None)}),
            pytest.raises(ValueError, match="CPU-only PyTorch build"),
        ):
            ensure_device_available("cuda")

    def test_cuda_unavailable_on_cuda_build_points_at_driver(self) -> None:
        with (
            patch.dict(sys.modules, {"torch": self._torch(cuda=False)}),
            pytest.raises(ValueError, match="sees no GPU"),
        ):
            ensure_device_available("cuda")

    def test_cuda_index_out_of_range(self) -> None:
        with (
            patch.dict(sys.modules, {"torch": self._torch(cuda=True, count=1)}),
            pytest.raises(ValueError, match="only 1 CUDA device"),
        ):
            ensure_device_available("cuda:1")

    def test_mps_unavailable(self) -> None:
        with (
            patch.dict(sys.modules, {"torch": self._torch(cuda=False, mps=False)}),
            pytest.raises(ValueError, match="MPS backend"),
        ):
            ensure_device_available("mps")

    @pytest.mark.parametrize("device", ["gpu", "cuda:", "cuda:x", "mps:0", "cpu:0", "auto:1", ""])
    def test_unknown_device_strings_rejected(self, device: str) -> None:
        with pytest.raises(ValueError, match="Unknown device"):
            ensure_device_available(device)

    def test_missing_torch_for_accelerator(self) -> None:
        with (
            patch.dict(sys.modules, {"torch": None}),
            pytest.raises(ValueError, match="requires PyTorch"),
        ):
            ensure_device_available("cuda")
