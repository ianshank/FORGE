"""Tests for forge.utils.device — compute device detection."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from forge.utils.device import get_device, is_gpu_available


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
