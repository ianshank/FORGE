"""Tests for MuZero model export to ONNX and TorchScript."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from forge.models.muzero_config import MuZeroConfig  # noqa: E402
from forge.models.muzero_export import MuZeroExporter  # noqa: E402
from forge.models.muzero_world_model import MuZeroWorldModel  # noqa: E402

OBS_DIM = 11 * 11 * 7 + 73
ACTION_DIM = 5
LATENT_DIM = 16
HIDDEN_DIM = 16


def _make_model() -> MuZeroWorldModel:
    return MuZeroWorldModel(MuZeroConfig(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_blocks=1,
        reward_support_size=11,
        value_support_size=11,
        cnn_channels=(8,),
        cnn_kernel_sizes=(3,),
        cnn_strides=(1,),
    ))


# ---------------------------------------------------------------------------
# TorchScript export
# ---------------------------------------------------------------------------


class TestTorchScriptExport:
    def test_export_creates_files(self, tmp_path: Any) -> None:
        model = _make_model()
        exporter = MuZeroExporter(model)
        paths = exporter.export_torchscript(tmp_path / "ts")
        assert len(paths) == 3
        for p in paths:
            assert p.exists()
            assert p.stat().st_size > 0

    def test_torchscript_representation_loadable(self, tmp_path: Any) -> None:
        model = _make_model()
        exporter = MuZeroExporter(model)
        exporter.export_torchscript(tmp_path / "ts")

        loaded = torch.jit.load(str(tmp_path / "ts" / "representation.pt"))
        obs = torch.randn(1, OBS_DIM)
        result = loaded(obs)
        assert result.shape == (1, LATENT_DIM)

    def test_torchscript_prediction_loadable(self, tmp_path: Any) -> None:
        model = _make_model()
        exporter = MuZeroExporter(model)
        exporter.export_torchscript(tmp_path / "ts")

        loaded = torch.jit.load(str(tmp_path / "ts" / "prediction.pt"))
        latent = torch.randn(1, LATENT_DIM)
        policy, _value = loaded(latent)
        assert policy.shape == (1, ACTION_DIM)

    def test_torchscript_dynamics_loadable(self, tmp_path: Any) -> None:
        model = _make_model()
        exporter = MuZeroExporter(model)
        exporter.export_torchscript(tmp_path / "ts")

        loaded = torch.jit.load(str(tmp_path / "ts" / "dynamics.pt"))
        input_dim = LATENT_DIM + ACTION_DIM
        x = torch.randn(1, input_dim)
        next_latent, _reward = loaded(x)
        assert next_latent.shape == (1, LATENT_DIM)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _can_import(name: str) -> bool:
    try:
        __import__(name)
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# ONNX export (skip if onnxruntime not available)
# ---------------------------------------------------------------------------


class TestOnnxExport:
    @pytest.mark.skipif(
        not _can_import("onnxscript"),
        reason="onnxscript not installed (required by torch.onnx.export)",
    )
    def test_export_creates_files(self, tmp_path: Any) -> None:
        model = _make_model()
        exporter = MuZeroExporter(model)
        paths = exporter.export_onnx(tmp_path / "onnx")
        assert len(paths) == 3
        for p in paths:
            assert p.exists()
            assert p.stat().st_size > 0
        # Check filenames
        names = {p.name for p in paths}
        assert "representation.onnx" in names
        assert "dynamics.onnx" in names
        assert "prediction.onnx" in names

    @pytest.mark.skipif(
        not (_can_import("onnxruntime") and _can_import("onnxscript")),
        reason="onnxruntime or onnxscript not installed",
    )
    def test_validate_onnx(self, tmp_path: Any) -> None:
        model = _make_model()
        exporter = MuZeroExporter(model)
        exporter.export_onnx(tmp_path / "onnx")
        assert exporter.validate_export(tmp_path / "onnx", fmt="onnx")


# ---------------------------------------------------------------------------
# WeightExporter integration
# ---------------------------------------------------------------------------


class TestWeightExporterIntegration:
    def test_export_muzero_weights(self, tmp_path: Any) -> None:
        from forge.mangomas.export import WeightExporter  # noqa: PLC0415

        exporter = WeightExporter(tmp_path / "export", platform="drone")
        weights = {"rep_w0": np.zeros((32, 16)), "dyn_w0": np.ones((16, 16))}
        path = exporter.export_muzero_weights(weights)
        assert path.exists()
        loaded = np.load(str(path))
        assert "rep_w0" in loaded

    def test_manifest_includes_muzero(self, tmp_path: Any) -> None:
        import json  # noqa: PLC0415

        from forge.mangomas.export import WeightExporter  # noqa: PLC0415

        exporter = WeightExporter(tmp_path / "export")
        exporter.export_muzero_weights({"w": np.zeros(10)})
        exporter.export_mcts_config({"c_puct": 1.25})
        out_dir = exporter.finalize()

        with (out_dir / "manifest.json").open() as f:
            manifest = json.load(f)
        assert "muzero" in manifest["components"]
