"""Tests for MangoMAS weight export pipeline."""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from forge.mangomas.export import ExportManifest, WeightExporter


class TestWeightExporter:
    """Tests for WeightExporter."""

    def test_export_bdi_weights(self, tmp_path: Any) -> None:
        exporter = WeightExporter(tmp_path / "export", platform="drone")
        weights = {"gru_w": np.zeros((64, 18)), "mlp_w": np.ones((8, 64))}
        path = exporter.export_bdi_weights(weights)
        assert path.exists()
        loaded = np.load(str(path))
        assert "gru_w" in loaded

    def test_export_constitutional_weights(self, tmp_path: Any) -> None:
        exporter = WeightExporter(tmp_path / "export")
        weights = {"policy_w": np.zeros((10, 18)), "value_w": np.zeros((1, 18))}
        path = exporter.export_constitutional_weights(weights)
        assert path.exists()

    def test_export_rssm_weights(self, tmp_path: Any) -> None:
        exporter = WeightExporter(tmp_path / "export")
        weights = {
            "gru_w_ih": np.zeros((192, 93)),
            "gru_w_hh": np.zeros((192, 64)),
            "prior_mean_w": np.zeros((16, 64)),
        }
        path = exporter.export_rssm_weights(weights)
        assert path.exists()

    def test_export_mcts_config(self, tmp_path: Any) -> None:
        exporter = WeightExporter(tmp_path / "export")
        config = {"c_puct": 1.5, "num_simulations": 200, "max_depth": 50}
        path = exporter.export_mcts_config(config)
        assert path.exists()
        with path.open() as f:
            data = json.load(f)
        assert data["c_puct"] == 1.5

    def test_export_curiosity_weights(self, tmp_path: Any) -> None:
        exporter = WeightExporter(tmp_path / "export")
        weights = {"social": 0.4, "epistemic": 0.3, "perceptual": 0.2, "metacognitive": 0.1}
        path = exporter.export_curiosity_weights(weights)
        assert path.exists()

    def test_export_curriculum_state(self, tmp_path: Any) -> None:
        exporter = WeightExporter(tmp_path / "export")
        state = {"current_tier": 3, "max_unlocked_tier": 3, "total_episodes": 500}
        path = exporter.export_curriculum_state(state)
        assert path.exists()

    def test_finalize_manifest(self, tmp_path: Any) -> None:
        exporter = WeightExporter(tmp_path / "export", platform="car")
        exporter.export_mcts_config({"c_puct": 1.0})
        exporter.export_curiosity_weights({"social": 0.5})
        result = exporter.finalize()

        manifest_path = result / "manifest.json"
        assert manifest_path.exists()
        with manifest_path.open() as f:
            manifest = json.load(f)
        assert manifest["platform"] == "car"
        assert "mcts" in manifest["components"]
        assert "curiosity" in manifest["components"]

    def test_full_export_pipeline(self, tmp_path: Any) -> None:
        """End-to-end: export all components and verify manifest."""
        exporter = WeightExporter(tmp_path / "full_export", platform="drone")

        exporter.export_bdi_weights({"w": np.zeros(10)})
        exporter.export_constitutional_weights({"w": np.zeros(10)})
        exporter.export_rssm_weights({"w": np.zeros(10)})
        exporter.export_mcts_config({"c_puct": 2.0})
        exporter.export_curiosity_weights({"social": 0.4})
        exporter.export_curriculum_state({"tier": 1})

        out_dir = exporter.finalize()

        manifest_path = out_dir / "manifest.json"
        with manifest_path.open() as f:
            manifest = json.load(f)

        assert len(manifest["components"]) == 6
        assert set(manifest["components"]) == {
            "bdi", "constitutional", "rssm", "mcts", "curiosity", "curriculum"
        }


class TestExportManifest:
    """Tests for ExportManifest dataclass."""

    def test_defaults(self) -> None:
        m = ExportManifest()
        assert m.version == "1.0"
        assert m.platform == "drone"
        assert m.components == []
