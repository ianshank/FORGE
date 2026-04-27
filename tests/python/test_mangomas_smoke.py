"""MangoMAS end-to-end smoke tests.

Verifies TOML config → episode loop wiring and export pipeline integrity
using real config files from configs/mangomas/.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from forge.mangomas.adapters import ActionSpaceAdapter, ObservationAdapter
from forge.mangomas.batch import BatchCollector, EpisodeData
from forge.mangomas.bdi_trainer import BDIPreTrainer
from forge.mangomas.config import MangoMASBridgeConfig
from forge.mangomas.constitutional_trainer import ConstitutionalPreTrainer
from forge.mangomas.curiosity_optimizer import CuriosityWeightOptimizer
from forge.mangomas.curriculum_controller import PlatformCurriculumController
from forge.mangomas.export import WeightExporter
from forge.mangomas.pipeline import CollectedTrainingData, MangoMASDroneTrainingPipeline
from forge.mangomas.rssm_pretrainer import RSSMPreTrainer

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs" / "mangomas"
assert CONFIGS_DIR.exists(), f"MangoMAS config directory not found: {CONFIGS_DIR}"

# Discover all TOML config files dynamically
_TOML_FILES = sorted(CONFIGS_DIR.glob("*.toml"))
assert len(_TOML_FILES) > 0, f"No TOML files found in {CONFIGS_DIR}"


# ---------------------------------------------------------------------------
# E1: TOML → Config Loading Smoke Tests
# ---------------------------------------------------------------------------


class TestTomlConfigLoading:
    """Each configs/mangomas/*.toml loads without error."""

    @pytest.mark.parametrize(
        "toml_path",
        _TOML_FILES,
        ids=[p.stem for p in _TOML_FILES],
    )
    def test_toml_loads_without_error(self, toml_path: Path) -> None:
        config = MangoMASBridgeConfig.from_toml(toml_path)
        assert config.platform in ("drone", "car", "Drone", "Car")
        assert config.action_adapter.bins_per_axis >= 1

    def test_default_toml_has_all_sections(self) -> None:
        default_path = CONFIGS_DIR / "default.toml"
        if not default_path.exists():
            pytest.skip("default.toml not found")
        config = MangoMASBridgeConfig.from_toml(default_path)
        # Verify nested configs were populated (not just defaults)
        assert config.action_adapter is not None
        assert config.observation_adapter is not None
        assert config.sweep is not None
        assert config.curriculum is not None
        assert config.transfer is not None

    def test_car_vs_drone_curriculum_differ(self) -> None:
        car_path = CONFIGS_DIR / "car_curriculum.toml"
        drone_path = CONFIGS_DIR / "drone_curriculum.toml"
        if not car_path.exists() or not drone_path.exists():
            pytest.skip("car_curriculum.toml or drone_curriculum.toml not found")
        car_cfg = MangoMASBridgeConfig.from_toml(car_path)
        drone_cfg = MangoMASBridgeConfig.from_toml(drone_path)
        # Different platforms should produce different curriculum tiers
        assert car_cfg.curriculum.tiers != drone_cfg.curriculum.tiers


# ---------------------------------------------------------------------------
# E1: Config → Episode Loop Wiring
# ---------------------------------------------------------------------------


def _make_collected_data(state_dim: int = 22) -> CollectedTrainingData:
    """Create minimal synthetic training data for pipeline tests."""
    num_episodes = 3
    ep_len = 10
    observations = [
        np.random.default_rng(i).standard_normal((ep_len + 1, state_dim)).astype(np.float32)
        for i in range(num_episodes)
    ]
    action_names = [
        ["Move", "Ascend", "Hover", "Scan", "Noop"] * 2 for _ in range(num_episodes)
    ]
    action_ids = [
        np.random.default_rng(i).integers(0, 50, size=ep_len).astype(np.int64)
        for i in range(num_episodes)
    ]
    rewards = [
        np.random.default_rng(i).uniform(0, 1, size=ep_len).astype(np.float32)
        for i in range(num_episodes)
    ]
    dones = [np.zeros(ep_len, dtype=np.float32) for _ in range(num_episodes)]
    for d in dones:
        d[-1] = 1.0
    raw_observations: list[list[dict[str, Any]]] = [
        [
            {
                "battery": 0.8 - step * 0.01,
                "altitude": 0.3 + step * 0.02,
                "stamina_inverse": 0.2,
                "boundary_distance": 0.5,
                "threat_proximity": 0.9,
            }
            for step in range(ep_len + 1)
        ]
        for _ in range(num_episodes)
    ]
    return CollectedTrainingData(
        observations=observations,
        action_names=action_names,
        action_ids=action_ids,
        rewards=rewards,
        dones=dones,
        raw_observations=raw_observations,
    )


class TestConfigToEpisodeWiring:
    """Config → adapter → episode data flow."""

    def test_action_adapter_from_config(self) -> None:
        config = MangoMASBridgeConfig()
        adapter = ActionSpaceAdapter(
            config=config.action_adapter,
            platform=config.platform,
        )
        # Drone: 4D continuous → discrete
        discrete_id = adapter.continuous_to_discrete(np.array([0.0, 0.0, 0.0, 0.0]))
        continuous = adapter.discrete_to_continuous(discrete_id)
        assert len(continuous) == 4
        assert all(-1.0 <= c <= 1.0 for c in continuous)

    def test_observation_adapter_from_config(self) -> None:
        config = MangoMASBridgeConfig()
        adapter = ObservationAdapter(
            config=config.observation_adapter,
            platform=config.platform,
        )
        # Build a minimal FORGE observation dict
        obs = {
            "grid_view": np.zeros((5, 5, 11), dtype=np.float32),
            "position": (8, 12),
            "health": 0.9,
            "stamina": 0.7,
            "heading": 2,
            "inventory": [(0, 0)],
            "altitude": 0.5,
            "battery": 0.8,
            "morphology": 0,
            "day_phase": 1,
        }
        flat = adapter.adapt(obs)
        assert isinstance(flat, np.ndarray)
        assert flat.ndim == 1

    def test_batch_collector_episode_shape(self) -> None:
        """BatchCollector produces episodes with correct array shapes."""
        state_dim = 22
        collector = BatchCollector()
        result = collector.collect(
            num_episodes=1,
            step_fn=lambda obs, act: (np.zeros(state_dim, dtype=np.float32), 0.1, act > 40),
            reset_fn=lambda ep: np.zeros(state_dim, dtype=np.float32),
            policy_fn=lambda obs: 0,
        )
        assert result is not None
        assert len(result.episodes) >= 1
        episode = result.episodes[0]
        assert isinstance(episode, EpisodeData)
        assert episode.observations.shape[1] == state_dim
        assert len(episode.actions) == len(episode.rewards)


# ---------------------------------------------------------------------------
# E1: Pipeline Stage Integration
# ---------------------------------------------------------------------------


class TestPipelineStageWiring:
    """Verify that pipeline stages can run end-to-end with synthetic data."""

    def test_bdi_stage_produces_dataset(self) -> None:
        config = MangoMASBridgeConfig()
        trainer = BDIPreTrainer(config=config.bdi_trainer)
        data = _make_collected_data()
        dataset = trainer.build_dataset(
            data.observations, data.action_names, data.rewards
        )
        assert dataset.num_samples > 0
        dist = dataset.intention_distribution()
        assert len(dist) == config.bdi_trainer.num_intentions

    def test_constitutional_stage_detects_violations(self) -> None:
        config = MangoMASBridgeConfig()
        trainer = ConstitutionalPreTrainer(config=config.constitutional_trainer)
        # Observation with battery below threshold → should violate
        obs = {
            "battery": 0.05,  # below 0.2 threshold
            "altitude": 0.3,
            "stamina_inverse": 0.2,
            "boundary_distance": 0.5,
            "threat_proximity": 0.9,
        }
        violations = trainer.check_violations(obs)
        assert any(v.constraint_name == "battery_minimum" for v in violations)

    def test_rssm_stage_builds_sequences(self) -> None:
        config = MangoMASBridgeConfig()
        # Use short sequence length to match our synthetic 10-step episodes
        config.rssm_pretrain.sequence_length = 5
        trainer = RSSMPreTrainer(config=config.rssm_pretrain)
        data = _make_collected_data()
        dataset = trainer.build_sequences(
            data.observations, data.action_ids, data.rewards, data.dones,
        )
        assert dataset.num_sequences > 0
        # Sequences should have correct length
        assert dataset.sequence_length == 5

    def test_curiosity_weights_normalize(self) -> None:
        config = MangoMASBridgeConfig()
        optimizer = CuriosityWeightOptimizer(
            channels=list(config.curiosity_optimizer.channels),
            initial_weights=list(config.curiosity_optimizer.initial_weights),
            seed=config.curiosity_optimizer.seed,
        )
        weights = optimizer._normalize(
            np.array([0.5, 0.3, 0.1, 0.1])
        )
        assert abs(weights.sum() - 1.0) < 1e-6
        assert all(w >= 0 for w in weights)

    def test_curriculum_controller_from_config(self) -> None:
        config = MangoMASBridgeConfig()
        controller = PlatformCurriculumController(
            platform=config.platform,
            config=config.curriculum,
        )
        # Should start at tier 1
        assert controller.current_tier == 1
        tier_info = controller.tier_info(tier=1)
        assert isinstance(tier_info, dict)
        assert len(tier_info) > 0


# ---------------------------------------------------------------------------
# E1: Export Pipeline Integrity
# ---------------------------------------------------------------------------


class TestExportPipelineIntegrity:
    """Verify export creates valid bundles from pipeline outputs."""

    def test_full_export_roundtrip(self, tmp_path: Any) -> None:
        exporter = WeightExporter(tmp_path / "export", platform="drone")

        # Export all component types
        bdi_w = {"gru_w": np.zeros((64, 22)), "mlp_w": np.ones((8, 64))}
        exporter.export_bdi_weights(bdi_w)

        const_w = {"policy_w": np.zeros((50, 22)), "value_w": np.zeros((1, 22))}
        exporter.export_constitutional_weights(const_w)

        rssm_w = {"gru_ih": np.zeros((192, 72)), "gru_hh": np.zeros((192, 64))}
        exporter.export_rssm_weights(rssm_w)

        exporter.export_mcts_config({"c_puct": 1.5, "simulations": 200})
        exporter.export_curiosity_weights({"social": 0.4, "epistemic": 0.3})
        exporter.export_curriculum_state({"current_tier": 3, "max_unlocked": 3})

        export_dir = exporter.finalize()

        # Read and validate manifest
        manifest_path = export_dir / "manifest.json"
        assert manifest_path.exists()
        with manifest_path.open() as f:
            manifest = json.load(f)
        assert manifest["platform"] == "drone"
        assert set(manifest["components"]) == {
            "bdi", "constitutional", "rssm", "mcts", "curiosity", "curriculum",
        }
        # Verify all weight files exist
        assert (export_dir / "bdi_weights.npz").exists()
        assert (export_dir / "constitutional_weights.npz").exists()
        assert (export_dir / "rssm_weights.npz").exists()
        assert (export_dir / "mcts_config.json").exists()
        assert (export_dir / "curiosity_weights.json").exists()
        assert (export_dir / "curriculum_state.json").exists()

    def test_export_weights_loadable(self, tmp_path: Any) -> None:
        """Exported .npz files can be loaded back with correct keys."""
        exporter = WeightExporter(tmp_path / "export")
        weights = {"layer1": np.random.default_rng(42).standard_normal((10, 5)).astype(np.float32)}
        path = exporter.export_bdi_weights(weights)

        loaded = dict(np.load(str(path)))
        assert "layer1" in loaded
        np.testing.assert_array_almost_equal(loaded["layer1"], weights["layer1"])


# ---------------------------------------------------------------------------
# E2: Pipeline End-to-End Smoke
# ---------------------------------------------------------------------------


class TestPipelineEndToEnd:
    """Full pipeline runs without error on synthetic data."""

    def test_pipeline_runs_bdi_stage(self, tmp_path: Any) -> None:
        config = MangoMASBridgeConfig()
        config.pipeline.paths.output_root = str(tmp_path)
        config.pipeline.paths.run_name = "smoke_test"
        config.pipeline.execution.stop_after_stage = "bdi"

        pipeline = MangoMASDroneTrainingPipeline(config=config)
        data = _make_collected_data()

        result = pipeline.run(data)
        assert result is not None
        bdi_stage = next(
            (s for s in result.stage_results if s.name == "bdi"), None
        )
        assert bdi_stage is not None
        assert bdi_stage.status == "completed"
