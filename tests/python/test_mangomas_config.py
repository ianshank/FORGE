"""Tests for MangoMAS configuration dataclasses."""
from __future__ import annotations

import pytest
from forge.mangomas.config import (
    ActionAdapterConfig,
    BatchCollectorConfig,
    BDITrainerConfig,
    ConstitutionalTrainerConfig,
    CuriosityOptimizerConfig,
    CurriculumConfig,
    MangoMASBridgeConfig,
    ObservationAdapterConfig,
    PipelineConfig,
    PipelineExecutionConfig,
    PipelinePathsConfig,
    RSSMPreTrainConfig,
    SweepConfig,
    TransferConfig,
)


class TestActionAdapterConfig:
    """Tests for ActionAdapterConfig defaults."""

    def test_defaults(self) -> None:
        c = ActionAdapterConfig()
        assert c.bins_per_axis == 7
        assert c.continuous_range_min == -1.0
        assert c.continuous_range_max == 1.0


class TestObservationAdapterConfig:
    """Tests for ObservationAdapterConfig defaults and new fields."""

    def test_defaults(self) -> None:
        c = ObservationAdapterConfig()
        assert c.grid_summary_dim == 64
        assert c.position_scale == 16.0
        assert c.grid_channels == 11
        assert c.scalar_fields == 5
        assert c.inventory_fields == 2
        assert c.drone_fields == 4

    def test_custom(self) -> None:
        c = ObservationAdapterConfig(position_scale=32.0, grid_channels=7)
        assert c.position_scale == 32.0
        assert c.grid_channels == 7


class TestSweepConfig:
    """Tests for SweepConfig."""

    def test_defaults(self) -> None:
        c = SweepConfig()
        assert c.c_puct_range == (0.5, 3.0)
        assert c.sim_budget_steps == 5
        assert c.seed == 42

    def test_custom_ranges(self) -> None:
        c = SweepConfig(c_puct_range=(1.0, 2.0), c_puct_steps=3)
        assert c.c_puct_range == (1.0, 2.0)
        assert c.c_puct_steps == 3


class TestBDITrainerConfig:
    """Tests for BDITrainerConfig with new fields."""

    def test_defaults(self) -> None:
        c = BDITrainerConfig()
        assert c.num_intentions == 8
        assert c.log_interval == 10
        assert c.default_intention == 7
        assert c.seed == 42


class TestConstitutionalTrainerConfig:
    """Tests for ConstitutionalTrainerConfig with constraint config."""

    def test_defaults(self) -> None:
        c = ConstitutionalTrainerConfig()
        assert c.num_constraints == 5
        assert c.value_loss_weight == 0.5
        assert c.log_interval == 20
        assert len(c.constraints) == 5

    def test_constraint_names(self) -> None:
        c = ConstitutionalTrainerConfig()
        names = {d["name"] for d in c.constraints}
        assert names == {
            "battery_minimum",
            "altitude_ceiling",
            "speed_ceiling",
            "geofence",
            "threat_exclusion",
        }

    def test_custom_constraints(self) -> None:
        custom = [{"name": "test", "forge_field": "x", "threshold": 0.5, "is_lower_bound": True}]
        c = ConstitutionalTrainerConfig(constraints=custom)
        assert len(c.constraints) == 1


class TestRSSMPreTrainConfig:
    """Tests for RSSMPreTrainConfig with new fields."""

    def test_defaults(self) -> None:
        c = RSSMPreTrainConfig()
        assert c.reward_loss_scale == 0.01
        assert c.log_interval == 20
        assert c.seed == 42


class TestTransferConfig:
    """Tests for MangoMAS transfer overrides."""

    def test_defaults(self) -> None:
        c = TransferConfig()
        assert c.bdi_mapping_overrides == {}


class TestPipelineConfig:
    """Tests for pipeline path and execution defaults."""

    def test_paths_defaults(self) -> None:
        c = PipelinePathsConfig()
        assert c.output_root == "artifacts/mangomas"
        assert c.export_dir_name == "export"
        assert c.manifest_name == "pipeline_manifest.json"

    def test_execution_defaults(self) -> None:
        c = PipelineExecutionConfig()
        assert c.resume is False
        assert c.fail_fast is True
        assert c.stop_after_stage == ""

    def test_top_level_defaults(self) -> None:
        c = PipelineConfig()
        assert isinstance(c.paths, PipelinePathsConfig)
        assert isinstance(c.execution, PipelineExecutionConfig)


class TestCurriculumConfig:
    """Tests for CurriculumConfig with new fields."""

    def test_defaults(self) -> None:
        c = CurriculumConfig()
        assert c.tiers == []
        assert c.seed == 42
        assert c.num_tiers == 5

    def test_custom_tiers(self) -> None:
        tiers = [{"tier": 1, "name": "Test", "forge_scenario": "patrol", "success_threshold": 0.5}]
        c = CurriculumConfig(tiers=tiers)
        assert len(c.tiers) == 1


class TestBatchCollectorConfig:
    """Tests for BatchCollectorConfig with new fields."""

    def test_defaults(self) -> None:
        c = BatchCollectorConfig()
        assert c.action_space_size == 75
        assert c.log_interval == 100


class TestCuriosityOptimizerConfig:
    """Tests for CuriosityOptimizerConfig."""

    def test_defaults(self) -> None:
        c = CuriosityOptimizerConfig()
        assert len(c.channels) == 4
        assert len(c.initial_weights) == 4
        assert c.population_size == 20
        assert c.sigma == pytest.approx(0.1)
        assert c.log_interval == 10

    def test_custom(self) -> None:
        c = CuriosityOptimizerConfig(channels=["a", "b"], initial_weights=[0.5, 0.5])
        assert c.channels == ["a", "b"]


class TestMangoMASBridgeConfig:
    """Tests for top-level bridge config."""

    def test_defaults(self) -> None:
        c = MangoMASBridgeConfig()
        assert c.platform == "drone"
        assert isinstance(c.action_adapter, ActionAdapterConfig)
        assert isinstance(c.curiosity_optimizer, CuriosityOptimizerConfig)

    def test_all_sub_configs_present(self) -> None:
        c = MangoMASBridgeConfig()
        assert c.action_adapter is not None
        assert c.observation_adapter is not None
        assert c.sweep is not None
        assert c.surprise_validator is not None
        assert c.bdi_trainer is not None
        assert c.constitutional_trainer is not None
        assert c.rssm_pretrain is not None
        assert c.curriculum is not None
        assert c.batch_collector is not None
        assert c.curiosity_optimizer is not None
        assert c.transfer is not None
        assert c.pipeline is not None


class TestMangoMASBridgeConfigToml:
    """Tests for TOML/dict config loading."""

    def test_from_dict_empty(self) -> None:
        config = MangoMASBridgeConfig._from_dict({})
        assert config.platform == "drone"

    def test_from_dict_platform_only(self) -> None:
        config = MangoMASBridgeConfig._from_dict({"platform": "car"})
        assert config.platform == "car"

    def test_from_dict_with_action_adapter(self) -> None:
        config = MangoMASBridgeConfig._from_dict({
            "action_adapter": {"bins_per_axis": 5}
        })
        assert config.action_adapter.bins_per_axis == 5

    def test_from_dict_with_observation_adapter(self) -> None:
        config = MangoMASBridgeConfig._from_dict({
            "observation_adapter": {"grid_summary_dim": 32}
        })
        assert config.observation_adapter.grid_summary_dim == 32

    def test_from_dict_with_sweep_nested(self) -> None:
        config = MangoMASBridgeConfig._from_dict({
            "sweep": {
                "c_puct": {"range": [1.0, 2.0], "steps": 3},
                "sim_budget": {"range": [20, 200], "steps": 4},
                "depth": {"range": [5, 50], "steps": 2},
                "discount": {"range": [0.95, 0.999], "steps": 2},
            }
        })
        assert config.sweep.c_puct_range == (1.0, 2.0)
        assert config.sweep.c_puct_steps == 3
        assert config.sweep.sim_budget_range == (20, 200)
        assert config.sweep.sim_budget_steps == 4

    def test_from_dict_sweep_with_surprise_ignored(self) -> None:
        config = MangoMASBridgeConfig._from_dict({
            "sweep": {"surprise": {"some": "data"}, "episodes_per_config": 10}
        })
        assert config.sweep.episodes_per_config == 10

    def test_from_toml_file(self, tmp_path: object) -> None:
        from pathlib import Path  # noqa: PLC0415

        path = Path(str(tmp_path)) / "test.toml"
        path.write_bytes(b'platform = "car"\n\n[action_adapter]\nbins_per_axis = 10\n')
        config = MangoMASBridgeConfig.from_toml(path)
        assert config.platform == "car"
        assert config.action_adapter.bins_per_axis == 10

    def test_from_dict_with_transfer_and_pipeline(self) -> None:
        config = MangoMASBridgeConfig._from_dict(
            {
                "transfer": {"bdi_mapping_overrides": {"39": 4}},
                "pipeline": {
                    "paths": {
                        "output_root": "artifacts/custom",
                        "run_name": "smoke-run",
                    },
                    "execution": {"resume": True, "stop_after_stage": "rssm"},
                },
            }
        )
        assert config.transfer.bdi_mapping_overrides == {39: 4}
        assert config.pipeline.paths.output_root == "artifacts/custom"
        assert config.pipeline.paths.run_name == "smoke-run"
        assert config.pipeline.execution.resume is True
        assert config.pipeline.execution.stop_after_stage == "rssm"
