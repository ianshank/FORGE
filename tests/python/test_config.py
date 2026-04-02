"""Tests for the FORGE Python configuration system."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import pytest
from forge.config import (
    ForgeConfig,
    HardwareConfig,
    SimulationConfig,
    TrainingConfig,
    _build_section,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_hardware_config() -> None:
    hw = HardwareConfig()
    assert hw.num_workers == 4
    assert hw.device == "cpu"
    assert hw.pin_memory is False
    assert hw.gpu_memory_fraction == 0.9


def test_default_simulation_config() -> None:
    sim = SimulationConfig()
    assert sim.grid_size == 64
    assert sim.max_agents_per_team == 16
    assert sim.seed == 0
    assert sim.fog_of_war is True


def test_default_training_config() -> None:
    train = TrainingConfig()
    assert train.learning_rate == pytest.approx(3e-4)
    assert train.gamma == pytest.approx(0.99)
    assert train.batch_size == 256


def test_default_forge_config() -> None:
    cfg = ForgeConfig()
    assert isinstance(cfg.hardware, HardwareConfig)
    assert isinstance(cfg.simulation, SimulationConfig)
    assert isinstance(cfg.training, TrainingConfig)


# ---------------------------------------------------------------------------
# from_file
# ---------------------------------------------------------------------------


def test_from_file_explicit(tmp_path: Path) -> None:
    toml_content = textwrap.dedent("""\
        [simulation]
        grid_size = 128
        seed = 42

        [hardware]
        device = "cuda"
    """)
    p = tmp_path / "test.toml"
    p.write_text(toml_content, encoding="utf-8")

    cfg = ForgeConfig.from_file(p)
    assert cfg.simulation.grid_size == 128
    assert cfg.simulation.seed == 42
    assert cfg.hardware.device == "cuda"
    # Defaults still work
    assert cfg.training.gamma == pytest.approx(0.99)


def test_from_file_missing_uses_defaults() -> None:
    cfg = ForgeConfig.from_file("/nonexistent/config.toml")
    defaults = ForgeConfig()
    assert cfg.simulation.grid_size == defaults.simulation.grid_size


def test_from_file_none_uses_defaults() -> None:
    # When no default path exists, should return defaults
    cfg = ForgeConfig.from_file(None)
    assert cfg.simulation.grid_size == 64


# ---------------------------------------------------------------------------
# from_dict
# ---------------------------------------------------------------------------


def test_from_dict_partial() -> None:
    data = {"simulation": {"grid_size": 96}, "hardware": {"num_workers": 8}}
    cfg = ForgeConfig.from_dict(data)
    assert cfg.simulation.grid_size == 96
    assert cfg.hardware.num_workers == 8
    assert cfg.training.epochs == 4  # default


def test_from_dict_empty() -> None:
    cfg = ForgeConfig.from_dict({})
    assert cfg.simulation.grid_size == 64


def test_from_dict_unknown_keys_ignored() -> None:
    data = {"simulation": {"grid_size": 48, "unknown_field": 999}}
    cfg = ForgeConfig.from_dict(data)
    assert cfg.simulation.grid_size == 48
    assert not hasattr(cfg.simulation, "unknown_field")


# ---------------------------------------------------------------------------
# Environment overrides
# ---------------------------------------------------------------------------


def test_env_overrides_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_SIMULATION_GRID_SIZE", "256")
    monkeypatch.setenv("FORGE_HARDWARE_DEVICE", "mps")
    cfg = ForgeConfig.from_dict({})
    assert cfg.simulation.grid_size == 256
    assert cfg.hardware.device == "mps"


def test_env_override_bool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_SIMULATION_FOG_OF_WAR", "false")
    cfg = ForgeConfig.from_dict({})
    assert cfg.simulation.fog_of_war is False


def test_env_override_invalid_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_SIMULATION_GRID_SIZE", "not_a_number")
    cfg = ForgeConfig.from_dict({})
    assert cfg.simulation.grid_size == 64  # unchanged default


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def test_to_dict_roundtrip() -> None:
    cfg = ForgeConfig()
    d = cfg.to_dict()
    assert d["simulation"]["grid_size"] == 64
    assert d["hardware"]["device"] == "cpu"
    assert d["training"]["gamma"] == pytest.approx(0.99)


def test_to_rust_config() -> None:
    cfg = ForgeConfig()
    rust = cfg.to_rust_config()
    assert rust["world"]["width"] == 64
    assert rust["world"]["height"] == 64
    assert rust["agents"]["num_agents"] == 16
    assert rust["task"]["max_episode_length"] == 10_000


def test_to_rust_config_custom() -> None:
    cfg = ForgeConfig(simulation=SimulationConfig(grid_size=128, seed=99, max_agents_per_team=8))
    rust = cfg.to_rust_config()
    assert rust["world"]["width"] == 128
    assert rust["world"]["seed"] == 99
    assert rust["agents"]["num_agents"] == 8


# ---------------------------------------------------------------------------
# _build_section helper
# ---------------------------------------------------------------------------


def test_build_section_filters_unknown() -> None:
    result = _build_section(HardwareConfig, {"num_workers": 2, "fake": 99})
    assert result.num_workers == 2
    assert not hasattr(result, "fake")


def test_build_section_empty() -> None:
    result = _build_section(SimulationConfig, {})
    assert result.grid_size == 64


# ---------------------------------------------------------------------------
# DryRunConfig
# ---------------------------------------------------------------------------


class TestDryRunConfig:
    def test_dry_run_defaults(self) -> None:
        from forge.config import DryRunConfig

        cfg = DryRunConfig()
        assert cfg.enabled is False
        assert cfg.grid_size == 8
        assert cfg.max_episode_length == 50
        assert cfg.max_episodes == 5
        assert cfg.max_agents == 2
        assert cfg.curriculum_max_tier == 2

    def test_effective_simulation_uses_dry_run(self) -> None:
        from forge.config import DryRunConfig, ForgeConfig

        config = ForgeConfig(dry_run=DryRunConfig(enabled=True))
        effective = config.effective_simulation()
        assert effective.grid_size == 8
        assert effective.max_episode_length == 50

    def test_effective_simulation_normal_mode(self) -> None:
        from forge.config import ForgeConfig

        config = ForgeConfig()
        effective = config.effective_simulation()
        assert effective.grid_size == 64

    def test_dry_run_env_override(self) -> None:
        import os

        os.environ["FORGE_DRY_RUN_ENABLED"] = "true"
        try:
            from forge.config import ForgeConfig

            config = ForgeConfig.from_dict({})
            assert config.dry_run.enabled is True
        finally:
            del os.environ["FORGE_DRY_RUN_ENABLED"]
