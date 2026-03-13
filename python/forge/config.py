"""Configuration system for the FORGE Python agent framework.

Loads configuration from TOML files with environment variable overrides.
All config values flow through dataclasses — no hard-coded values.

Usage::

    from forge.config import ForgeConfig
    config = ForgeConfig.from_file("forge.toml")
    sim = config.simulation
    print(sim.grid_size)  # 64
"""

from __future__ import annotations

import logging
import os
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Environment variable prefix for overrides.
_ENV_PREFIX = "FORGE_"

# Default config file search paths (checked in order).
_DEFAULT_CONFIG_PATHS: tuple[str, ...] = (
    "forge.toml",
    "config/forge.toml",
    "../forge.toml",
)


def _apply_env_overrides(obj: Any, section: str) -> None:
    """Apply ``FORGE_<SECTION>_<FIELD>`` environment overrides to *obj*.

    Only fields that are declared on the dataclass are considered.
    Invalid values are logged and skipped.
    """
    for f in fields(obj):
        key = f"{_ENV_PREFIX}{section.upper()}_{f.name.upper()}"
        val = os.environ.get(key)
        if val is None:
            continue
        try:
            parsed: bool | int | float | str
            if f.type in ("bool", bool):
                parsed = val.lower() in ("1", "true", "yes")
            elif f.type in ("int", int):
                parsed = int(val)
            elif f.type in ("float", float):
                parsed = float(val)
            else:
                parsed = val
            setattr(obj, f.name, parsed)
            logger.debug("Env override applied: %s = %s", key, parsed)
        except (ValueError, TypeError) as exc:
            logger.warning("Invalid env override %s=%s: %s", key, val, exc)


@dataclass
class HardwareConfig:
    """Hardware and device configuration."""

    num_workers: int = 4
    device: str = "cpu"
    pin_memory: bool = False
    gpu_memory_fraction: float = 0.9


@dataclass
class SimulationConfig:
    """Simulation parameters matching the Rust ``ForgeConfig``."""

    grid_size: int = 64
    max_agents_per_team: int = 16
    seed: int = 0
    max_episode_length: int = 10_000
    fog_of_war: bool = True
    biome_scale: float = 0.1
    resource_density: float = 0.3
    day_night_cycle_length: int = 1000
    comm_vocab_size: int = 16
    comm_radius: int = 10
    schema_version: int = 1


DEFAULT_LEARNING_RATE: float = 3e-4
DEFAULT_GAMMA: float = 0.99
DEFAULT_GAE_LAMBDA: float = 0.95
DEFAULT_CLIP_RATIO: float = 0.2
DEFAULT_EPOCHS: int = 4
DEFAULT_BATCH_SIZE: int = 256
DEFAULT_ROLLOUT_LENGTH: int = 2048
DEFAULT_ENTROPY_COEFF: float = 0.01
DEFAULT_VALUE_COEFF: float = 0.5
DEFAULT_MAX_GRAD_NORM: float = 0.5

# Feature-extractor defaults (ForgeGridCnnExtractor / ForgeObsExtractor).
DEFAULT_CNN_CHANNELS: tuple[int, ...] = (32, 64)
DEFAULT_CNN_KERNEL_SIZES: tuple[int, ...] = (3, 3)
DEFAULT_CNN_STRIDES: tuple[int, ...] = (1, 1)
DEFAULT_FEATURES_DIM: int = 256
DEFAULT_MLP_HIDDEN_SIZES: tuple[int, ...] = (128,)

# Logging / curriculum defaults.
DEFAULT_LOG_FREQ: int = 1000
DEFAULT_CURRICULUM_TARGET_SUCCESS_RATE: float = 0.7
DEFAULT_CURRICULUM_WINDOW_SIZE: int = 100
DEFAULT_CURRICULUM_ADJUSTMENT_RATE: int = 1


@dataclass
class TrainingConfig:
    """Training hyper-parameters.

    All fields have sensible defaults so that existing code that constructs
    ``TrainingConfig()`` without arguments continues to work unchanged.
    New fields added here are backwards-compatible additions only.
    """

    learning_rate: float = DEFAULT_LEARNING_RATE
    gamma: float = DEFAULT_GAMMA
    gae_lambda: float = DEFAULT_GAE_LAMBDA
    clip_ratio: float = DEFAULT_CLIP_RATIO
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    rollout_length: int = DEFAULT_ROLLOUT_LENGTH
    target_success_rate: float = DEFAULT_CURRICULUM_TARGET_SUCCESS_RATE
    curriculum_window_size: int = DEFAULT_CURRICULUM_WINDOW_SIZE
    checkpoint_interval: int = 100
    entropy_coeff: float = DEFAULT_ENTROPY_COEFF
    value_coeff: float = DEFAULT_VALUE_COEFF
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM

    # --- Feature-extractor configuration (SB3 / CleanRL) -------------------
    cnn_channels: tuple[int, ...] = DEFAULT_CNN_CHANNELS
    cnn_kernel_sizes: tuple[int, ...] = DEFAULT_CNN_KERNEL_SIZES
    cnn_strides: tuple[int, ...] = DEFAULT_CNN_STRIDES
    features_dim: int = DEFAULT_FEATURES_DIM
    mlp_hidden_sizes: tuple[int, ...] = DEFAULT_MLP_HIDDEN_SIZES

    # --- Logging configuration ----------------------------------------------
    log_freq: int = DEFAULT_LOG_FREQ

    # --- Curriculum (mirrors CurriculumConfig on the Rust side) -------------
    curriculum_enabled: bool = False
    curriculum_adjustment_rate: int = DEFAULT_CURRICULUM_ADJUSTMENT_RATE


@dataclass
class ForgeConfig:
    """Top-level FORGE configuration.

    Loads from TOML files and applies environment variable overrides.
    Backwards compatible — all fields have sensible defaults.
    """

    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_file(cls, path: str | Path | None = None) -> ForgeConfig:
        """Load config from a TOML file with env overrides.

        Parameters
        ----------
        path:
            Explicit path to a TOML file.  If ``None``, searches
            :data:`_DEFAULT_CONFIG_PATHS` in order.

        Returns
        -------
        ForgeConfig
            Parsed configuration with env overrides applied.
        """
        try:
            import tomllib  # noqa: PLC0415  # Python 3.11+
        except ModuleNotFoundError:
            import tomli as tomllib  # noqa: PLC0415

        resolved = cls._resolve_path(path)
        if resolved is not None:
            raw = resolved.read_text(encoding="utf-8")
            data: dict[str, Any] = tomllib.loads(raw)
            logger.info("Loaded config from %s", resolved)
        else:
            data = {}
            if path is not None:
                logger.warning("Config file not found: %s — using defaults", path)
            else:
                logger.debug("No config file found — using defaults")

        return cls._from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ForgeConfig:
        """Create a config from a plain dictionary (e.g. from JSON)."""
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> ForgeConfig:
        hw = _build_section(HardwareConfig, data.get("hardware", {}))
        sim = _build_section(SimulationConfig, data.get("simulation", {}))
        train = _build_section(TrainingConfig, data.get("training", {}))

        _apply_env_overrides(hw, "HARDWARE")
        _apply_env_overrides(sim, "SIMULATION")
        _apply_env_overrides(train, "TRAINING")

        return cls(hardware=hw, simulation=sim, training=train)

    @staticmethod
    def _resolve_path(path: str | Path | None) -> Path | None:
        if path is not None:
            p = Path(path)
            return p if p.is_file() else None
        for candidate in _DEFAULT_CONFIG_PATHS:
            p = Path(candidate)
            if p.is_file():
                return p
        return None

    def to_rust_config(self) -> dict[str, Any]:
        """Convert to a dict compatible with the Rust ``ForgeConfig`` struct.

        This mapping bridges the Python config naming conventions to the
        Rust serde field names expected by the PyO3 bindings.
        """
        return {
            "world": {
                "width": self.simulation.grid_size,
                "height": self.simulation.grid_size,
                "seed": self.simulation.seed,
                "biome_scale": self.simulation.biome_scale,
                "resource_density": self.simulation.resource_density,
                "day_night_cycle_length": self.simulation.day_night_cycle_length,
            },
            "agents": {
                "num_agents": self.simulation.max_agents_per_team,
                "comm_vocab_size": self.simulation.comm_vocab_size,
                "comm_radius": self.simulation.comm_radius,
            },
            "task": {
                "max_episode_length": self.simulation.max_episode_length,
            },
            "curriculum": {
                "target_success_rate": self.training.target_success_rate,
                "window_size": self.training.curriculum_window_size,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialise the full config as a nested dict."""
        from dataclasses import asdict  # noqa: PLC0415

        return asdict(self)


def _build_section(cls: type, data: dict[str, Any]) -> Any:
    """Instantiate a dataclass from *data*, ignoring unknown keys.

    Lists loaded from TOML are converted to tuples for fields whose default
    value is a tuple, preserving the declared type contract.
    """
    # Collect fields that default to a tuple so we can coerce TOML lists.
    _tuple_fields: set[str] = set()
    for f in fields(cls):
        # Prefer an explicit default; fall back to default_factory if present.
        default: Any
        if f.default is not MISSING:
            default = f.default
        elif getattr(f, "default_factory", MISSING) is not MISSING:  # type: ignore[attr-defined]
            try:
                default = f.default_factory()  # type: ignore[misc]
            except TypeError:
                # Non-callable or requires arguments; treat as no usable default.
                continue
        else:
            continue
        if isinstance(default, tuple):
            _tuple_fields.add(f.name)

    known = {f.name for f in fields(cls)}
    filtered: dict[str, Any] = {}
    for k, v in data.items():
        if k not in known:
            continue
        if k in _tuple_fields and isinstance(v, list):
            filtered[k] = tuple(v)
        else:
            filtered[k] = v
    return cls(**filtered)
