"""MangoMAS bridge configuration dataclasses.

All values flow through config — no hard-coded constants.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_BINS_PER_AXIS = 7
DEFAULT_GRID_SUMMARY_DIM = 64
DEFAULT_STATE_DIM_CAR = 18
DEFAULT_STATE_DIM_DRONE = 22
DEFAULT_NUM_BDI_INTENTIONS = 8
DEFAULT_NUM_CONSTITUTIONAL_CONSTRAINTS = 5
DEFAULT_CURIOSITY_WEIGHTS = (0.4, 0.3, 0.2, 0.1)
DEFAULT_NUM_CURRICULUM_TIERS = 5
DEFAULT_TARGET_SUCCESS_RATE = 0.6
DEFAULT_MAX_STEPS = 1000
DEFAULT_NUM_ENVS = 8
DEFAULT_SWEEP_EPISODES = 50
DEFAULT_SWEEP_WORKERS = 4


@dataclass
class ActionAdapterConfig:
    """Configuration for continuous↔discrete action mapping."""

    bins_per_axis: int = DEFAULT_BINS_PER_AXIS
    continuous_range_min: float = -1.0
    continuous_range_max: float = 1.0


@dataclass
class ObservationAdapterConfig:
    """Configuration for FORGE observation → flat state vector."""

    grid_summary_dim: int = DEFAULT_GRID_SUMMARY_DIM
    include_inventory: bool = True
    include_drone_fields: bool = False
    position_scale: float = 16.0
    grid_channels: int = 11
    scalar_fields: int = 5
    inventory_fields: int = 2
    drone_fields: int = 4


@dataclass
class SweepConfig:
    """MCTS hyperparameter sweep configuration."""

    c_puct_range: tuple[float, float] = (0.5, 3.0)
    c_puct_steps: int = 6
    sim_budget_range: tuple[int, int] = (10, 500)
    sim_budget_steps: int = 5
    depth_range: tuple[int, int] = (10, 100)
    depth_steps: int = 4
    discount_range: tuple[float, float] = (0.9, 0.999)
    discount_steps: int = 4
    episodes_per_config: int = DEFAULT_SWEEP_EPISODES
    num_workers: int = DEFAULT_SWEEP_WORKERS
    seed: int = 42


@dataclass
class SurpriseValidatorConfig:
    """Surprise-adaptive budget validation configuration."""

    low_budget: int = 25
    base_budget: int = 100
    full_budget: int = 300
    low_threshold: float = 0.1
    high_threshold: float = 0.5
    episodes_per_level: int = 100


@dataclass
class BDITrainerConfig:
    """BDI pre-training configuration."""

    num_intentions: int = DEFAULT_NUM_BDI_INTENTIONS
    hidden_size: int = 128
    gru_layers: int = 1
    learning_rate: float = 1e-3
    batch_size: int = 64
    num_epochs: int = 50
    sequence_length: int = 50
    log_interval: int = 10
    default_intention: int = 7  # Idle


@dataclass
class ConstitutionalTrainerConfig:
    """Constitutional RL pre-training configuration."""

    num_constraints: int = DEFAULT_NUM_CONSTITUTIONAL_CONSTRAINTS
    penalty_weight: float = 10.0
    learning_rate: float = 3e-4
    batch_size: int = 64
    num_epochs: int = 100
    value_loss_weight: float = 0.5
    log_interval: int = 20
    constraints: list[dict[str, Any]] = field(default_factory=lambda: [
        {"name": "battery_minimum", "forge_field": "battery", "threshold": 0.2, "is_lower_bound": True},
        {"name": "altitude_ceiling", "forge_field": "altitude", "threshold": 0.9, "is_lower_bound": False},
        {"name": "speed_ceiling", "forge_field": "stamina_inverse", "threshold": 0.8, "is_lower_bound": False},
        {"name": "geofence", "forge_field": "boundary_distance", "threshold": 0.1, "is_lower_bound": True},
        {"name": "threat_exclusion", "forge_field": "threat_proximity", "threshold": 0.3, "is_lower_bound": True},
    ])


@dataclass
class RSSMPreTrainConfig:
    """RSSM world model pre-training configuration."""

    state_dim: int = DEFAULT_STATE_DIM_DRONE
    hidden_dim: int = 200
    latent_dim: int = 30
    action_dim: int = 75
    learning_rate: float = 3e-4
    batch_size: int = 50
    num_epochs: int = 100
    sequence_length: int = 50
    kl_weight: float = 1.0
    free_nats: float = 3.0
    reward_head_hidden: list[int] = field(default_factory=lambda: [200, 200])
    value_head_hidden: list[int] = field(default_factory=lambda: [200, 200])
    value_discount: float = 0.99
    reward_loss_scale: float = 0.01
    log_interval: int = 20


@dataclass
class CurriculumConfig:
    """Platform curriculum configuration."""

    num_tiers: int = DEFAULT_NUM_CURRICULUM_TIERS
    target_success_rate: float = DEFAULT_TARGET_SUCCESS_RATE
    window_size: int = 100
    warmup_episodes: int = 20
    adjustment_rate: float = 0.1
    tiers: list[dict[str, Any]] = field(default_factory=list)
    seed: int = 42


@dataclass
class BatchCollectorConfig:
    """Batch episode collection configuration."""

    max_steps: int = DEFAULT_MAX_STEPS
    num_envs: int = DEFAULT_NUM_ENVS
    seed: int = 42
    action_space_size: int = 75
    log_interval: int = 100


@dataclass
class CuriosityOptimizerConfig:
    """Curiosity weight optimizer configuration."""

    channels: list[str] = field(
        default_factory=lambda: ["social", "epistemic", "perceptual", "metacognitive"]
    )
    initial_weights: list[float] = field(default_factory=lambda: [0.4, 0.3, 0.2, 0.1])
    population_size: int = 20
    sigma: float = 0.1
    learning_rate: float = 0.05
    seed: int = 42
    log_interval: int = 10


@dataclass
class MangoMASBridgeConfig:
    """Top-level MangoMAS integration configuration."""

    platform: str = "drone"  # "car" or "drone"
    action_adapter: ActionAdapterConfig = field(default_factory=ActionAdapterConfig)
    observation_adapter: ObservationAdapterConfig = field(
        default_factory=ObservationAdapterConfig
    )
    sweep: SweepConfig = field(default_factory=SweepConfig)
    surprise_validator: SurpriseValidatorConfig = field(
        default_factory=SurpriseValidatorConfig
    )
    bdi_trainer: BDITrainerConfig = field(default_factory=BDITrainerConfig)
    constitutional_trainer: ConstitutionalTrainerConfig = field(
        default_factory=ConstitutionalTrainerConfig
    )
    rssm_pretrain: RSSMPreTrainConfig = field(default_factory=RSSMPreTrainConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    batch_collector: BatchCollectorConfig = field(default_factory=BatchCollectorConfig)
    curiosity_optimizer: CuriosityOptimizerConfig = field(
        default_factory=CuriosityOptimizerConfig
    )

    @classmethod
    def from_toml(cls, path: str | Path) -> MangoMASBridgeConfig:
        """Load configuration from a TOML file."""
        try:
            import tomllib  # noqa: PLC0415
        except ModuleNotFoundError:
            import tomli as tomllib  # noqa: PLC0415

        path = Path(path)
        with path.open("rb") as f:
            data = tomllib.load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> MangoMASBridgeConfig:
        """Build config from a nested dictionary."""
        config = cls()
        if "platform" in data:
            config.platform = data["platform"]
        if "action_adapter" in data:
            config.action_adapter = ActionAdapterConfig(**data["action_adapter"])
        if "observation_adapter" in data:
            config.observation_adapter = ObservationAdapterConfig(
                **data["observation_adapter"]
            )
        if "sweep" in data:
            sweep_data = data["sweep"]
            # Handle range fields that come as lists from TOML
            if "c_puct" in sweep_data:
                cp = sweep_data.pop("c_puct")
                sweep_data["c_puct_range"] = tuple(cp.get("range", [0.5, 3.0]))
                sweep_data["c_puct_steps"] = cp.get("steps", 6)
            if "sim_budget" in sweep_data:
                sb = sweep_data.pop("sim_budget")
                sweep_data["sim_budget_range"] = tuple(sb.get("range", [10, 500]))
                sweep_data["sim_budget_steps"] = sb.get("steps", 5)
            if "depth" in sweep_data:
                d = sweep_data.pop("depth")
                sweep_data["depth_range"] = tuple(d.get("range", [10, 100]))
                sweep_data["depth_steps"] = d.get("steps", 4)
            if "discount" in sweep_data:
                dc = sweep_data.pop("discount")
                sweep_data["discount_range"] = tuple(dc.get("range", [0.9, 0.999]))
                sweep_data["discount_steps"] = dc.get("steps", 4)
            # Remove surprise sub-table handled separately
            sweep_data.pop("surprise", None)
            config.sweep = SweepConfig(**sweep_data)
        logger.debug("MangoMASBridgeConfig loaded: platform=%s", config.platform)
        return config
