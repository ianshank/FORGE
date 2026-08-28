"""MangoMAS bridge configuration dataclasses.

All values flow through config — no hard-coded constants.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from forge.cognitive.providers import DEFAULT_LMSTUDIO_BASE_URL

logger = logging.getLogger(__name__)

DEFAULT_SEED = 42
DEFAULT_BINS_PER_AXIS = 7
DEFAULT_GRID_SUMMARY_DIM = 64
DEFAULT_STATE_DIM_CAR = 18
DEFAULT_STATE_DIM_DRONE = 22
DEFAULT_NUM_BDI_INTENTIONS = 8
DEFAULT_NUM_CONSTITUTIONAL_CONSTRAINTS = 5
DEFAULT_MANGOMAS_ACTION_DIM = 75
DEFAULT_CURIOSITY_CHANNELS = (
    "social",
    "epistemic",
    "perceptual",
    "metacognitive",
)
DEFAULT_CURIOSITY_WEIGHTS = (0.4, 0.3, 0.2, 0.1)
DEFAULT_NUM_CURRICULUM_TIERS = 5
DEFAULT_TARGET_SUCCESS_RATE = 0.6
DEFAULT_MAX_STEPS = 1000
DEFAULT_NUM_ENVS = 8
DEFAULT_SWEEP_EPISODES = 50
DEFAULT_SWEEP_WORKERS = 4
DEFAULT_PIPELINE_OUTPUT_ROOT = "artifacts/mangomas"
DEFAULT_PIPELINE_EXPORT_DIR = "export"
DEFAULT_PIPELINE_LOG_FILE = "pipeline.log"
DEFAULT_PIPELINE_MANIFEST = "pipeline_manifest.json"

DEFAULT_TEACHER_PROVIDER = "lmstudio"
# Single source of truth lives in forge.cognitive.providers; aliasing keeps
# config.py's TeacherConfig stable without duplicating the URL literal.
DEFAULT_TEACHER_BASE_URL = DEFAULT_LMSTUDIO_BASE_URL
DEFAULT_TEACHER_MODEL = ""
DEFAULT_TEACHER_TEMPERATURE = 0.0
DEFAULT_TEACHER_TOP_P = 1.0
DEFAULT_TEACHER_MAX_TOKENS = 1024
DEFAULT_TEACHER_SEED = 42
DEFAULT_TEACHER_TIMEOUT_SECS = 120.0
DEFAULT_TEACHER_MAX_RETRIES = 2
DEFAULT_TEACHER_RETRY_BACKOFF_SECS = 1.0
DEFAULT_TEACHER_CONCURRENCY = 1
DEFAULT_TEACHER_OUTPUT_ROOT = "artifacts/teacher_traces"
DEFAULT_TEACHER_SHARD_SIZE = 1000
DEFAULT_TEACHER_TRACE_SCHEMA_VERSION = "1.0"
DEFAULT_TEACHER_PAYLOAD_PREVIEW_CHARS = 256

# Canonical constraint definitions for constitutional RL
DEFAULT_CONSTITUTIONAL_CONSTRAINTS: list[dict[str, Any]] = [
    {"name": "battery_minimum", "forge_field": "battery", "threshold": 0.2, "is_lower_bound": True},
    {
        "name": "altitude_ceiling",
        "forge_field": "altitude",
        "threshold": 0.9,
        "is_lower_bound": False,
    },
    {
        "name": "speed_ceiling",
        "forge_field": "stamina_inverse",
        "threshold": 0.8,
        "is_lower_bound": False,
    },
    {
        "name": "geofence",
        "forge_field": "boundary_distance",
        "threshold": 0.1,
        "is_lower_bound": True,
    },
    {
        "name": "threat_exclusion",
        "forge_field": "threat_proximity",
        "threshold": 0.3,
        "is_lower_bound": True,
    },
]

# Canonical tier definitions for platform curriculum
DEFAULT_CAR_TIERS: list[dict[str, Any]] = [
    {"tier": 1, "name": "Straight Line", "forge_scenario": "patrol", "success_threshold": 0.8},
    {"tier": 2, "name": "Obstacle Avoidance", "forge_scenario": "patrol", "success_threshold": 0.7},
    {"tier": 3, "name": "Multi-Waypoint", "forge_scenario": "patrol", "success_threshold": 0.6},
    {"tier": 4, "name": "Dynamic Traffic", "forge_scenario": "escort", "success_threshold": 0.5},
    {
        "tier": 5,
        "name": "Full Mission",
        "forge_scenario": "search_and_rescue",
        "success_threshold": 0.4,
    },
]

DEFAULT_DRONE_TIERS: list[dict[str, Any]] = [
    {"tier": 1, "name": "Hover and Altitude", "forge_scenario": "patrol", "success_threshold": 0.7},
    {
        "tier": 2,
        "name": "Waypoint Navigation",
        "forge_scenario": "patrol",
        "success_threshold": 0.6,
    },
    {"tier": 3, "name": "Patrol Pattern", "forge_scenario": "patrol", "success_threshold": 0.5},
    {
        "tier": 4,
        "name": "Search and Rescue",
        "forge_scenario": "search_and_rescue",
        "success_threshold": 0.4,
    },
    {"tier": 5, "name": "Multi-Drone Escort", "forge_scenario": "escort", "success_threshold": 0.3},
]


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
    seed: int = DEFAULT_SEED


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
    seed: int = DEFAULT_SEED


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
    state_dim: int = DEFAULT_STATE_DIM_CAR
    action_dim: int = DEFAULT_MANGOMAS_ACTION_DIM
    seed: int = DEFAULT_SEED
    constraints: list[dict[str, Any]] = field(
        default_factory=lambda: [dict(c) for c in DEFAULT_CONSTITUTIONAL_CONSTRAINTS]
    )


@dataclass
class RSSMPreTrainConfig:
    """RSSM world model pre-training configuration."""

    state_dim: int = DEFAULT_STATE_DIM_DRONE
    hidden_dim: int = 200
    latent_dim: int = 30
    action_dim: int = DEFAULT_MANGOMAS_ACTION_DIM
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
    seed: int = DEFAULT_SEED


@dataclass
class CurriculumConfig:
    """Platform curriculum configuration."""

    num_tiers: int = DEFAULT_NUM_CURRICULUM_TIERS
    target_success_rate: float = DEFAULT_TARGET_SUCCESS_RATE
    window_size: int = 100
    warmup_episodes: int = 20
    adjustment_rate: float = 0.1
    platform: str = "drone"
    tiers: list[dict[str, Any]] = field(default_factory=list)
    seed: int = DEFAULT_SEED

    def resolved_tiers(self, platform: str | None = None) -> list[dict[str, Any]]:
        """Return explicit curriculum tiers for the selected platform."""
        if self.tiers:
            return [dict(tier) for tier in self.tiers]

        selected_platform = platform or self.platform
        default_tiers = DEFAULT_DRONE_TIERS if selected_platform == "drone" else DEFAULT_CAR_TIERS
        return [dict(tier) for tier in default_tiers]


@dataclass
class BatchCollectorConfig:
    """Batch episode collection configuration."""

    max_steps: int = DEFAULT_MAX_STEPS
    num_envs: int = DEFAULT_NUM_ENVS
    seed: int = DEFAULT_SEED
    action_space_size: int = DEFAULT_MANGOMAS_ACTION_DIM
    log_interval: int = 100


@dataclass
class CuriosityOptimizerConfig:
    """Curiosity weight optimizer configuration."""

    channels: list[str] = field(default_factory=lambda: list(DEFAULT_CURIOSITY_CHANNELS))
    initial_weights: list[float] = field(default_factory=lambda: list(DEFAULT_CURIOSITY_WEIGHTS))
    population_size: int = 20
    sigma: float = 0.1
    learning_rate: float = 0.05
    seed: int = DEFAULT_SEED
    log_interval: int = 10


@dataclass
class TransferConfig:
    """Transfer-time overrides for MangoMAS weight initialization."""

    bdi_mapping_overrides: dict[int, int] = field(default_factory=dict)


@dataclass
class PipelinePathsConfig:
    """Filesystem layout for MangoMAS pipeline artifacts."""

    output_root: str = DEFAULT_PIPELINE_OUTPUT_ROOT
    run_name: str = ""
    export_dir_name: str = DEFAULT_PIPELINE_EXPORT_DIR
    log_file_name: str = DEFAULT_PIPELINE_LOG_FILE
    manifest_name: str = DEFAULT_PIPELINE_MANIFEST


@dataclass
class PipelineExecutionConfig:
    """Execution controls for the stage-based MangoMAS pipeline."""

    resume: bool = False
    fail_fast: bool = True
    stop_after_stage: str = ""


@dataclass
class PipelineLoggingConfig:
    """Logging options for MangoMAS pipeline runs."""

    level: str = "INFO"
    json_format: bool = False


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    paths: PipelinePathsConfig = field(default_factory=PipelinePathsConfig)
    execution: PipelineExecutionConfig = field(default_factory=PipelineExecutionConfig)
    logging: PipelineLoggingConfig = field(default_factory=PipelineLoggingConfig)


@dataclass
class MuZeroTrainerConfig:
    """MuZero training pipeline configuration."""

    latent_dim: int = 256
    hidden_dim: int = 256
    num_blocks: int = 4
    reward_support_size: int = 31
    value_support_size: int = 31
    num_unroll_steps: int = 5
    td_steps: int = 10
    discount: float = 0.997
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 256
    training_steps_per_iter: int = 100
    self_play_games_per_iter: int = 10
    num_simulations: int = 50
    c_puct: float = 1.25
    dirichlet_alpha: float = 0.3
    temperature_init: float = 1.0
    temperature_final: float = 0.25
    temperature_schedule_steps: int = 500
    buffer_capacity: int = 10000
    max_episode_steps: int = 500
    checkpoint_interval: int = 50
    seed: int = DEFAULT_SEED


@dataclass
class TeacherConfig:
    """LM Studio / Qwen offline teacher configuration.

    Drives both the cognitive provider (base URL, timeouts, retries) and
    the structured agent (template / schema / few-shots) so a single
    ``[teacher]`` TOML section captures the whole pipeline.
    """

    enabled: bool = False
    provider: str = DEFAULT_TEACHER_PROVIDER
    base_url: str = DEFAULT_TEACHER_BASE_URL
    model: str = DEFAULT_TEACHER_MODEL
    api_key: str = ""
    temperature: float = DEFAULT_TEACHER_TEMPERATURE
    top_p: float = DEFAULT_TEACHER_TOP_P
    max_tokens: int = DEFAULT_TEACHER_MAX_TOKENS
    seed: int = DEFAULT_TEACHER_SEED
    timeout_secs: float = DEFAULT_TEACHER_TIMEOUT_SECS
    max_retries: int = DEFAULT_TEACHER_MAX_RETRIES
    retry_backoff_secs: float = DEFAULT_TEACHER_RETRY_BACKOFF_SECS
    concurrency: int = DEFAULT_TEACHER_CONCURRENCY
    prompt_template_path: str = ""
    response_schema_path: str = ""
    few_shot_examples_path: str = ""
    system_prompt: str = ""
    output_root: str = DEFAULT_TEACHER_OUTPUT_ROOT
    log_payloads: bool = False
    validate_action: bool = True
    include_legal_actions: bool = True
    response_format_enabled: bool = True
    shard_size: int = DEFAULT_TEACHER_SHARD_SIZE
    compress_traces: bool = True
    trace_schema_version: str = DEFAULT_TEACHER_TRACE_SCHEMA_VERSION
    payload_preview_chars: int = DEFAULT_TEACHER_PAYLOAD_PREVIEW_CHARS


@dataclass
class MangoMASBridgeConfig:
    """Top-level MangoMAS integration configuration."""

    platform: str = "drone"  # "car" or "drone"
    action_adapter: ActionAdapterConfig = field(default_factory=ActionAdapterConfig)
    observation_adapter: ObservationAdapterConfig = field(default_factory=ObservationAdapterConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    surprise_validator: SurpriseValidatorConfig = field(default_factory=SurpriseValidatorConfig)
    bdi_trainer: BDITrainerConfig = field(default_factory=BDITrainerConfig)
    constitutional_trainer: ConstitutionalTrainerConfig = field(
        default_factory=ConstitutionalTrainerConfig
    )
    rssm_pretrain: RSSMPreTrainConfig = field(default_factory=RSSMPreTrainConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    batch_collector: BatchCollectorConfig = field(default_factory=BatchCollectorConfig)
    curiosity_optimizer: CuriosityOptimizerConfig = field(default_factory=CuriosityOptimizerConfig)
    muzero_trainer: MuZeroTrainerConfig = field(default_factory=MuZeroTrainerConfig)
    transfer: TransferConfig = field(default_factory=TransferConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)

    @classmethod
    def from_toml(cls, path: str | Path) -> MangoMASBridgeConfig:
        """Load configuration from a TOML file."""
        # sys.version_info (not try/except ModuleNotFoundError) so mypy
        # resolves exactly one branch statically instead of flagging a name
        # redefinition once its python_version target is 3.11+ (where
        # tomllib is unconditionally a stdlib module).
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib

        path = Path(path)
        with path.open("rb") as f:
            data = tomllib.load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> MangoMASBridgeConfig:  # noqa: PLR0912, PLR0915
        # Linear dispatcher over optional TOML sections. Splitting this into a
        # registry-style table would obscure the 1:1 correspondence between
        # TOML keys and config attributes, and the branches are intentionally
        # symmetric (one `if "X" in data` per config field).
        """Build config from a nested dictionary."""
        config = cls()
        if "platform" in data:
            config.platform = data["platform"]
        if "action_adapter" in data:
            config.action_adapter = ActionAdapterConfig(**data["action_adapter"])
        if "observation_adapter" in data:
            config.observation_adapter = ObservationAdapterConfig(**data["observation_adapter"])
        if "sweep" in data:
            sweep_data = dict(data["sweep"])
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
        if "surprise_validator" in data:
            config.surprise_validator = SurpriseValidatorConfig(**data["surprise_validator"])
        if "bdi_trainer" in data:
            config.bdi_trainer = BDITrainerConfig(**data["bdi_trainer"])
        if "constitutional_trainer" in data:
            config.constitutional_trainer = ConstitutionalTrainerConfig(
                **data["constitutional_trainer"]
            )
        if "rssm_pretrain" in data:
            config.rssm_pretrain = RSSMPreTrainConfig(**data["rssm_pretrain"])
        if "curriculum" in data:
            config.curriculum = CurriculumConfig(**data["curriculum"])
        if "batch_collector" in data:
            config.batch_collector = BatchCollectorConfig(**data["batch_collector"])
        if "curiosity_optimizer" in data:
            config.curiosity_optimizer = CuriosityOptimizerConfig(**data["curiosity_optimizer"])
        if "muzero_trainer" in data:
            config.muzero_trainer = MuZeroTrainerConfig(**data["muzero_trainer"])
        if "transfer" in data:
            transfer_data = dict(data["transfer"])
            overrides = transfer_data.get("bdi_mapping_overrides")
            if isinstance(overrides, dict):
                transfer_data["bdi_mapping_overrides"] = {
                    int(key): int(value) for key, value in overrides.items()
                }
            config.transfer = TransferConfig(**transfer_data)
        if "pipeline" in data:
            pipeline_data = data["pipeline"]
            config.pipeline = PipelineConfig(
                paths=PipelinePathsConfig(**pipeline_data.get("paths", {})),
                execution=PipelineExecutionConfig(**pipeline_data.get("execution", {})),
                logging=PipelineLoggingConfig(**pipeline_data.get("logging", {})),
            )
        if "teacher" in data:
            config.teacher = TeacherConfig(**data["teacher"])
        # Apply FORGE_TEACHER_<FIELD> env overrides to the (possibly TOML-loaded) teacher.
        from forge.utils.config_env import apply_env_overrides

        apply_env_overrides(config.teacher, "TEACHER")
        logger.debug("MangoMASBridgeConfig loaded: platform=%s", config.platform)
        return config
