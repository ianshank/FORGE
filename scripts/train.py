#!/usr/bin/env python3
"""CLI training entrypoint for FORGE agents.

Wires the ForgeGymnasiumEnv to MAPPO, MCTS, or Random agents and runs
the training loop. All hyperparameters flow through ForgeConfig.

Usage::

    python scripts/train.py --config forge.toml --agent mappo --num-updates 10
    python scripts/train.py --agent random --episodes 100
    python scripts/train.py --agent mcts --episodes 50
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

# Add python/ to path so forge and forge_env packages are importable.
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

logger = logging.getLogger("forge.train")

# --- CLI argument defaults (no magic numbers) ---
_DEFAULT_EPISODES = 100
_DEFAULT_NUM_UPDATES = 10
_DEFAULT_SEED = 42
_DEFAULT_CHECKPOINT_DIR = "checkpoints"
_DEFAULT_LOG_LEVEL = "INFO"
_DEFAULT_MAX_EPISODE_STEPS = 1000
_DEFAULT_DASHBOARD_URL = ""
_DEFAULT_EVAL_INTERVAL = 0
_DEFAULT_EVAL_EPISODES = 10
_DEFAULT_EARLY_STOP_PATIENCE = 0
_DEFAULT_COLLECTION_POLICY = "random"
_DEFAULT_OPTIONAL_PATH = ""
_DEFAULT_MLFLOW_EXPERIMENT = "forge-train"
_AGENT_CHOICES = ("random", "mcts", "mappo", "mangomas", "mangomas-collect")
_COLLECTION_POLICY_CHOICES = ("random", "mcts", "llm")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the training script.

    Args:
        argv: Argument list to parse. Defaults to ``sys.argv[1:]``.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Train a FORGE agent")
    parser.add_argument("--config", type=str, default="forge.toml", help="Path to config file")
    parser.add_argument(
        "--agent",
        type=str,
        default="random",
        choices=list(_AGENT_CHOICES),
        help="Agent type to train",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=_DEFAULT_EPISODES,
        help="Training episodes",
    )
    parser.add_argument(
        "--num-updates",
        type=int,
        default=_DEFAULT_NUM_UPDATES,
        help="PPO update iterations (mappo only)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_DEFAULT_SEED,
        help="Random seed",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=_DEFAULT_CHECKPOINT_DIR,
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=_DEFAULT_LOG_LEVEL,
        help="Logging level",
    )
    parser.add_argument(
        "--dashboard-url",
        type=str,
        default=_DEFAULT_DASHBOARD_URL,
        help="URL of forge-server for live dashboard metrics (e.g. http://localhost:8080)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=_DEFAULT_EVAL_INTERVAL,
        help="Evaluate every N episodes (0=disabled)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=_DEFAULT_EVAL_EPISODES,
        help="Number of episodes per evaluation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Use dry-run config (small grid, short episodes)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=_DEFAULT_EARLY_STOP_PATIENCE,
        help="Stop training if reward doesn't improve for N evals (0=disabled)",
    )
    parser.add_argument(
        "--collection-policy",
        type=str,
        default=_DEFAULT_COLLECTION_POLICY,
        choices=list(_COLLECTION_POLICY_CHOICES),
        help="Action policy to use when collecting MangoMAS training data",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Scenario id or TOML path for MangoMAS collection; repeat to collect multiple scenarios",
    )
    parser.add_argument(
        "--mangomas-config",
        action="append",
        default=[],
        help="Path to a MangoMAS bridge TOML config; repeatable, last one wins",
    )
    parser.add_argument(
        "--pipeline-run-name",
        type=str,
        default=_DEFAULT_OPTIONAL_PATH,
        help="Optional MangoMAS pipeline run directory name override",
    )
    parser.add_argument(
        "--pipeline-output-root",
        type=str,
        default=_DEFAULT_OPTIONAL_PATH,
        help="Optional MangoMAS pipeline output root override",
    )
    parser.add_argument(
        "--collection-report-path",
        type=str,
        default=_DEFAULT_OPTIONAL_PATH,
        help="Optional JSON report path for MangoMAS collection summaries",
    )
    parser.add_argument(
        "--teacher-config",
        type=str,
        default=_DEFAULT_OPTIONAL_PATH,
        help=(
            "Path to a TOML preset with [teacher] section (e.g. "
            "configs/cognitive/qwen14b_teacher.toml). Required for "
            "--collection-policy llm unless [teacher] is already in the "
            "MangoMAS bridge config."
        ),
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        default=None,
        help="Override teacher.model (e.g. google/gemma-4-e4b, qwen2.5-14b-instruct).",
    )
    parser.add_argument(
        "--teacher-base-url",
        type=str,
        default=None,
        help="Override teacher.base_url (e.g. http://localhost:1234/v1).",
    )
    parser.add_argument(
        "--teacher-concurrency",
        type=int,
        default=None,
        help="Override teacher.concurrency (>=1). Triggers the asyncio path when >1.",
    )
    parser.add_argument(
        "--teacher-output-root",
        type=str,
        default=None,
        help="Override teacher.output_root for JSONL trace shards.",
    )
    # The BC stage runs automatically inside MangoMASPipeline whenever
    # CollectedTrainingData carries teacher labels (see pipeline._run_bc_stage).
    # No CLI flag is needed for that decision today; if a future operator
    # workflow requires forcing BC on/off, add a PipelineExecutionConfig
    # field and surface it explicitly.
    parser.add_argument(
        "--mlflow-enabled",
        action="store_true",
        default=False,
        help="Enable MLflow experiment tracking for this run.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help=(
            "Override MLflow tracking URI (e.g. http://mlflow:5000 or file:./mlruns). "
            "Falls back to MLFLOW_TRACKING_URI then the MLflow library default."
        ),
    )
    parser.add_argument(
        "--mlflow-registry-uri",
        type=str,
        default=None,
        help="Override MLflow model registry URI (defaults to MLFLOW_REGISTRY_URI).",
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default=None,
        help=(
            "MLflow experiment name. Defaults to MLFLOW_EXPERIMENT_NAME env var, "
            "then 'forge-train'."
        ),
    )
    parser.add_argument(
        "--mlflow-run-name",
        type=str,
        default=None,
        help="MLflow run name (defaults to MLFLOW_RUN_NAME, then '<agent>-<seed>').",
    )
    parser.add_argument(
        "--mlflow-artifact-location",
        type=str,
        default=None,
        help="Optional MLflow artifact location for new experiments.",
    )
    parser.add_argument(
        "--mlflow-tags",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Add MLflow tags in KEY=VALUE or KEY=VAL,KEY=VAL form. Repeatable.",
    )
    parser.add_argument(
        "--mlflow-system-metrics",
        action="store_true",
        default=False,
        help="Enable MLflow system-metrics autologging (CPU/GPU/RAM).",
    )
    args = parser.parse_args(argv)

    if args.eval_interval < 0:
        parser.error("--eval-interval must be non-negative")
    if args.eval_episodes < 1:
        parser.error("--eval-episodes must be at least 1")
    if args.early_stopping_patience < 0:
        parser.error("--early-stopping-patience must be non-negative")

    return args


def _load_mangomas_bridge_config(config_paths: list[str]) -> Any:
    """Load the MangoMAS bridge config, defaulting to in-code dataclass defaults."""
    from forge.mangomas.config import MangoMASBridgeConfig

    bridge_config = MangoMASBridgeConfig()
    for path in config_paths:
        bridge_config = MangoMASBridgeConfig.from_toml(path)
    return bridge_config


def _resolve_mangomas_scenarios(args: argparse.Namespace, bridge_config: Any) -> list[str]:
    """Resolve scenario refs from CLI or curriculum defaults."""
    if args.scenario:
        return [str(scenario) for scenario in args.scenario]

    scenarios: list[str] = []
    for tier in bridge_config.curriculum.resolved_tiers(bridge_config.platform):
        scenario = tier.get("forge_scenario")
        if scenario is not None:
            scenario_ref = str(scenario)
            if scenario_ref not in scenarios:
                scenarios.append(scenario_ref)

    if scenarios:
        return scenarios

    raise ValueError(
        "No MangoMAS scenarios were provided and the bridge curriculum does not define defaults"
    )


def _apply_mangomas_cli_overrides(bridge_config: Any, args: argparse.Namespace) -> None:
    """Apply CLI overrides to the loaded MangoMAS bridge configuration.

    Precedence: CLI flag (when explicitly provided) > env var > TOML >
    dataclass default. CLI flags default to ``None`` so absent flags do
    not clobber TOML/env values.
    """
    if args.pipeline_run_name:
        bridge_config.pipeline.paths.run_name = args.pipeline_run_name
    if args.pipeline_output_root:
        bridge_config.pipeline.paths.output_root = args.pipeline_output_root

    # Teacher overrides (only applied when explicitly provided).
    if getattr(args, "teacher_config", None):
        from forge.mangomas.config import MangoMASBridgeConfig

        teacher_cfg = MangoMASBridgeConfig.from_toml(args.teacher_config).teacher
        bridge_config.teacher = teacher_cfg
    if getattr(args, "teacher_model", None) is not None:
        bridge_config.teacher.model = args.teacher_model
    if getattr(args, "teacher_base_url", None) is not None:
        bridge_config.teacher.base_url = args.teacher_base_url
    if getattr(args, "teacher_concurrency", None) is not None:
        bridge_config.teacher.concurrency = int(args.teacher_concurrency)
    if getattr(args, "teacher_output_root", None) is not None:
        bridge_config.teacher.output_root = args.teacher_output_root


def _maybe_write_mangomas_report(
    collection_result: Any,
    scenario_refs: list[str],
    args: argparse.Namespace,
    bridge_config: Any,
) -> None:
    """Persist the optional MangoMAS collection report when requested."""
    if not args.collection_report_path:
        return

    from forge.mangomas.collector import write_collection_report

    write_collection_report(
        collection_result,
        args.collection_report_path,
        mode=args.agent,
        platform=bridge_config.platform,
        policy_name=args.collection_policy,
        base_seed=args.seed,
        scenario_refs=scenario_refs,
        config_paths=[args.config, *args.mangomas_config],
        run_name=args.pipeline_run_name or bridge_config.pipeline.paths.run_name or args.agent,
    )


def _collect_mangomas_training_data(config: Any, args: argparse.Namespace) -> tuple[Any, Any, list[str]]:
    """Collect MangoMAS training data using the configured FORGE scenarios."""
    from forge.mangomas.collector import collect_training_data_from_scenarios

    bridge_config = _load_mangomas_bridge_config(args.mangomas_config)
    _apply_mangomas_cli_overrides(bridge_config, args)
    scenario_refs = _resolve_mangomas_scenarios(args, bridge_config)

    teacher_config = bridge_config.teacher if args.collection_policy == "llm" else None

    collection_result = collect_training_data_from_scenarios(
        base_forge_config=config.to_rust_config(),
        mangomas_config=bridge_config,
        scenario_refs=scenario_refs,
        total_episodes=args.episodes,
        base_seed=args.seed,
        policy_name=args.collection_policy,
        teacher_config=teacher_config,
    )
    _maybe_write_mangomas_report(collection_result, scenario_refs, args, bridge_config)
    return bridge_config, collection_result, scenario_refs


def _create_env(config: Any) -> Any:
    """Create and return a ForgeGymnasiumEnv from config.

    Args:
        config: A ForgeConfig instance.

    Returns:
        A ForgeGymnasiumEnv environment.

    Raises:
        RuntimeError: If environment creation fails.
    """
    from forge_env.gymnasium_env import ForgeGymnasiumEnv

    try:
        rust_config = config.to_rust_config()
        env = ForgeGymnasiumEnv(config=rust_config)
    except Exception as exc:
        msg = f"Failed to create environment: {exc}"
        raise RuntimeError(msg) from exc

    logger.info("Environment created: action_space=%s", env.action_space)
    return env


def _build_mlflow_settings(args: argparse.Namespace) -> Any:
    """Compose MLflow settings from env vars and CLI overrides.

    Returns:
        An :class:`MlflowSettings` instance.  Defaults to env-driven
        values, then layers any CLI flags that were explicitly supplied.
    """
    from forge.training.mlflow_config import MlflowSettings, parse_tag_string

    base = MlflowSettings.from_env()
    cli_tags: dict[str, str] = {}
    for entry in getattr(args, "mlflow_tags", []) or []:
        cli_tags.update(parse_tag_string(entry))

    default_run_name = f"{args.agent}-seed{args.seed}"
    return base.merge(
        tracking_uri=args.mlflow_tracking_uri,
        registry_uri=args.mlflow_registry_uri,
        experiment_name=(
            args.mlflow_experiment or base.experiment_name or _DEFAULT_MLFLOW_EXPERIMENT
        ),
        run_name=(args.mlflow_run_name or base.run_name or default_run_name),
        artifact_location=args.mlflow_artifact_location,
        tags=cli_tags,
        log_system_metrics=True if args.mlflow_system_metrics else None,
    )


def _params_for_run(args: argparse.Namespace, config: Any) -> dict[str, Any]:
    """Flatten CLI args + config into an MLflow-friendly param dict.

    MLflow rejects ``None`` and only accepts string-coercible values, so
    we filter aggressively here rather than letting the SDK error out.
    """
    flat: dict[str, Any] = {}
    for key, value in vars(args).items():
        if value is None:
            continue
        if key.startswith("mlflow"):
            continue
        coerced: Any = value
        if isinstance(value, (list, tuple)):
            if not value:
                continue
            coerced = ",".join(str(item) for item in value)
        flat[f"cli.{key}"] = coerced

    config_dict = getattr(config, "to_dict", None)
    if callable(config_dict):
        try:
            for key, value in _flatten_for_params(config_dict()).items():
                flat[f"config.{key}"] = value
        except Exception:
            logger.debug("ForgeConfig.to_dict() failed; skipping config params", exc_info=True)
    return flat


def _flatten_for_params(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Recursively flatten a nested dict/list into dotted-key scalars."""
    flat: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            child_key = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten_for_params(value, child_key))
    elif isinstance(obj, (list, tuple)):
        if not obj:
            return flat
        flat[prefix] = ",".join(str(item) for item in obj)
    elif obj is not None:
        flat[prefix] = str(obj)
    return flat


def _maybe_make_mlflow_logger(args: argparse.Namespace, config: Any) -> Any:
    """Build an :class:`MLflowLogger` if ``--mlflow`` was supplied.

    Falls back to ``None`` so existing callers and CI workflows that
    don't opt in remain untouched.
    """
    if not getattr(args, "mlflow_enabled", False):
        return None
    try:
        from forge.training.loggers import MLflowLogger
    except ImportError:
        logger.warning("MLflow requested but loggers module unavailable; tracking disabled")
        return None

    settings = _build_mlflow_settings(args)
    try:
        mlflow_logger = MLflowLogger(settings=settings)
    except ImportError as exc:
        logger.warning("MLflow not installed (%s); continuing without tracking", exc)
        return None
    except Exception:
        logger.warning("Failed to initialise MLflowLogger; continuing without tracking", exc_info=True)
        return None

    mlflow_logger.log_params(_params_for_run(args, config))

    config_path = getattr(args, "config", None)
    if config_path and Path(config_path).exists():
        mlflow_logger.log_artifact(config_path, artifact_path="config")
    mlflow_logger.log_dict(settings.describe(), "config/mlflow_settings.json")
    return mlflow_logger


def _make_early_stopping(args: argparse.Namespace) -> Any:
    """Create an EarlyStopping instance if enabled via CLI args.

    Args:
        args: Parsed CLI arguments.

    Returns:
        An ``EarlyStopping`` instance, or ``None`` if disabled.
    """
    if args.early_stopping_patience > 0 and args.eval_interval > 0:
        from forge.training.stability import (
            EarlyStopping,
            EarlyStoppingConfig,
        )

        return EarlyStopping(
            EarlyStoppingConfig(patience=args.early_stopping_patience),
        )
    return None


def _train_mappo(  # noqa: PLR0915 - orchestration function spans many distinct stages
    env: Any,
    config: Any,
    args: argparse.Namespace,
    mlflow_logger: Any = None,
) -> None:
    """Train a MAPPO agent with PPO rollout collection and updates.

    Args:
        env: A ForgeGymnasiumEnv instance.
        config: A ForgeConfig instance.
        args: Parsed CLI arguments.
        mlflow_logger: Optional :class:`MLflowLogger` for tracking per-update
            metrics and checkpoint artifacts.
    """
    from forge.agents.mappo_agent import MAPPOAgent, MAPPOConfig
    from forge.training.checkpointing import CheckpointManager
    from forge.training.trainer import PPOTrainer, PPOTrainerConfig
    from forge.utils.observation import compute_obs_dim, flatten_obs

    obs_dim, action_dim = compute_obs_dim(env), int(env.action_space.n)
    logger.info("Env obs_dim=%d, action_dim=%d", obs_dim, action_dim)

    mappo_config = MAPPOConfig.from_forge_config(config)
    agent = MAPPOAgent(mappo_config, obs_dim=obs_dim, action_dim=action_dim)

    trainer_config = PPOTrainerConfig.from_forge_config(config)
    trainer_config.checkpoint_dir = args.checkpoint_dir
    trainer = PPOTrainer(agent=agent, config=trainer_config)

    checkpoint_mgr = CheckpointManager(args.checkpoint_dir)

    def reset_fn() -> NDArray[np.float32]:
        obs, _info = env.reset()
        return flatten_obs(obs)

    def step_fn(action: int) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = env.step(action)
        return flatten_obs(obs), reward, terminated, truncated, info

    # Optional dashboard client for live metrics streaming
    dashboard = None
    if getattr(args, "dashboard_url", ""):
        from forge.utils.dashboard_client import DashboardClient

        dashboard = DashboardClient(args.dashboard_url)

    early_stopping = _make_early_stopping(args)

    # Build eval callback if --eval-interval is set
    eval_cb = None
    if args.eval_interval > 0:
        trainer_config.eval_interval = args.eval_interval

        def _mappo_eval_callback(update: int, _agent: Any) -> None:
            from forge.evaluation.evaluator import EvalConfig, Evaluator

            evaluator = Evaluator(EvalConfig(num_episodes=args.eval_episodes, seed=args.seed))
            result = evaluator.evaluate(env, agent)
            logger.info(
                "Eval at update %d: reward_mean=%.3f\u00b1%.3f",
                update,
                result.reward_mean,
                result.reward_std,
            )
            if dashboard is not None:
                dashboard.post_training_metrics(
                    episode=trainer.episode_count,
                    total_steps=trainer.total_steps,
                    mean_reward=result.reward_mean,
                )
            if early_stopping is not None and early_stopping.step(result.reward_mean):
                logger.info(
                    "Early stopping at update %d (best=%.3f)",
                    update,
                    early_stopping.best_metric,
                )
                trainer.request_stop()

        eval_cb = _mappo_eval_callback

    logger.info("Starting MAPPO training: %d updates", args.num_updates)
    all_metrics = trainer.train(
        reset_fn, step_fn, num_updates=args.num_updates, eval_callback=eval_cb,
    )

    # Post each update's metrics to the dashboard
    for update_idx, metrics in enumerate(all_metrics, start=1):
        if dashboard is not None:
            dashboard.post_training_metrics(
                episode=int(metrics.get("episodes", 0)),
                total_steps=int(metrics.get("total_steps", 0)),
                mean_reward=metrics.get("mean_reward", 0.0),
                loss_policy=metrics.get("policy_loss", 0.0),
                loss_value=metrics.get("value_loss", 0.0),
                entropy=metrics.get("entropy", 0.0),
            )
        if mlflow_logger is not None:
            numeric_metrics = {
                key: float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            if numeric_metrics:
                mlflow_logger.log(numeric_metrics, step=update_idx)

    if all_metrics:
        checkpoint_mgr.save(agent, episode=trainer.episode_count, metrics=all_metrics[-1])
        logger.info("Final checkpoint saved to %s", args.checkpoint_dir)
        if mlflow_logger is not None and Path(args.checkpoint_dir).exists():
            mlflow_logger.log_artifacts(args.checkpoint_dir, artifact_path="checkpoints")

    if dashboard is not None:
        dashboard.close()

    logger.info(
        "Training complete: %d updates, %d episodes, %d total steps",
        len(all_metrics),
        trainer.episode_count,
        trainer.total_steps,
    )
    if all_metrics:
        final = all_metrics[-1]
        logger.info(
            "Final metrics: policy_loss=%.4f, value_loss=%.4f, entropy=%.4f",
            final.get("policy_loss", 0.0),
            final.get("value_loss", 0.0),
            final.get("entropy", 0.0),
        )


def _train_basic(
    env: Any,
    agent: Any,
    args: argparse.Namespace,
    mlflow_logger: Any = None,
) -> None:
    """Run episodes for non-learning agents (random, mcts).

    Args:
        env: A Gymnasium-compatible environment.
        agent: An agent implementing ``act(obs) -> (action, trace)``.
        args: Parsed CLI arguments.
        mlflow_logger: Optional :class:`MLflowLogger` to receive per-episode
            metrics.  When ``None`` (default) MLflow tracking is skipped.
    """
    from forge.utils.observation import flatten_obs

    early_stopping = _make_early_stopping(args)

    dashboard = None
    if getattr(args, "dashboard_url", ""):
        from forge.utils.dashboard_client import DashboardClient

        dashboard = DashboardClient(args.dashboard_url)

    log_interval = max(1, args.episodes // 10)
    total_steps = 0

    for episode in range(1, args.episodes + 1):
        obs, _info = env.reset()
        flat_obs = flatten_obs(obs)
        total_reward = 0.0
        done = False
        steps = 0

        while not done and steps < _DEFAULT_MAX_EPISODE_STEPS:
            action, _trace = agent.act(flat_obs)
            obs, reward, terminated, truncated, _info = env.step(action)
            flat_obs = flatten_obs(obs)
            total_reward += reward
            done = terminated or truncated
            steps += 1

        total_steps += steps

        if mlflow_logger is not None:
            mlflow_logger.log(
                {
                    "episode_reward": float(total_reward),
                    "episode_steps": float(steps),
                    "total_steps": float(total_steps),
                },
                step=episode,
            )

        if args.eval_interval > 0 and episode % args.eval_interval == 0:
            from forge.evaluation.evaluator import EvalConfig, Evaluator

            evaluator = Evaluator(EvalConfig(num_episodes=args.eval_episodes, seed=args.seed))
            result = evaluator.evaluate(env, agent)
            logger.info(
                "Eval at episode %d: reward_mean=%.3f\u00b1%.3f",
                episode,
                result.reward_mean,
                result.reward_std,
            )
            if mlflow_logger is not None:
                mlflow_logger.log(
                    {
                        "eval_reward_mean": float(result.reward_mean),
                        "eval_reward_std": float(result.reward_std),
                    },
                    step=episode,
                )
            if dashboard is not None:
                dashboard.post_training_metrics(
                    episode=episode,
                    total_steps=total_steps,
                    mean_reward=result.reward_mean,
                )
            if early_stopping is not None and early_stopping.step(result.reward_mean):
                logger.info(
                    "Early stopping at episode %d (best=%.3f)",
                    episode,
                    early_stopping.best_metric,
                )
                break

        if episode % log_interval == 0:
            logger.info(
                "Episode %d/%d: reward=%.2f, steps=%d",
                episode,
                args.episodes,
                total_reward,
                steps,
            )
            if dashboard is not None:
                dashboard.post_training_metrics(
                    episode=episode,
                    total_steps=total_steps,
                    mean_reward=total_reward,
                )

    if dashboard is not None:
        dashboard.close()

    logger.info("Training complete: %d episodes", args.episodes)


def main(argv: list[str] | None = None) -> None:  # noqa: PLR0915 - top-level CLI dispatcher
    """Run the FORGE training pipeline.

    Args:
        argv: Optional argument list for testing. Defaults to ``sys.argv[1:]``.
    """
    args = parse_args(argv)

    from forge.utils.logging_config import setup_logging

    setup_logging(level=args.log_level)

    from forge.config import ForgeConfig
    from forge.utils.seed import set_all_seeds

    try:
        config = ForgeConfig.from_file(args.config)
    except Exception as exc:
        logger.error("Failed to load config from '%s': %s", args.config, exc)
        sys.exit(1)

    if args.dry_run:
        config.dry_run.enabled = True
        logger.info("Dry-run mode enabled — using effective_simulation() overrides")

    set_all_seeds(args.seed)

    logger.info(
        "Starting training: agent=%s, seed=%d, config=%s",
        args.agent,
        args.seed,
        args.config,
    )

    try:
        if args.agent == "mangomas-collect":
            _bridge_config, collection_result, scenario_refs = _collect_mangomas_training_data(
                config, args
            )
            logger.info(
                "MangoMAS collection complete: scenarios=%d episodes=%d steps=%d",
                len(scenario_refs),
                collection_result.total_episodes(),
                collection_result.total_steps(),
            )
        elif args.agent == "mangomas":
            from forge.mangomas.pipeline import (
                MangoMASDroneTrainingPipeline,
            )

            bridge_config, collection_result, scenario_refs = _collect_mangomas_training_data(
                config, args
            )
            pipeline = MangoMASDroneTrainingPipeline(
                bridge_config,
                base_output_dir=args.pipeline_output_root or None,
            )
            pipeline_result = pipeline.run(
                collection_result.training_data,
                base_seed=args.seed,
                curriculum_outcomes=collection_result.curriculum_outcomes,
                run_name=args.pipeline_run_name or None,
            )
            logger.info(
                "MangoMAS pipeline complete: scenarios=%d episodes=%d manifest=%s",
                len(scenario_refs),
                collection_result.total_episodes(),
                pipeline_result.manifest_path,
            )
        else:
            env = _create_env(config)
            mlflow_logger = _maybe_make_mlflow_logger(args, config)
            try:
                if args.agent == "mappo":
                    _train_mappo(env, config, args, mlflow_logger=mlflow_logger)
                elif args.agent == "random":
                    from forge.agents.base_agent import AgentConfig
                    from forge.agents.random_agent import RandomAgent

                    action_dim: int = env.action_space.n
                    agent = RandomAgent(
                        config=AgentConfig(name="random"),
                        action_space_size=action_dim,
                        seed=args.seed,
                    )
                    _train_basic(env, agent, args, mlflow_logger=mlflow_logger)
                elif args.agent == "mcts":
                    from forge.agents.mcts_agent import MCTSAgent, MCTSConfig

                    action_dim = env.action_space.n
                    mcts_agent = MCTSAgent(
                        config=MCTSConfig(name="mcts"),
                        action_space_size=action_dim,
                        seed=args.seed,
                    )
                    _train_basic(env, mcts_agent, args, mlflow_logger=mlflow_logger)
            finally:
                if mlflow_logger is not None:
                    try:
                        mlflow_logger.close()
                    except Exception:
                        logger.warning("MLflow logger close failed", exc_info=True)
                env.close()
                logger.info("Environment closed")
    except Exception:
        logger.exception("Training failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
