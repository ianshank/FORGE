"""End-to-end long-run orchestrator for the FORGE evaluation pipeline.

Wires the four production components into a single resumable invocation:

    1.  Live LM Studio teacher (Gemma 4B by default)
    2.  MangoMAS scenario collector
    3.  numpy-path :class:`forge.mangomas.bc_trainer.BCTrainer` student
    4.  Rust ``forge-eval-longrun`` binary exporting to HTTP MLflow + HF JSONL

The orchestrator is the only place where these stages are stitched together;
the underlying components stay fully usable in isolation. A
``ProgressState`` checkpoint after collection means a crash mid-eval picks up
without re-collecting, and a crash mid-collection picks up at the next
episode boundary (granularity: scenario batch, not per-episode).

Configuration policy
--------------------
There are **no hard-coded URLs, paths, timeouts, or batch sizes** in this
file. Every leaf is sourced — in priority order — from:

    1.  CLI flags (``--config``, ``--output-dir``, ``--episodes``)
    2.  Environment variables (``FORGE_*``, listed in
        :data:`FORGE_E2E_ENV_VARS`)
    3.  The TOML preset declared by ``--config``

The env layer is what CI uses to point at a real MLflow service; the TOML
preset is for human-friendly local defaults. Mixing both is fine — env wins.

Logging
-------
INFO at every stage boundary (start/done), DEBUG per scenario. Subprocess
stdout/stderr inherits the parent terminal so the Rust ``tracing`` lines
interleave with the Python ones in CI logs.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from collections.abc import Sequence

    from forge.mangomas.collector import ScenarioCollectionResult

# scripts/_e2e_progress.py lives next to this file; sys.path mutation here
# matches the test harness in tests/python/test_e2e_progress.py.
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# python/ holds the in-tree ``forge`` package (collector, BCTrainer, config).
# Mirrors the path-mutation guard in scripts/train.py so this orchestrator
# runs from a fresh checkout without ``pip install -e .``.
_PYTHON_PKG_DIR = Path(__file__).resolve().parents[1] / "python"
if _PYTHON_PKG_DIR.is_dir() and str(_PYTHON_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_PKG_DIR))

from _e2e_progress import ProgressState  # noqa: E402
from _e2e_progress import load as load_progress  # noqa: E402
from _e2e_progress import save as save_progress  # noqa: E402

logger = logging.getLogger(__name__)


# Env vars the orchestrator honours, in the same "registry constant" style
# as crates/forge-server/src/config.rs::FORGE_SERVER_ENV_VARS. Keeping them
# in one tuple makes it trivial to document or pre-scrub in CI.
FORGE_E2E_ENV_VARS: tuple[str, ...] = (
    "FORGE_MLFLOW_TRACKING_URI",
    "FORGE_E2E_EPISODES",
    "FORGE_E2E_BATCH_SIZE",
    "FORGE_E2E_OUTPUT_DIR",
    "FORGE_EVAL_CLI_BIN",
    "FORGE_HF_EXPORT_ROOT",
    "FORGE_E2E_EXPERIMENT_NAME",
    "FORGE_E2E_RUN_ID",
    "FORGE_E2E_TEACHER_PRESET",
    "MLFLOW_TRACKING_TOKEN",
)

# Path to the FORGE repo root, computed once. All default paths are relative
# to it so the orchestrator can be invoked from any cwd.
REPO_ROOT: Path = Path(__file__).resolve().parents[1]

# Default location of the Rust CLI bin produced by
# ``cargo build -p forge-eval --bin forge-eval-longrun --features http-mlflow --release``.
# CI overrides via ``FORGE_EVAL_CLI_BIN`` when the binary lives elsewhere.
# Cargo appends ``.exe`` on Windows (``os.name == "nt"``); resolving the
# suffix here keeps the existence check in ``run_eval_subprocess`` portable
# without forcing every Windows caller to set ``FORGE_EVAL_CLI_BIN``.
_EVAL_CLI_BIN_NAME: str = "forge-eval-longrun.exe" if os.name == "nt" else "forge-eval-longrun"
DEFAULT_EVAL_CLI_BIN: Path = REPO_ROOT / "target" / "release" / _EVAL_CLI_BIN_NAME

# Default output root for orchestrator-owned artefacts (scorecard, BC
# weights, progress checkpoint). The Rust binary writes its own per-run
# directory under this root via FORGE_E2E_OUTPUT_DIR.
DEFAULT_OUTPUT_ROOT: Path = REPO_ROOT / "artifacts" / "e2e-long"


@dataclass(frozen=True)
class E2ELongConfig:
    """Resolved configuration after merging TOML preset + env + CLI flags."""

    scenario_refs: tuple[str, ...]
    total_episodes: int
    base_seed: int
    bc_epochs: int
    experiment_name: str
    run_id: str
    eval_cli_bin: Path
    mlflow_tracking_uri: str
    mlflow_batch_size: int
    hf_export_root: Path
    output_root: Path
    teacher_preset: str

    @classmethod
    def from_toml_with_env_override(
        cls,
        config_path: Path,
        *,
        output_dir_override: Path | None = None,
        episodes_override: int | None = None,
        progress_run_id: str | None = None,
    ) -> E2ELongConfig:
        """Resolve config: CLI flag > env var > TOML preset > built-in default.

        ``progress_run_id``, when provided, is preferred over a fresh UUID
        so resuming preserves the original run id (and therefore the MLflow
        parent run identity).
        """
        data = _read_toml(config_path)

        def _from_env(key: str, fallback: str) -> str:
            return os.environ.get(key, fallback)

        # ---- run-level --------------------------------------------------
        total_episodes = (
            episodes_override
            if episodes_override is not None
            else int(_from_env("FORGE_E2E_EPISODES", str(data["run"]["total_episodes"])))
        )
        experiment_name = _from_env(
            "FORGE_E2E_EXPERIMENT_NAME", str(data["run"]["experiment_name"])
        )
        teacher_preset = _from_env("FORGE_E2E_TEACHER_PRESET", str(data["teacher"]["preset"]))
        base_seed = int(data["run"]["base_seed"])
        bc_epochs = int(data["run"]["bc_epochs"])

        # ---- run id: explicit > env > resumed > fresh UUID --------------
        run_id = os.environ.get("FORGE_E2E_RUN_ID") or progress_run_id or uuid4().hex

        # ---- exporter targets -------------------------------------------
        mlflow_uri = _from_env("FORGE_MLFLOW_TRACKING_URI", str(data["mlflow"]["tracking_uri"]))
        # MLflow log-batch chunk size: env override wins; otherwise the TOML
        # preset value flows through. Forwarded to the Rust CLI via the
        # FORGE_E2E_BATCH_SIZE env var so the preset leaf is no longer dead.
        mlflow_batch_size = int(
            _from_env("FORGE_E2E_BATCH_SIZE", str(data["mlflow"]["batch_size"]))
        )
        hf_root = Path(_from_env("FORGE_HF_EXPORT_ROOT", str(data["huggingface"]["export_root"])))
        if not hf_root.is_absolute():
            hf_root = REPO_ROOT / hf_root

        # ---- output + bin paths -----------------------------------------
        output_root = (
            output_dir_override
            if output_dir_override is not None
            else Path(_from_env("FORGE_E2E_OUTPUT_DIR", str(DEFAULT_OUTPUT_ROOT)))
        )
        if not output_root.is_absolute():
            output_root = REPO_ROOT / output_root
        eval_cli_bin = Path(_from_env("FORGE_EVAL_CLI_BIN", str(DEFAULT_EVAL_CLI_BIN)))

        return cls(
            scenario_refs=tuple(str(ref) for ref in data["scenarios"]["refs"]),
            total_episodes=total_episodes,
            base_seed=base_seed,
            bc_epochs=bc_epochs,
            experiment_name=experiment_name,
            run_id=run_id,
            eval_cli_bin=eval_cli_bin,
            mlflow_tracking_uri=mlflow_uri,
            mlflow_batch_size=mlflow_batch_size,
            hf_export_root=hf_root,
            output_root=output_root,
            teacher_preset=teacher_preset,
        )


def _read_toml(path: Path) -> dict[str, Any]:
    """Load a TOML file as a dict; py39/3.10 fall back to ``tomli``."""
    # sys.version_info (not try/except ModuleNotFoundError) so mypy resolves
    # exactly one branch statically instead of flagging a name redefinition
    # once its python_version target is 3.11+ (where tomllib is
    # unconditionally a stdlib module).
    if sys.version_info >= (3, 11):
        import tomllib as _toml
    else:  # pragma: no cover - py39/py310
        import tomli as _toml
    with path.open("rb") as fh:
        loaded: dict[str, Any] = _toml.load(fh)
    return loaded


def run_collection(
    cfg: E2ELongConfig,
    *,
    remaining_episodes: int,
    start_seed: int,
) -> ScenarioCollectionResult:
    """Drive the MangoMAS scenario collector against the LM Studio teacher.

    Returns the raw :class:`ScenarioCollectionResult`. Lazy-imports the
    collector + teacher config because both pull in numpy + the native
    extension; orchestrator unit tests stub this function entirely.
    """
    # Imports are lazy so unit tests can mock the function without paying
    # for the heavy collector + native-ext import chain.
    from forge.mangomas.collector import collect_training_data_from_scenarios
    from forge.mangomas.config import MangoMASBridgeConfig

    teacher_preset_path = REPO_ROOT / "configs" / "cognitive" / f"{cfg.teacher_preset}.toml"
    if not teacher_preset_path.exists():
        msg = f"teacher preset not found at {teacher_preset_path}"
        raise FileNotFoundError(msg)
    mangomas_cfg = MangoMASBridgeConfig.from_toml(teacher_preset_path)

    # base_forge_config: load the first scenario's TOML as the base. The
    # collector overlays each scenario's own config on top via
    # _prepare_env_config, so the choice of base only matters for fields no
    # scenario sets.
    base_scenario_path = REPO_ROOT / "configs" / "scenarios" / f"{cfg.scenario_refs[0]}.toml"
    if not base_scenario_path.exists():
        msg = f"base scenario not found at {base_scenario_path}"
        raise FileNotFoundError(msg)
    base_forge = _read_toml(base_scenario_path)

    logger.info(
        "collect start: remaining=%d seed=%d scenarios=%d preset=%s",
        remaining_episodes,
        start_seed,
        len(cfg.scenario_refs),
        cfg.teacher_preset,
    )
    result = collect_training_data_from_scenarios(
        base_forge_config=base_forge,
        mangomas_config=mangomas_cfg,
        scenario_refs=list(cfg.scenario_refs),
        total_episodes=remaining_episodes,
        base_seed=start_seed,
        policy_name="llm",
        teacher_config=mangomas_cfg.teacher,
    )
    logger.info("collect done: episodes=%d steps=%d", result.total_episodes(), result.total_steps())
    return result


def run_bc_training(result: Any, cfg: E2ELongConfig) -> Path:
    """Flatten collected episodes through the numpy-path BC trainer.

    Returns the absolute path of the exported ``.npz`` weights file.
    """
    import numpy as np

    from forge.mangomas.bc_trainer import BCTrainer, BCTrainerConfig

    training = result.training_data
    obs_episodes = training.step_observations()
    action_episodes = [episode.astype(np.int64, copy=False) for episode in training.action_ids]
    # Authoritative action-space size (mirrors pipeline.py:424-427). Falling
    # back to flat_actions.max()+1 would underestimate the space whenever no
    # episode happens to select the highest legal action.
    if getattr(training, "action_space_sizes", None):
        num_actions = int(max(training.action_space_sizes))
    else:
        max_observed = int(
            max((int(arr.max()) for arr in action_episodes if arr.size > 0), default=0)
        )
        num_actions = max_observed + 1
    logger.info(
        "bc start: episodes=%d num_actions=%d epochs=%d",
        len(obs_episodes),
        num_actions,
        cfg.bc_epochs,
    )

    dataset = BCTrainer.build_dataset(
        obs_episodes,
        action_episodes,
        num_actions=num_actions,
        top_k_probs=getattr(training, "teacher_top_k_probs", None) or None,
        value_hats=getattr(training, "teacher_value_hats", None) or None,
    )
    trainer = BCTrainer(BCTrainerConfig(num_epochs=cfg.bc_epochs, seed=cfg.base_seed))
    train_result = trainer.train(dataset)
    logger.info(
        "bc done: epochs=%d final_loss=%.4f final_acc=%.4f",
        train_result.epochs_run,
        train_result.final_loss,
        train_result.final_top1_accuracy,
    )

    weights_path = cfg.output_root / "bc_weights.npz"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.export_weights(weights_path)
    return weights_path


def run_eval_subprocess(cfg: E2ELongConfig, weights_path: Path) -> None:
    """Invoke the Rust ``forge-eval-longrun`` binary via subprocess.

    Args + env are entirely derived from ``cfg``; the binary's own ``--help``
    documents the full surface.
    """
    if not cfg.eval_cli_bin.exists():
        msg = (
            f"forge-eval-longrun binary not found at {cfg.eval_cli_bin}. Build it with: "
            "cargo build -p forge-eval --bin forge-eval-longrun --features http-mlflow --release"
        )
        raise FileNotFoundError(msg)

    suite_dir = REPO_ROOT / "configs" / "scenarios"
    # Derive per-scenario episode budget from the orchestrator's total
    # (the Rust harness multiplies episodes_per_scenario by len(suite));
    # round up so a non-divisible total never silently truncates the run.
    num_scenarios = max(1, len(cfg.scenario_refs))
    episodes_per_scenario = max(1, -(-cfg.total_episodes // num_scenarios))
    env = {
        **os.environ,
        "FORGE_E2E_SUITE": str(suite_dir),
        "FORGE_E2E_EPISODES": str(cfg.total_episodes),
        # The Rust CLI reads FORGE_E2E_EPISODES_PER_SCENARIO; without this
        # the binary would always default to 1, ignoring --episodes /
        # FORGE_E2E_EPISODES entirely.
        "FORGE_E2E_EPISODES_PER_SCENARIO": str(episodes_per_scenario),
        "FORGE_E2E_BATCH_SIZE": str(cfg.mlflow_batch_size),
        "FORGE_MLFLOW_TRACKING_URI": cfg.mlflow_tracking_uri,
        "FORGE_E2E_EXPERIMENT_NAME": cfg.experiment_name,
        "FORGE_E2E_RUN_ID": cfg.run_id,
        "FORGE_HF_EXPORT_ROOT": str(cfg.hf_export_root),
        "FORGE_E2E_OUTPUT_DIR": str(cfg.output_root),
        # BC weights path surfaced for any downstream agent factory that
        # wants to load them; the bin itself uses noop_agent_factory today.
        "FORGE_E2E_BC_WEIGHTS": str(weights_path),
    }
    logger.info(
        "eval subprocess start: bin=%s suite=%s uri=%s episodes=%d eps_per_scn=%d batch=%d run_id=%s",
        cfg.eval_cli_bin,
        suite_dir,
        cfg.mlflow_tracking_uri,
        cfg.total_episodes,
        episodes_per_scenario,
        cfg.mlflow_batch_size,
        cfg.run_id,
    )
    subprocess.run([str(cfg.eval_cli_bin)], env=env, check=True)
    logger.info("eval subprocess done")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint. Returns the subprocess exit code so CI gates work."""
    parser = argparse.ArgumentParser(
        description="FORGE end-to-end long-run orchestrator (collector -> BC -> eval)",
    )
    parser.add_argument("--config", type=Path, required=True, help="path to e2e_long_preset.toml")
    parser.add_argument("--output-dir", type=Path, default=None, help="override [output_root]")
    parser.add_argument("--episodes", type=int, default=None, help="override total_episodes")
    parser.add_argument("--log-level", default="INFO", help="root logger level (DEBUG/INFO/WARN)")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Peek at the checkpoint before resolving config so we can preserve the
    # original run_id across a resume. Use the override dir if given,
    # otherwise the env/TOML-resolved one.
    peek_output_dir = args.output_dir or (
        Path(os.environ["FORGE_E2E_OUTPUT_DIR"])
        if "FORGE_E2E_OUTPUT_DIR" in os.environ
        else DEFAULT_OUTPUT_ROOT
    )
    if not peek_output_dir.is_absolute():
        peek_output_dir = REPO_ROOT / peek_output_dir
    progress_path = peek_output_dir / ".e2e_progress.json"
    existing = load_progress(progress_path)
    progress_run_id = existing.run_id if existing is not None else None

    cfg = E2ELongConfig.from_toml_with_env_override(
        args.config,
        output_dir_override=args.output_dir,
        episodes_override=args.episodes,
        progress_run_id=progress_run_id,
    )
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = cfg.output_root / ".e2e_progress.json"

    progress = existing or ProgressState(
        run_id=cfg.run_id,
        episodes_completed=0,
        scenario_cursor=0,
        last_seed=cfg.base_seed,
    )
    logger.info(
        "e2e long-run: run_id=%s total=%d completed=%d output=%s",
        progress.run_id,
        cfg.total_episodes,
        progress.episodes_completed,
        cfg.output_root,
    )

    weights_path = cfg.output_root / "bc_weights.npz"

    if progress.episodes_completed < cfg.total_episodes:
        remaining = cfg.total_episodes - progress.episodes_completed
        start_seed = cfg.base_seed + progress.episodes_completed
        result = run_collection(cfg, remaining_episodes=remaining, start_seed=start_seed)
        # Run BC training BEFORE marking the collection step complete so a
        # crash here doesn't permanently skip the trainer on resume (would
        # otherwise leave bc_weights.npz missing/stale and proceed to eval).
        weights_path = run_bc_training(result, cfg)
        progress = ProgressState(
            run_id=progress.run_id,
            episodes_completed=cfg.total_episodes,
            scenario_cursor=len(cfg.scenario_refs),
            last_seed=start_seed,
        )
        save_progress(progress_path, progress)
    else:
        logger.info(
            "collection already complete (episodes_completed=%d >= total=%d); skipping",
            progress.episodes_completed,
            cfg.total_episodes,
        )
        if not weights_path.exists():
            logger.warning(
                "resumed past collection but %s is missing; eval may fail. Re-run with a fresh "
                "--output-dir or delete the progress checkpoint to re-collect.",
                weights_path,
            )

    run_eval_subprocess(cfg, weights_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
