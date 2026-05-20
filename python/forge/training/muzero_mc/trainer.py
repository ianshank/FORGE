"""MuZero training loop for the Minecraft RL integration.

Consumes ``TrajectoryV2`` JSONL written by ``forge_mc_runner``, drives
the MuZero loss via the shared :func:`forge.training._muzero_step.train_with_gradients`
helper, periodically exports an ONNX bundle, and bumps the manifest
version the Rust runner's hot-reload watcher polls.

Design contract:

- **Reuse first**: loss + clipping + L2 + gradient scaling come from
  :mod:`forge.training._muzero_step` (extracted in the preceding
  commit). The n-step value-target formula comes from
  :func:`forge.training._targets.compute_n_step_return`. No
  parallel hyperparameter struct — model knobs flow from
  ``MuZeroConfig``, gradient-step knobs from
  ``MuZeroStepConfig``.
- **Atomic manifest bump**: the trainer writes the manifest via
  :func:`forge.training.muzero_mc.manifest.save_manifest`, which
  already uses the ``.tmp + os.replace`` discipline so the Rust
  runner's :class:`HotReloadWatcher` never sees a half-written file.
- **No hard-coded values**: every numeric flows through
  :class:`MuZeroMcTrainerConfig` (trainer knobs) or
  :class:`forge.models.muzero_config.MuZeroConfig` (loss-side knobs).

The CLI surface is added in :mod:`forge.training.muzero_mc.cli`'s
``train`` subcommand.
"""

from __future__ import annotations

__all__ = [
    "MuZeroMcTrainerConfig",
    "MuzeroMcTrainer",
    "build_batch_from_trajectory",
]

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from forge.training._targets import compute_n_step_return
from forge.training.muzero_mc.manifest import (
    DEFAULT_BUNDLE_FILENAMES,
    ManifestError,
    build_manifest,
    load_manifest,
    save_manifest,
)

if TYPE_CHECKING:
    from forge.models.muzero_world_model import MuZeroWorldModel
    from forge.training.muzero_mc.replay import TrajectoryReader

logger = logging.getLogger(__name__)

#: Default checkpoint cadence — how often the trainer exports an ONNX
#: bundle and bumps the manifest version. Tuned so a typical 1-hour
#: training run produces ~10 hot-reloadable checkpoints.
DEFAULT_EXPORT_EVERY_N_ITERS: int = 100

#: Default log cadence. Loud enough to spot regressions, quiet enough
#: not to flood TTY-attached training shells.
DEFAULT_LOG_EVERY_N_ITERS: int = 10

#: Default batch size for training. Mirrors the existing
#: ``MuZeroTrainerConfig`` default for symmetry.
DEFAULT_BATCH_SIZE: int = 32

#: Default torch device the trainer constructs the model on. ``"cpu"``
#: is the lowest-common-denominator that runs on every host (laptop +
#: CI). Override to ``"cuda"`` for GPU training or ``"auto"`` to let
#: the trainer pick ``cuda`` if ``torch.cuda.is_available()``.
DEFAULT_DEVICE: Final[str] = "cpu"

#: Allowed values for ``MuZeroMcTrainerConfig.device``. Pinned as a
#: tuple-literal so a typo at config-time surfaces in ``__post_init__``
#: validation rather than as a confusing torch error later.
_ALLOWED_DEVICES: Final[tuple[str, ...]] = ("cpu", "cuda", "auto")


def _resolve_device(device: str) -> Any:  # actually torch.device, but kept Any to avoid the runtime torch import here
    """Resolve a config-level device string to a concrete
    ``torch.device``.

    - ``"cpu"`` / ``"cuda"`` map to the corresponding device.
    - ``"auto"`` picks ``cuda`` if ``torch.cuda.is_available()`` else
      ``cpu``. Logged at INFO so the operator sees which path was
      chosen.

    Lazy-imports torch so callers that just want to validate a config
    (e.g. CLI ``--help``) don't pay the import cost.
    """
    import torch

    if device == "auto":
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("device=auto resolved to %s", resolved)
        return torch.device(resolved)
    return torch.device(device)


@dataclass
class MuZeroMcTrainerConfig:
    """Trainer-side knobs that are NOT part of the model's
    ``MuZeroConfig``.

    Attributes:
        train_iters: Total gradient steps to run. ``0`` is a no-op.
        export_every_n_iters: Export the ONNX bundle + bump the
            manifest version every N gradient steps. Set to ``0`` to
            disable mid-run exports (only the initial bundle, if
            present, is honoured).
        log_every_n_iters: Emit a `tracing::info` line every N
            gradient steps with the loss components.
        output_dir: Bundle directory. The trainer writes
            ``representation.onnx`` / ``dynamics.onnx`` /
            ``prediction.onnx`` plus ``model_manifest.json`` here.
            Created on first export if missing.
        manifest_path: Path to the manifest file. If absent, the
            first export writes version 1; otherwise, every export
            increments by 1 starting from the last seen value.
        schema_id: sha256 the env handshake advertises. Stamped on
            every exported manifest; must match the env handshake.
        batch_size: Number of (obs, action, ...) tuples per gradient
            step.
        seed: RNG seed for batch sampling and gradient noise.
            ``None`` derives a seed from the system clock.
        max_grad_norm: Forwarded to ``MuZeroStepConfig.max_grad_norm``.
        gradient_scale: Forwarded to ``MuZeroStepConfig.gradient_scale``.
    """

    train_iters: int = 100
    export_every_n_iters: int = DEFAULT_EXPORT_EVERY_N_ITERS
    log_every_n_iters: int = DEFAULT_LOG_EVERY_N_ITERS
    output_dir: Path = field(default_factory=lambda: Path("models"))
    manifest_path: Path | None = None
    schema_id: str = "unset"
    batch_size: int = DEFAULT_BATCH_SIZE
    seed: int | None = 0
    max_grad_norm: float = 1.0
    gradient_scale: float = 0.5
    #: ``"cpu"`` / ``"cuda"`` / ``"auto"``. See :data:`DEFAULT_DEVICE`.
    device: str = DEFAULT_DEVICE

    def __post_init__(self) -> None:
        if self.train_iters < 0:
            raise ValueError(f"train_iters must be >= 0, got {self.train_iters}")
        if self.export_every_n_iters < 0:
            raise ValueError(
                f"export_every_n_iters must be >= 0, got {self.export_every_n_iters}"
            )
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if not self.schema_id:
            raise ValueError("schema_id must be non-empty")
        if self.device not in _ALLOWED_DEVICES:
            raise ValueError(
                f"device must be one of {_ALLOWED_DEVICES!r}, got {self.device!r}"
            )


def build_batch_from_trajectory(
    trajectory: dict[str, Any],
    *,
    indices: list[int],
    num_unroll_steps: int,
    td_steps: int,
    discount: float,
) -> dict[str, Any]:
    """Slice a single ``TrajectoryV2`` dict into a MuZero-shaped
    training batch.

    Returns a dict matching ``MuZeroReplayBuffer.sample_batch``'s
    keys: ``observations``, ``actions``, ``target_values``,
    ``target_rewards``, ``target_policies``. The shapes are aligned
    so :func:`forge.training._muzero_step.train_with_gradients` can
    consume the result as-is.

    Args:
        trajectory: A loaded ``TrajectoryV2`` dict.
        indices: Per-row starting step positions (one entry per
            batch row).
        num_unroll_steps: K-step MuZero unroll budget.
        td_steps: Bootstrap horizon for n-step value targets.
        discount: Reward discount factor (typically near 1).

    Returns:
        A dict whose values are Python lists. Caller converts to
        tensors at the boundary (the gradient helper handles this).
    """
    steps = trajectory["steps"]
    action_dim = int(trajectory["action_count"])
    obs_dim = int(trajectory["obs_dim"])
    rewards = [float(s["reward"]) for s in steps]
    values = [float(s["value_target"]) for s in steps]

    obs: list[list[float]] = []
    actions: list[list[int]] = []
    target_values: list[list[float]] = []
    target_rewards: list[list[float]] = []
    target_policies: list[list[list[float]]] = []

    for pos in indices:
        # Initial observation at row[pos].
        if pos >= len(steps):
            raise IndexError(
                f"trajectory has {len(steps)} steps; cannot start a batch at {pos}"
            )
        obs.append(list(steps[pos]["obs"]))

        # Actions[k] for k in 0..num_unroll_steps-1
        row_actions: list[int] = []
        # Target rewards[k]
        row_rewards: list[float] = []
        # Target policies[k] for k in 0..num_unroll_steps (one extra row
        # for the initial-inference policy target)
        row_policies: list[list[float]] = []
        # Target values[k] for k in 0..num_unroll_steps (one extra row)
        row_values: list[float] = []

        # First policy + value target — at the starting position.
        row_policies.append(list(steps[pos]["policy_target"]))
        row_values.append(
            compute_n_step_return(
                rewards=rewards,
                values=values,
                position=pos,
                td_steps=td_steps,
                discount=discount,
            )
        )

        for k in range(num_unroll_steps):
            step_index = pos + k
            if step_index < len(steps):
                row_actions.append(int(steps[step_index]["action_id"]))
                row_rewards.append(rewards[step_index])
            else:
                # Padding past end-of-trajectory: action 0, reward 0.
                row_actions.append(0)
                row_rewards.append(0.0)

            future_index = pos + k + 1
            if future_index < len(steps):
                row_policies.append(list(steps[future_index]["policy_target"]))
                row_values.append(
                    compute_n_step_return(
                        rewards=rewards,
                        values=values,
                        position=future_index,
                        td_steps=td_steps,
                        discount=discount,
                    )
                )
            else:
                # Past the trajectory; uniform policy + zero value
                # target keep the loss bounded without leaking into the
                # gradient.
                row_policies.append([1.0 / action_dim] * action_dim)
                row_values.append(0.0)

        actions.append(row_actions)
        target_values.append(row_values)
        target_rewards.append(row_rewards)
        target_policies.append(row_policies)

    return {
        "observations": obs,
        "actions": actions,
        "target_values": target_values,
        "target_rewards": target_rewards,
        "target_policies": target_policies,
        "_meta": {
            "obs_dim": obs_dim,
            "action_count": action_dim,
            "rows": len(indices),
        },
    }


class MuzeroMcTrainer:
    """Drives MuZero training against a directory of TrajectoryV2
    files written by ``forge-mc-runner``.

    Lifecycle::

        trainer = MuzeroMcTrainer(model, reader, config)
        outcome = trainer.train()

    Each gradient step:

    1. Pulls a fresh trajectory from the reader (round-robin over the
       available files).
    2. Samples ``batch_size`` random step positions from the chosen
       trajectory.
    3. Builds a batch via :func:`build_batch_from_trajectory`.
    4. Calls :func:`forge.training._muzero_step.train_with_gradients`.

    Every ``export_every_n_iters`` steps the trainer writes a fresh
    ONNX bundle + bumped manifest into ``output_dir``. The Rust
    runner's ``HotReloadWatcher`` picks it up on its next poll.
    """

    def __init__(
        self,
        model: MuZeroWorldModel,
        reader: TrajectoryReader,
        config: MuZeroMcTrainerConfig,
    ) -> None:
        import torch

        self._model = model
        self._reader = reader
        self._config = config
        self._rng = random.Random(config.seed if config.seed is not None else None)
        # Resolve `auto` → `cuda` / `cpu` once at construction; subsequent
        # `train_step` calls operate against the locked device. The
        # config's validated literal (`cpu` / `cuda` / `auto`) means
        # `_resolve_device` never sees an unknown value.
        self._device = _resolve_device(config.device)
        self._model.to(self._device)
        self._optimizer = torch.optim.Adam(
            model.all_parameters(),
            lr=model.config.learning_rate,
            weight_decay=model.config.weight_decay,
        )
        self._iter: int = 0
        self._last_manifest_version: int = 0
        # If the configured manifest already exists on disk, pick up
        # its version so the next export bumps from there.
        if config.manifest_path is not None and Path(config.manifest_path).exists():
            try:
                m = load_manifest(config.manifest_path)
                self._last_manifest_version = int(m.version)
                logger.info(
                    "resuming from manifest %s @ version %d",
                    config.manifest_path,
                    self._last_manifest_version,
                )
            except (ManifestError, OSError, ValueError, json.JSONDecodeError) as e:
                # Corrupt manifest on disk shouldn't block training —
                # log and start versioning at 1. Narrowed from a bare
                # `except Exception` so an unexpected error type
                # (e.g. a bug in `load_manifest`) propagates rather
                # than getting swallowed as "unreadable manifest".
                logger.warning("ignoring unreadable manifest %s: %s", config.manifest_path, e)

    @property
    def iter(self) -> int:
        """Number of gradient steps completed so far."""
        return self._iter

    @property
    def last_manifest_version(self) -> int:
        """Last manifest version the trainer exported (or loaded
        from disk at init time)."""
        return self._last_manifest_version

    @property
    def device(self) -> Any:
        """Resolved ``torch.device`` the model + optimiser run on. For
        ``config.device == "auto"`` this is whichever CUDA / CPU
        device :func:`_resolve_device` picked at construction."""
        return self._device

    def train_step(self) -> dict[str, float]:
        """One gradient step. Returns the same dict shape the
        existing ``MuZeroTrainer.train_step`` produces (``loss``,
        ``policy_loss``, ``value_loss``, ``reward_loss``, ``l2_reg``).
        """
        from forge.training._muzero_step import (
            MuZeroStepConfig,
            train_with_gradients,
        )

        trajectory = self._sample_trajectory()
        steps = trajectory["steps"]
        if not steps:
            # Empty trajectory — emit a zero-loss snapshot so the
            # caller can decide to bail.
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "reward_loss": 0.0,
                "l2_reg": 0.0,
            }

        # Sample batch_size starting positions (with replacement so
        # short trajectories still produce full batches).
        indices = [self._rng.randrange(len(steps)) for _ in range(self._config.batch_size)]
        batch = build_batch_from_trajectory(
            trajectory,
            indices=indices,
            num_unroll_steps=self._model.config.num_unroll_steps,
            td_steps=self._model.config.td_steps,
            discount=self._model.config.discount,
        )
        # Strip our own metadata before handing the batch off.
        batch.pop("_meta", None)

        metrics = train_with_gradients(
            model=self._model,
            optimizer=self._optimizer,
            batch=batch,
            step_config=MuZeroStepConfig(
                max_grad_norm=self._config.max_grad_norm,
                gradient_scale=self._config.gradient_scale,
            ),
        )
        self._iter += 1
        return metrics.to_dict()

    def train(self) -> dict[str, Any]:
        """Run ``train_iters`` gradient steps, exporting on the
        configured cadence. Returns a summary dict.
        """
        if self._config.train_iters <= 0:
            return {"iters_completed": 0, "exports": 0}

        exports = 0
        last_metrics: dict[str, float] = {}
        for _ in range(self._config.train_iters):
            last_metrics = self.train_step()
            if (
                self._config.log_every_n_iters > 0
                and self._iter % self._config.log_every_n_iters == 0
            ):
                logger.info(
                    "iter %d loss=%.4f policy=%.4f value=%.4f reward=%.4f",
                    self._iter,
                    last_metrics["loss"],
                    last_metrics["policy_loss"],
                    last_metrics["value_loss"],
                    last_metrics["reward_loss"],
                )
            if (
                self._config.export_every_n_iters > 0
                and self._iter % self._config.export_every_n_iters == 0
            ):
                self._export_bundle()
                exports += 1

        # Always export at the end so the final state is reachable.
        if self._config.export_every_n_iters > 0 and (
            exports == 0 or self._iter % self._config.export_every_n_iters != 0
        ):
            self._export_bundle()
            exports += 1

        return {
            "iters_completed": self._iter,
            "exports": exports,
            "last_metrics": last_metrics,
            "last_manifest_version": self._last_manifest_version,
        }

    def _sample_trajectory(self) -> dict[str, Any]:
        """Pick the next trajectory file in round-robin order. Falls
        back to a fresh `iter_episodes` walk when the reader's list
        is exhausted.
        """
        paths = self._reader.episode_paths()
        if not paths:
            raise RuntimeError(
                f"no trajectory files in {self._reader.directory} matching "
                f"{getattr(self._reader, '_glob', 'ep-*.json')}"
            )
        # Round-robin via _iter as the index.
        from forge.training.muzero_mc.replay import load_trajectory

        return load_trajectory(paths[self._iter % len(paths)])

    def _export_bundle(self) -> None:
        """Export the current model weights as ONNX + write a bumped
        manifest atomically.
        """
        from forge.models.muzero_export import MuZeroExporter

        out_dir = Path(self._config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        exporter = MuZeroExporter(self._model)
        exporter.export_onnx(out_dir)

        self._last_manifest_version += 1
        manifest = build_manifest(
            version=self._last_manifest_version,
            schema_id=self._config.schema_id,
            files_dir=out_dir,
            representation_filename=DEFAULT_BUNDLE_FILENAMES["representation"],
            dynamics_filename=DEFAULT_BUNDLE_FILENAMES["dynamics"],
            prediction_filename=DEFAULT_BUNDLE_FILENAMES["prediction"],
        )
        manifest_path = self._config.manifest_path or (out_dir / "model_manifest.json")
        save_manifest(manifest, manifest_path)
        logger.info(
            "exported bundle iter=%d manifest_version=%d schema_id=%s",
            self._iter,
            self._last_manifest_version,
            self._config.schema_id,
        )
