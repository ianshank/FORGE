"""MuZero trainer: self-play data collection and network training.

Orchestrates the MuZero training loop:
1. Self-play: generate game histories using the current model + MCTS
2. Store games in the replay buffer
3. Sample batches and train the networks
4. Repeat

Usage::

    from forge.models.muzero_config import MuZeroConfig
    from forge.models.muzero_world_model import MuZeroWorldModel
    from forge.training.muzero_trainer import MuZeroTrainer, MuZeroTrainerConfig

    model = MuZeroWorldModel(MuZeroConfig(obs_dim=920, action_dim=75))
    trainer = MuZeroTrainer(MuZeroTrainerConfig(), model)
    trainer.train(env_factory=lambda: ForgeEnv(config), num_iterations=100)
"""

from __future__ import annotations

__all__ = ["MuZeroTrainer", "MuZeroTrainerConfig"]

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from forge.agents.muzero_mcts import MuZeroMCTS, MuZeroMCTSConfig
from forge.training.muzero_buffer import (
    GameHistory,
    MuZeroBufferConfig,
    MuZeroReplayBuffer,
)

if TYPE_CHECKING:
    from forge.models.muzero_world_model import MuZeroWorldModel

logger = logging.getLogger(__name__)

DEFAULT_TRAINING_STEPS_PER_ITER: int = 100
DEFAULT_SELF_PLAY_GAMES_PER_ITER: int = 10
DEFAULT_BATCH_SIZE: int = 256
DEFAULT_CHECKPOINT_INTERVAL: int = 50
DEFAULT_LOG_INTERVAL: int = 10
DEFAULT_TEMPERATURE_SCHEDULE_STEPS: int = 500
DEFAULT_TEMPERATURE_INIT: float = 1.0
DEFAULT_TEMPERATURE_FINAL: float = 0.25


@dataclass
class MuZeroTrainerConfig:
    """Configuration for the MuZero training loop.

    Attributes:
        training_steps_per_iter: Number of gradient steps per training iteration.
        self_play_games_per_iter: Games to generate per iteration via self-play.
        batch_size: Training batch size.
        checkpoint_interval: Save model every N iterations.
        checkpoint_dir: Directory for checkpoints.
        log_interval: Log metrics every N training steps.
        temperature_init: Initial action selection temperature.
        temperature_final: Final temperature after schedule.
        temperature_schedule_steps: Steps over which to anneal temperature.
        max_episode_steps: Maximum steps per self-play episode.
        max_grad_norm: Maximum gradient norm for gradient clipping.
        gradient_scale: Scale factor for latent gradient in dynamics unroll.
        seed: Random seed.
        buffer_config: Replay buffer configuration.
        mcts_config: MCTS search configuration.
    """

    training_steps_per_iter: int = DEFAULT_TRAINING_STEPS_PER_ITER
    self_play_games_per_iter: int = DEFAULT_SELF_PLAY_GAMES_PER_ITER
    batch_size: int = DEFAULT_BATCH_SIZE
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL
    checkpoint_dir: str = "checkpoints/muzero"
    log_interval: int = DEFAULT_LOG_INTERVAL
    temperature_init: float = DEFAULT_TEMPERATURE_INIT
    temperature_final: float = DEFAULT_TEMPERATURE_FINAL
    temperature_schedule_steps: int = DEFAULT_TEMPERATURE_SCHEDULE_STEPS
    max_episode_steps: int = 500
    max_grad_norm: float = 1.0
    gradient_scale: float = 0.5
    seed: int = 42
    buffer_config: MuZeroBufferConfig = field(default_factory=MuZeroBufferConfig)
    mcts_config: MuZeroMCTSConfig = field(default_factory=MuZeroMCTSConfig)


class MuZeroTrainer:
    """MuZero self-play and training orchestrator.

    Manages the full MuZero training pipeline:

    1. **Self-play**: Uses the current model with MCTS to generate game
       histories. Search statistics (visit counts, values) become
       training targets.

    2. **Training**: Samples batches from the replay buffer and performs
       gradient descent on the combined policy + value + reward loss.

    3. **Checkpointing**: Periodically saves model weights for evaluation.

    Args:
        config: Trainer configuration.
        model: The MuZero world model to train.
    """

    def __init__(
        self,
        config: MuZeroTrainerConfig,
        model: MuZeroWorldModel,
    ) -> None:
        import torch

        self._config = config
        self._model = model
        self._buffer = MuZeroReplayBuffer(config.buffer_config)
        self._mcts = MuZeroMCTS(model, config.mcts_config)
        self._optimizer = torch.optim.Adam(
            model.all_parameters(),
            lr=model.config.learning_rate,
            weight_decay=model.config.weight_decay,
        )
        self._total_games: int = 0
        self._total_train_steps: int = 0
        self._rng = np.random.default_rng(config.seed)
        logger.info(
            "MuZeroTrainer: train_steps=%d/iter, self_play=%d/iter, batch=%d",
            config.training_steps_per_iter,
            config.self_play_games_per_iter,
            config.batch_size,
        )

    @property
    def total_games(self) -> int:
        """Total number of self-play games generated."""
        return self._total_games

    @property
    def total_train_steps(self) -> int:
        """Total number of gradient update steps."""
        return self._total_train_steps

    def current_temperature(self) -> float:
        """Compute the current action selection temperature based on schedule."""
        c = self._config
        if c.temperature_schedule_steps <= 0:
            return c.temperature_final
        progress = min(1.0, self._total_games / c.temperature_schedule_steps)
        return c.temperature_init + progress * (c.temperature_final - c.temperature_init)

    def self_play(self, env: Any) -> GameHistory:
        """Generate one game history via self-play.

        Args:
            env: A Gymnasium-compatible FORGE environment with
                ``reset()`` and ``step()`` methods.

        Returns:
            A complete :class:`GameHistory`.
        """
        history = GameHistory()
        obs, _info = env.reset()
        obs = np.asarray(obs, dtype=np.float32)
        history.observations.append(obs)

        temperature = self.current_temperature()

        for _ in range(self._config.max_episode_steps):
            action_id, info = self._mcts.search(obs, temperature=temperature)

            history.actions.append(action_id)
            history.root_values.append(info["root_value"])
            history.child_visits.append(info["visit_counts"])

            obs, reward, terminated, truncated, _info = env.step(action_id)
            obs = np.asarray(obs, dtype=np.float32)

            history.observations.append(obs)
            history.rewards.append(float(reward))
            history.dones.append(bool(terminated or truncated))

            if terminated or truncated:
                break

        self._total_games += 1
        logger.debug(
            "Self-play game %d: length=%d, total_reward=%.2f",
            self._total_games,
            history.length,
            sum(history.rewards),
        )
        return history

    def train_step(self) -> dict[str, float]:
        """Perform a single training step.

        Samples a batch from the replay buffer and updates the networks.
        The forward pass, backward pass, and optimizer step all happen
        inside :meth:`_train_with_gradients` to avoid redundant computation.

        Returns:
            Training metrics dictionary.

        Raises:
            RuntimeError: If the replay buffer is empty.
        """
        if self._buffer.num_games == 0:
            msg = "Cannot train: replay buffer is empty. Run self-play first."
            raise RuntimeError(msg)

        batch = self._buffer.sample_batch(
            batch_size=self._config.batch_size,
            num_unroll_steps=self._model.config.num_unroll_steps,
            td_steps=self._model.config.td_steps,
            discount=self._model.config.discount,
        )

        metrics = self._train_with_gradients(batch)

        self._total_train_steps += 1

        if self._total_train_steps % self._config.log_interval == 0:
            logger.info(
                "Train step %d: loss=%.4f, policy=%.4f, value=%.4f, reward=%.4f",
                self._total_train_steps,
                metrics.get("loss", 0.0),
                metrics.get("policy_loss", 0.0),
                metrics.get("value_loss", 0.0),
                metrics.get("reward_loss", 0.0),
            )

        return metrics

    def _train_with_gradients(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """Run forward + backward + optimizer step with gradient tracking.

        Args:
            batch: Training batch dictionary.

        Returns:
            Training metrics dictionary.
        """
        import torch
        from torch import nn

        from forge.models.muzero_networks import scalar_to_support

        c = self._model.config
        tc = self._config
        device = torch.device(c.device)

        obs = torch.as_tensor(batch["observations"], dtype=torch.float32, device=device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=device)
        target_values = torch.as_tensor(batch["target_values"], dtype=torch.float32, device=device)
        target_rewards = torch.as_tensor(
            batch["target_rewards"], dtype=torch.float32, device=device
        )
        target_policies = torch.as_tensor(
            batch["target_policies"], dtype=torch.float32, device=device
        )

        self._optimizer.zero_grad()

        # Initial inference
        latent = self._model.representation.forward(obs)
        policy_logits, value_logits = self._model.prediction.forward(latent)

        # Initial losses
        target_val_dist = scalar_to_support(target_values[:, 0], c.value_support_size)
        value_loss = nn.functional.cross_entropy(value_logits, target_val_dist)
        log_probs = torch.log_softmax(policy_logits, dim=-1)
        policy_loss = -torch.mean(torch.sum(target_policies[:, 0] * log_probs, dim=-1))
        reward_loss = torch.tensor(0.0, device=device)

        # Unroll
        gs = tc.gradient_scale
        num_steps = min(c.num_unroll_steps, actions.shape[1])
        for k in range(num_steps):
            action_oh = nn.functional.one_hot(actions[:, k], num_classes=c.action_dim).float()

            # Scale gradient for dynamics (balance initial vs unrolled)
            latent_scaled = latent.detach() * (1.0 - gs) + latent * gs
            latent, rew_logits = self._model.dynamics.forward(latent_scaled, action_oh)
            pol_logits, val_logits = self._model.prediction.forward(latent)

            target_rew_dist = scalar_to_support(target_rewards[:, k], c.reward_support_size)
            reward_loss = reward_loss + nn.functional.cross_entropy(rew_logits, target_rew_dist)

            target_val_dist = scalar_to_support(target_values[:, k + 1], c.value_support_size)
            value_loss = value_loss + nn.functional.cross_entropy(val_logits, target_val_dist)

            log_p = torch.log_softmax(pol_logits, dim=-1)
            policy_loss = policy_loss + (
                -torch.mean(torch.sum(target_policies[:, k + 1] * log_p, dim=-1))
            )

        scale = 1.0 / (num_steps + 1)
        total_loss = scale * (value_loss + policy_loss + reward_loss)

        # L2 regularization
        l2_reg = sum(p.pow(2).sum() for p in self._model.all_parameters())
        total_loss = total_loss + c.weight_decay * l2_reg

        total_loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self._model.all_parameters(), max_norm=tc.max_grad_norm)
        self._optimizer.step()

        return {
            "loss": float(total_loss.item()),
            "policy_loss": float(policy_loss.item() * scale),
            "value_loss": float(value_loss.item() * scale),
            "reward_loss": float(reward_loss.item() * scale),
            "l2_reg": float(l2_reg.item()),
        }

    def train(
        self,
        env_factory: Callable[[], Any],
        num_iterations: int,
    ) -> dict[str, list[float]]:
        """Run the full MuZero training loop.

        Args:
            env_factory: Callable that creates a Gymnasium-compatible env.
            num_iterations: Number of training iterations.

        Returns:
            Dictionary of metric histories (lists of floats).
        """
        metric_history: dict[str, list[float]] = {
            "loss": [],
            "policy_loss": [],
            "value_loss": [],
            "reward_loss": [],
            "game_length": [],
            "game_reward": [],
        }

        for iteration in range(num_iterations):
            iter_start = time.time()

            # Self-play phase
            for _ in range(self._config.self_play_games_per_iter):
                env = env_factory()
                try:
                    history = self.self_play(env)
                    self._buffer.save_game(history)
                    metric_history["game_length"].append(float(history.length))
                    metric_history["game_reward"].append(float(sum(history.rewards)))
                finally:
                    env.close()

            # Training phase
            if self._buffer.num_games > 0:
                for _ in range(self._config.training_steps_per_iter):
                    metrics = self.train_step()
                    for key in ("loss", "policy_loss", "value_loss", "reward_loss"):
                        metric_history[key].append(metrics.get(key, 0.0))

            # Checkpointing
            if (
                self._config.checkpoint_interval > 0
                and (iteration + 1) % self._config.checkpoint_interval == 0
            ):
                self._save_checkpoint(iteration + 1)

            elapsed = time.time() - iter_start
            logger.info(
                "Iteration %d/%d: games=%d, train_steps=%d, buffer=%d games, elapsed=%.1fs",
                iteration + 1,
                num_iterations,
                self._total_games,
                self._total_train_steps,
                self._buffer.num_games,
                elapsed,
            )

        return metric_history

    def _save_checkpoint(self, iteration: int) -> None:
        """Save a training checkpoint.

        Args:
            iteration: Current iteration number (used in filename).
        """
        checkpoint_dir = Path(self._config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"muzero_iter_{iteration:06d}.pt"
        self._model.save(str(path))
        logger.info("Checkpoint saved: %s", path)
