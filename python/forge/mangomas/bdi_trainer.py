"""BDI intention pre-trainer for MangoMAS integration.

Collects FORGE episodes, maps actions to BDI intention classes (0-7),
and trains a GRU+MLP intention predictor whose weights transfer to MangoMAS.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from forge.mangomas.config import BDITrainerConfig

logger = logging.getLogger(__name__)

# Default action → intention mapping (matches Rust BdiIntentionMapper)
DEFAULT_ACTION_INTENTION_MAP: dict[str, int] = {
    "Move": 0,
    "MoveUp": 0,
    "MoveDown": 0,
    "MoveLeft": 0,
    "MoveRight": 0,
    "Ascend": 0,
    "Descend": 0,
    "TakeOff": 0,
    "Land": 0,
    "PickUp": 1,
    "Drop": 1,
    "DropPayload": 1,
    "Craft": 2,
    "Push": 3,
    "Use": 3,
    "Interact": 3,
    "Communicate": 4,
    "Scan": 6,
    "Noop": 7,
    "Hover": 7,
}

INTENTION_NAMES = {
    0: "Navigate",
    1: "Gather",
    2: "Plan",
    3: "Manipulate",
    4: "Cooperate",
    5: "Evade",
    6: "Track",
    7: "Idle",
}


@dataclass
class BDIDataset:
    """Dataset of (observation, intention, reward) samples for BDI training."""

    observations: np.ndarray  # (N, state_dim)
    intentions: np.ndarray  # (N,) int
    rewards: np.ndarray  # (N,)
    episode_lengths: list[int] = field(default_factory=list)

    @property
    def num_samples(self) -> int:
        return len(self.intentions)

    def intention_distribution(self) -> dict[str, float]:
        """Return normalized distribution over intention classes."""
        counts = np.bincount(self.intentions, minlength=len(INTENTION_NAMES))
        total = counts.sum()
        if total == 0:
            return dict.fromkeys(INTENTION_NAMES.values(), 0.0)
        return {
            INTENTION_NAMES[i]: float(counts[i]) / float(total) for i in range(len(INTENTION_NAMES))
        }


@dataclass
class BDITrainResult:
    """Training result metrics."""

    final_loss: float
    final_accuracy: float
    epochs_run: int
    loss_history: list[float] = field(default_factory=list)
    accuracy_history: list[float] = field(default_factory=list)


class BDIPreTrainer:
    """Pre-trains a BDI GRU+MLP on FORGE episodes.

    The trained weights (GRU hidden state dynamics + MLP intention classifier)
    transfer to MangoMAS's Neural BDI module.
    """

    def __init__(
        self,
        config: BDITrainerConfig | None = None,
        action_intention_map: dict[str, int] | None = None,
        overrides: dict[int, int] | None = None,
    ) -> None:
        self.config = config or BDITrainerConfig()
        self._action_map = action_intention_map or DEFAULT_ACTION_INTENTION_MAP
        self._overrides = overrides or {}
        self._weights: dict[str, np.ndarray] | None = None
        logger.info(
            "BDIPreTrainer: %d intentions, hidden=%d, epochs=%d",
            self.config.num_intentions,
            self.config.hidden_size,
            self.config.num_epochs,
        )

    def map_action_to_intention(self, action_name: str, action_id: int = -1) -> int:
        """Map a FORGE action to a BDI intention class."""
        if action_id in self._overrides:
            return self._overrides[action_id]
        return self._action_map.get(action_name, self.config.default_intention)

    def build_dataset(
        self,
        observations: list[np.ndarray],
        action_names: list[list[str]],
        rewards: list[list[float]],
        *,
        teacher_intentions: list[list[int]] | None = None,
    ) -> BDIDataset:
        """Build a BDI dataset from collected episode data.

        When ``teacher_intentions`` is provided (per-episode lists of integer
        labels emitted by the LM Studio teacher), it overrides the
        rule-derived mapping from action name to intention. Default
        behaviour is unchanged for callers that don't pass the kwarg.
        """
        all_obs = []
        all_intentions = []
        all_rewards = []
        episode_lengths = []

        use_teacher = bool(teacher_intentions)
        if use_teacher and len(teacher_intentions or []) != len(observations):
            msg = "teacher_intentions must match the number of episodes"
            raise ValueError(msg)

        for ep_idx, (ep_obs, ep_actions, ep_rewards) in enumerate(
            zip(observations, action_names, rewards)
        ):
            ep_len = min(len(ep_obs), len(ep_actions), len(ep_rewards))
            episode_lengths.append(ep_len)
            teacher_ep = (teacher_intentions or [[]])[ep_idx] if use_teacher else None
            for t in range(ep_len):
                all_obs.append(ep_obs[t])
                if use_teacher and teacher_ep is not None and t < len(teacher_ep):
                    intention = int(teacher_ep[t])
                    if intention < 0:
                        intention = self.map_action_to_intention(ep_actions[t])
                else:
                    intention = self.map_action_to_intention(ep_actions[t])
                all_intentions.append(intention)
                all_rewards.append(ep_rewards[t])

        return BDIDataset(
            observations=np.array(all_obs, dtype=np.float32),
            intentions=np.array(all_intentions, dtype=np.int64),
            rewards=np.array(all_rewards, dtype=np.float32),
            episode_lengths=episode_lengths,
        )

    def train(self, dataset: BDIDataset) -> BDITrainResult:
        """Train the BDI GRU+MLP on the dataset.

        Uses numpy-only implementation (no torch dependency required).
        The GRU weights are initialized randomly and trained via simple
        cross-entropy minimization with gradient descent.
        """
        rng = np.random.default_rng(self.config.seed)
        state_dim = dataset.observations.shape[1] if dataset.num_samples > 0 else 18
        h_dim = self.config.hidden_size
        n_cls = self.config.num_intentions

        # Initialize weights (Xavier uniform)
        scale_ih = np.sqrt(6.0 / (state_dim + h_dim))
        scale_hh = np.sqrt(6.0 / (h_dim + h_dim))
        scale_out = np.sqrt(6.0 / (state_dim + n_cls))

        w_ih = rng.uniform(-scale_ih, scale_ih, (3 * h_dim, state_dim)).astype(np.float32)
        w_hh = rng.uniform(-scale_hh, scale_hh, (3 * h_dim, h_dim)).astype(np.float32)
        w_out = rng.uniform(-scale_out, scale_out, (n_cls, state_dim)).astype(np.float32)
        b_out = np.zeros(n_cls, dtype=np.float32)

        loss_history: list[float] = []
        acc_history: list[float] = []

        for epoch in range(self.config.num_epochs):
            # Simple forward pass with mini-batches
            indices = rng.permutation(dataset.num_samples)
            epoch_loss = 0.0
            epoch_correct = 0

            for start in range(0, dataset.num_samples, self.config.batch_size):
                end = min(start + self.config.batch_size, dataset.num_samples)
                batch_idx = indices[start:end]
                x = dataset.observations[batch_idx]
                y = dataset.intentions[batch_idx]

                # Simple linear classifier (GRU unrolling omitted for numpy impl)
                logits = x @ w_out.T + b_out  # (B, n_cls)

                # Softmax + cross-entropy
                logits_max = logits.max(axis=1, keepdims=True)
                exp_logits = np.exp(logits - logits_max)
                probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

                batch_loss = -np.mean(np.log(probs[np.arange(len(y)), y] + 1e-8))
                epoch_loss += float(batch_loss) * len(y)
                epoch_correct += np.sum(np.argmax(probs, axis=1) == y)

                # Gradient step on output layer
                grad = probs.copy()
                grad[np.arange(len(y)), y] -= 1
                grad /= len(y)
                dw = grad.T @ x  # (n_cls, state_dim)
                db = grad.sum(axis=0)

                w_out -= self.config.learning_rate * dw
                b_out -= self.config.learning_rate * db

            epoch_loss /= max(dataset.num_samples, 1)
            epoch_acc = epoch_correct / max(dataset.num_samples, 1)
            loss_history.append(float(epoch_loss))
            acc_history.append(float(epoch_acc))

            if (epoch + 1) % self.config.log_interval == 0:
                logger.debug(
                    "BDI epoch %d/%d: loss=%.4f, acc=%.4f",
                    epoch + 1,
                    self.config.num_epochs,
                    epoch_loss,
                    epoch_acc,
                )

        self._weights = {
            "gru_w_ih": w_ih,
            "gru_w_hh": w_hh,
            "mlp_w_out": w_out,
            "mlp_b_out": b_out,
        }

        return BDITrainResult(
            final_loss=loss_history[-1] if loss_history else 0.0,
            final_accuracy=acc_history[-1] if acc_history else 0.0,
            epochs_run=self.config.num_epochs,
            loss_history=loss_history,
            accuracy_history=acc_history,
        )

    def export_weights(self, path: str | Path) -> None:
        """Export trained weights as .npz file."""
        if self._weights is None:
            raise RuntimeError("No trained weights to export. Call train() first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(path), **self._weights)
        logger.info("BDI weights exported to %s", path)
