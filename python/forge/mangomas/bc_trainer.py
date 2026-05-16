"""Behavioural-cloning trainer that distils the LM Studio teacher.

Scope: pure BC only — train a student policy (linear softmax classifier
or, optionally, an ``ActorCriticNetwork`` actor head) on
``(observation, teacher_action_id)`` pairs with an optional KL term
against ``top_k_probs``. **No DAgger, no DPO, no preference learning.**
Future preference-style work belongs in a separate module.

Two execution paths are provided:

* **NumPy path** (always available): trains a softmax linear classifier
  with mini-batch SGD. Exports an ``.npz`` matching the conventions of
  ``BDIPreTrainer.export_weights``.
* **Torch path** (optional): when an ``ActorCriticNetwork`` instance is
  provided, fine-tunes its actor head in-place with cross-entropy on
  teacher actions and optional KL on top-k probabilities. The import of
  ``torch`` is guarded so the module loads on systems without it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

DEFAULT_BC_LEARNING_RATE: float = 3e-4
DEFAULT_BC_NUM_EPOCHS: int = 30
DEFAULT_BC_BATCH_SIZE: int = 64
DEFAULT_BC_KL_WEIGHT: float = 1.0
DEFAULT_BC_VALUE_LOSS_WEIGHT: float = 0.5
DEFAULT_BC_SEED: int = 42


@dataclass
class BCTrainerConfig:
    """Configuration for the BC trainer."""

    learning_rate: float = DEFAULT_BC_LEARNING_RATE
    num_epochs: int = DEFAULT_BC_NUM_EPOCHS
    batch_size: int = DEFAULT_BC_BATCH_SIZE
    kl_weight: float = DEFAULT_BC_KL_WEIGHT
    value_loss_weight: float = DEFAULT_BC_VALUE_LOSS_WEIGHT
    seed: int = DEFAULT_BC_SEED
    num_actions: int = 0  # 0 → inferred from teacher_action_ids.max()+1


@dataclass
class BCDataset:
    """Per-sample training inputs for the BC trainer."""

    observations: NDArray[np.float32]
    teacher_action_ids: NDArray[np.int64]
    teacher_top_k_probs: NDArray[np.float32] | None = None
    teacher_value_hats: NDArray[np.float32] | None = None

    @property
    def num_samples(self) -> int:
        return int(self.observations.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.observations.shape[1]) if self.observations.ndim == 2 else 0


@dataclass
class BCTrainResult:
    """Output metrics from a BC run."""

    final_loss: float = 0.0
    final_top1_accuracy: float = 0.0
    epochs_run: int = 0
    loss_history: list[float] = field(default_factory=list)
    accuracy_history: list[float] = field(default_factory=list)


class BCTrainer:
    """Behavioural-cloning trainer (numpy + optional torch path)."""

    def __init__(self, config: BCTrainerConfig | None = None) -> None:
        self.config = config or BCTrainerConfig()
        self._weights: dict[str, np.ndarray] | None = None
        logger.info(
            "BCTrainer initialized lr=%g epochs=%d batch_size=%d kl_weight=%g",
            self.config.learning_rate,
            self.config.num_epochs,
            self.config.batch_size,
            self.config.kl_weight,
        )

    @staticmethod
    def build_dataset(
        observations: list[NDArray[np.float32]],
        teacher_action_ids: list[NDArray[np.int64]],
        *,
        top_k_probs: list[list[list[dict[str, Any]]]] | None = None,
        value_hats: list[list[float]] | None = None,
        num_actions: int,
    ) -> BCDataset:
        """Flatten per-episode collected data into a single BC dataset.

        ``top_k_probs`` is converted into a dense ``(N, num_actions)`` float
        matrix; entries not in the teacher's top-k are filled with zeros.
        """
        if not observations:
            return BCDataset(
                observations=np.zeros((0, 0), dtype=np.float32),
                teacher_action_ids=np.zeros((0,), dtype=np.int64),
            )
        flat_obs: list[NDArray[np.float32]] = []
        flat_actions: list[int] = []
        flat_topk_rows: list[NDArray[np.float32]] = []
        flat_values: list[float] = []
        for ep_idx, (ep_obs, ep_actions) in enumerate(
            zip(observations, teacher_action_ids)
        ):
            steps = min(int(ep_obs.shape[0]), int(ep_actions.shape[0]))
            # `top_k_probs` / `value_hats` are Optional[Sequence[...]]; the
            # truthiness check on the same expression narrows them to the
            # non-None branch so the indexing is type-safe.
            ep_topk = top_k_probs[ep_idx] if top_k_probs else None
            ep_values = value_hats[ep_idx] if value_hats else None
            for t in range(steps):
                flat_obs.append(ep_obs[t])
                flat_actions.append(int(ep_actions[t]))
                if ep_topk is not None:
                    row = np.zeros(num_actions, dtype=np.float32)
                    for entry in ep_topk[t] if t < len(ep_topk) else []:
                        a = int(entry.get("action_id", -1))
                        if 0 <= a < num_actions:
                            row[a] = float(entry.get("prob", 0.0))
                    s = float(row.sum())
                    if s > 0.0:
                        row /= s
                    flat_topk_rows.append(row)
                if ep_values is not None and t < len(ep_values):
                    flat_values.append(float(ep_values[t]))

        obs_array = np.asarray(flat_obs, dtype=np.float32)
        action_array = np.asarray(flat_actions, dtype=np.int64)
        topk_array: NDArray[np.float32] | None = (
            np.asarray(flat_topk_rows, dtype=np.float32) if flat_topk_rows else None
        )
        value_array: NDArray[np.float32] | None = (
            np.asarray(flat_values, dtype=np.float32) if flat_values else None
        )
        return BCDataset(
            observations=obs_array,
            teacher_action_ids=action_array,
            teacher_top_k_probs=topk_array,
            teacher_value_hats=value_array,
        )

    def train(
        self,
        dataset: BCDataset,
        *,
        actor_critic: Any | None = None,
    ) -> BCTrainResult:
        """Train the BC student.

        If ``actor_critic`` is supplied it is interpreted as an
        ``ActorCriticNetwork`` and trained via the torch path. Otherwise
        the numpy path trains a linear softmax classifier.

        Empty datasets are a no-op: the method logs a warning and returns
        an empty :class:`BCTrainResult` (``epochs_run=0``) instead of
        running zero-sample epochs that would silently report
        ``loss=0.0, accuracy=0.0`` and look like a successful run.
        """
        if dataset.num_samples == 0:
            logger.warning(
                "BCTrainer.train called with empty dataset; skipping training "
                "(epochs_run=0). Check upstream collection produced teacher actions."
            )
            return BCTrainResult(epochs_run=0)
        if actor_critic is not None:
            return self._train_torch(dataset, actor_critic)
        return self._train_numpy(dataset)

    def _resolve_num_actions(self, dataset: BCDataset) -> int:
        """Resolve the actor's output dimensionality.

        Priority (highest first):

        1. Explicit ``BCTrainerConfig.num_actions`` when set — the caller has
           authoritative knowledge of the action space.
        2. The dataset's ``teacher_top_k_probs`` column count — sized by
           :meth:`build_dataset` from the per-scenario ``num_actions``, so it
           reflects the true env action space even when the observed action
           ids are a strict subset.
        3. ``teacher_action_ids.max() + 1`` — last-resort fallback for
           datasets with no top-k matrix (e.g. greedy teachers).

        Previously this method skipped (2), which produced a shape mismatch
        in the KL term when the actor was sized smaller than the top-k
        matrix (regression covered by
        ``tests/python/test_pipeline_bc_stage.py::
        test_bc_stage_uses_action_space_sizes_when_provided``).
        """
        if self.config.num_actions > 0:
            return self.config.num_actions
        topk = dataset.teacher_top_k_probs
        if topk is not None and topk.size > 0 and topk.shape[1] > 0:
            return int(topk.shape[1])
        if dataset.num_samples == 0:
            return 1
        return int(dataset.teacher_action_ids.max()) + 1

    def _train_numpy(self, dataset: BCDataset) -> BCTrainResult:
        rng = np.random.default_rng(self.config.seed)
        n_actions = self._resolve_num_actions(dataset)
        state_dim = dataset.state_dim or 1
        scale = np.sqrt(6.0 / (state_dim + n_actions))
        w = rng.uniform(-scale, scale, (n_actions, state_dim)).astype(np.float32)
        b = np.zeros(n_actions, dtype=np.float32)

        loss_history: list[float] = []
        acc_history: list[float] = []

        for epoch in range(self.config.num_epochs):
            indices = rng.permutation(dataset.num_samples)
            epoch_loss = 0.0
            epoch_correct = 0
            for start in range(0, dataset.num_samples, self.config.batch_size):
                end = min(start + self.config.batch_size, dataset.num_samples)
                idx = indices[start:end]
                x = dataset.observations[idx]
                y = dataset.teacher_action_ids[idx]

                logits = x @ w.T + b
                logits_max = logits.max(axis=1, keepdims=True)
                exp_logits = np.exp(logits - logits_max)
                probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                ce = -np.log(probs[np.arange(len(y)), y] + 1e-8)
                loss = float(ce.mean())

                if (
                    dataset.teacher_top_k_probs is not None
                    and self.config.kl_weight > 0.0
                ):
                    target = dataset.teacher_top_k_probs[idx]
                    eps = 1e-8
                    kl = (
                        target
                        * (np.log(target + eps) - np.log(probs + eps))
                    ).sum(axis=1)
                    loss += float(self.config.kl_weight * kl.mean())

                # Cross-entropy gradient on the linear head only.
                grad = probs.copy()
                grad[np.arange(len(y)), y] -= 1.0
                grad /= len(y)
                w -= self.config.learning_rate * (grad.T @ x)
                b -= self.config.learning_rate * grad.sum(axis=0)

                epoch_loss += loss * len(y)
                epoch_correct += int((probs.argmax(axis=1) == y).sum())

            avg_loss = epoch_loss / max(1, dataset.num_samples)
            avg_acc = epoch_correct / max(1, dataset.num_samples)
            loss_history.append(avg_loss)
            acc_history.append(avg_acc)
            if epoch == 0 or (epoch + 1) % 10 == 0:
                logger.info(
                    "BC numpy epoch=%d loss=%.4f top1_acc=%.4f",
                    epoch,
                    avg_loss,
                    avg_acc,
                )

        self._weights = {"actor_w": w, "actor_b": b}
        return BCTrainResult(
            final_loss=loss_history[-1] if loss_history else 0.0,
            final_top1_accuracy=acc_history[-1] if acc_history else 0.0,
            epochs_run=self.config.num_epochs,
            loss_history=loss_history,
            accuracy_history=acc_history,
        )

    def _train_torch(
        self, dataset: BCDataset, actor_critic: Any
    ) -> BCTrainResult:
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as exc:
            msg = "torch is required for the BC torch path"
            raise ImportError(msg) from exc

        device = next(actor_critic.parameters()).device
        x = torch.as_tensor(dataset.observations, dtype=torch.float32, device=device)
        y = torch.as_tensor(dataset.teacher_action_ids, dtype=torch.long, device=device)
        topk = (
            torch.as_tensor(
                dataset.teacher_top_k_probs, dtype=torch.float32, device=device
            )
            if dataset.teacher_top_k_probs is not None
            else None
        )
        optimizer = torch.optim.Adam(
            actor_critic.parameters(), lr=self.config.learning_rate
        )

        loss_history: list[float] = []
        acc_history: list[float] = []
        for epoch in range(self.config.num_epochs):
            perm = torch.randperm(x.shape[0], device=device)
            epoch_loss = 0.0
            epoch_correct = 0
            for start in range(0, x.shape[0], self.config.batch_size):
                end = min(start + self.config.batch_size, x.shape[0])
                idx = perm[start:end]
                xb = x[idx]
                yb = y[idx]
                logits, values = actor_critic(xb)
                ce = F.cross_entropy(logits, yb)
                loss = ce
                if topk is not None and self.config.kl_weight > 0.0:
                    log_probs = F.log_softmax(logits, dim=-1)
                    kl = F.kl_div(log_probs, topk[idx], reduction="batchmean")
                    loss = loss + self.config.kl_weight * kl
                if (
                    dataset.teacher_value_hats is not None
                    and self.config.value_loss_weight > 0.0
                ):
                    vh = torch.as_tensor(
                        dataset.teacher_value_hats[idx.cpu().numpy()],
                        dtype=torch.float32,
                        device=device,
                    )
                    loss = loss + self.config.value_loss_weight * F.mse_loss(
                        values.squeeze(-1), vh
                    )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu()) * (end - start)
                epoch_correct += int((logits.argmax(dim=-1) == yb).sum().item())
            avg_loss = epoch_loss / max(1, x.shape[0])
            avg_acc = epoch_correct / max(1, x.shape[0])
            loss_history.append(avg_loss)
            acc_history.append(avg_acc)
            if epoch == 0 or (epoch + 1) % 10 == 0:
                logger.info(
                    "BC torch epoch=%d loss=%.4f top1_acc=%.4f",
                    epoch,
                    avg_loss,
                    avg_acc,
                )
        self._weights = None  # actor_critic owns the weights in this path
        return BCTrainResult(
            final_loss=loss_history[-1] if loss_history else 0.0,
            final_top1_accuracy=acc_history[-1] if acc_history else 0.0,
            epochs_run=self.config.num_epochs,
            loss_history=loss_history,
            accuracy_history=acc_history,
        )

    def export_weights(self, path: str | Path) -> None:
        """Export the numpy-path BC weights to an ``.npz`` file."""
        if self._weights is None:
            msg = "BC trainer has no exportable weights (use train() with no actor_critic)"
            raise RuntimeError(msg)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(path), **self._weights)
        logger.info("BC weights exported to %s", path)
