"""Shared MuZero gradient-step primitive.

Extracted from ``MuZeroTrainer._train_with_gradients``
(muzero_trainer.py:239-320) so both the existing
``MuZeroTrainer.train_step`` and the new ``MuzeroMcTrainer.train_step``
call the same loss / clipping / L2 / gradient-scaling logic. Diverging
copies would silently shift the training dynamics between trainers.

The function is the smallest unit that knows about MuZero loss
internals; trainer classes own the optimiser + the batch sampling and
hand them in.

No hard-coded values; every hyperparameter flows from the model's
``MuZeroConfig`` (loss-side) or the caller's gradient-step config
(clipping / scaling).
"""

from __future__ import annotations

__all__ = ["MuZeroStepConfig", "TrainStepMetrics", "train_with_gradients"]

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from forge.models.muzero_world_model import MuZeroWorldModel


@dataclass(frozen=True)
class MuZeroStepConfig:
    """Per-step optimisation knobs that are NOT part of the model's
    own ``MuZeroConfig``.

    Attributes:
        max_grad_norm: Maximum L2 norm for gradient clipping.
        gradient_scale: Scale factor for the latent gradient during the
            dynamics unroll. ``0.5`` balances initial-inference and
            unrolled-inference gradients (MuZero paper default).
    """

    max_grad_norm: float = 1.0
    gradient_scale: float = 0.5


@dataclass(frozen=True)
class TrainStepMetrics:
    """Metrics returned by :func:`train_with_gradients`. Frozen so
    callers can't mutate the snapshot accidentally.
    """

    loss: float
    policy_loss: float
    value_loss: float
    reward_loss: float
    l2_reg: float

    def to_dict(self) -> dict[str, float]:
        """Convert to a plain dict for ``ForgeLogger`` / W&B / MLflow."""
        return {
            "loss": self.loss,
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "reward_loss": self.reward_loss,
            "l2_reg": self.l2_reg,
        }


def train_with_gradients(
    model: MuZeroWorldModel,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Any],
    step_config: MuZeroStepConfig,
) -> TrainStepMetrics:
    """Run forward + MuZero loss + backward + optimiser step.

    Args:
        model: The ``MuZeroWorldModel`` being trained. Reads
            ``model.config`` for loss-side hyperparameters
            (``num_unroll_steps``, ``value_support_size``,
            ``reward_support_size``, ``action_dim``, ``weight_decay``,
            ``device``).
        optimizer: The torch optimiser the caller already attached to
            ``model.all_parameters()``. We do not construct it — the
            trainer owns its lifecycle.
        batch: Training batch dict with NumPy arrays / tensors at the
            following keys, matching
            ``MuZeroReplayBuffer.sample_batch``'s return shape:
            ``observations``, ``actions``, ``target_values``,
            ``target_rewards``, ``target_policies``.
        step_config: Per-step optimisation knobs (clipping, scaling).

    Returns:
        A :class:`TrainStepMetrics` snapshot.

    Notes:
        - This function is loss-identical to the original
          ``MuZeroTrainer._train_with_gradients``; behaviour drift
          here is a bug.
        - The torch import is local so the broader training module
          stays importable without torch installed (matches the
          ``minecraft`` optional-deps discipline).
    """
    import torch
    from torch import nn

    from forge.models.muzero_networks import scalar_to_support

    c = model.config
    # The weights' live device, not the construction-time ``config.device``:
    # a trainer that calls ``model.to("cuda")`` must get CUDA batches.
    device = model.device

    obs = torch.as_tensor(batch["observations"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=device)
    target_values = torch.as_tensor(batch["target_values"], dtype=torch.float32, device=device)
    target_rewards = torch.as_tensor(batch["target_rewards"], dtype=torch.float32, device=device)
    target_policies = torch.as_tensor(batch["target_policies"], dtype=torch.float32, device=device)

    optimizer.zero_grad()

    # Initial inference
    latent = model.representation.forward(obs)
    policy_logits, value_logits = model.prediction.forward(latent)

    # Initial losses
    target_val_dist = scalar_to_support(target_values[:, 0], c.value_support_size)
    value_loss = nn.functional.cross_entropy(value_logits, target_val_dist)
    log_probs = torch.log_softmax(policy_logits, dim=-1)
    policy_loss = -torch.mean(torch.sum(target_policies[:, 0] * log_probs, dim=-1))
    reward_loss = torch.tensor(0.0, device=device)

    # Unroll
    gs = step_config.gradient_scale
    num_steps = min(c.num_unroll_steps, actions.shape[1])
    for k in range(num_steps):
        action_oh = nn.functional.one_hot(actions[:, k], num_classes=c.action_dim).float()

        # Scale gradient for dynamics (balance initial vs unrolled)
        latent_scaled = latent.detach() * (1.0 - gs) + latent * gs
        latent, rew_logits = model.dynamics.forward(latent_scaled, action_oh)
        pol_logits, val_logits = model.prediction.forward(latent)

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
    l2_reg = sum(p.pow(2).sum() for p in model.all_parameters())
    total_loss = total_loss + c.weight_decay * l2_reg

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.all_parameters(), max_norm=step_config.max_grad_norm)
    optimizer.step()

    return TrainStepMetrics(
        loss=float(total_loss.item()),
        policy_loss=float(policy_loss.item() * scale),
        value_loss=float(value_loss.item() * scale),
        reward_loss=float(reward_loss.item() * scale),
        l2_reg=float(l2_reg.item()),
    )
