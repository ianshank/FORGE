"""Constitutional RL pre-trainer for MangoMAS integration.

Maps FORGE safety constraints to MangoMAS's 5 constitutional principles
and trains constraint-aware policy/value networks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from forge.mangomas.config import (
    DEFAULT_CONSTITUTIONAL_CONSTRAINTS,
    ConstitutionalTrainerConfig,
)

logger = logging.getLogger(__name__)

# Backward-compatible alias — canonical source is config.DEFAULT_CONSTITUTIONAL_CONSTRAINTS
DEFAULT_CONSTRAINTS: list[dict[str, Any]] = DEFAULT_CONSTITUTIONAL_CONSTRAINTS


@dataclass
class ConstraintViolation:
    """Record of a constraint violation."""

    constraint_name: str
    value: float
    threshold: float
    is_lower_bound: bool
    severity: float  # how far past the threshold


@dataclass
class ConstitutionalDataset:
    """Dataset for constitutional RL training."""

    observations: np.ndarray  # (N, state_dim)
    actions: np.ndarray  # (N,)
    rewards: np.ndarray  # (N,)
    constraint_violations: np.ndarray  # (N, num_constraints) binary
    penalties: np.ndarray  # (N,)

    @property
    def num_samples(self) -> int:
        return len(self.actions)

    @property
    def violation_rate(self) -> float:
        """Fraction of steps with any constraint violation."""
        if self.num_samples == 0:
            return 0.0
        return float(np.any(self.constraint_violations > 0, axis=1).mean())


@dataclass
class ConstitutionalTrainResult:
    """Training result metrics."""

    final_loss: float
    final_violation_rate: float
    epochs_run: int
    loss_history: list[float] = field(default_factory=list)


class ConstitutionalPreTrainer:
    """Pre-trains constitutional RL policy on FORGE constraint scenarios.

    Learns to respect safety boundaries (battery, altitude, speed, geofence,
    threat exclusion) that transfer to MangoMAS's 5 constitutional principles.
    """

    def __init__(
        self,
        config: ConstitutionalTrainerConfig | None = None,
        constraints: list[dict[str, Any]] | None = None,
    ) -> None:
        self.config = config or ConstitutionalTrainerConfig()
        self.constraints = constraints or self.config.constraints
        self._weights: dict[str, np.ndarray] | None = None
        logger.info(
            "ConstitutionalPreTrainer: %d constraints, penalty_weight=%.1f",
            len(self.constraints),
            self.config.penalty_weight,
        )

    def check_violations(self, obs: dict[str, float]) -> list[ConstraintViolation]:
        """Check which constraints are violated in a given observation."""
        violations = []
        for c in self.constraints:
            value = obs.get(c["forge_field"], 0.0)
            threshold = c["threshold"]
            is_lb = c["is_lower_bound"]

            violated = value < threshold if is_lb else value > threshold
            if violated:
                severity = abs(value - threshold)
                violations.append(
                    ConstraintViolation(
                        constraint_name=c["name"],
                        value=value,
                        threshold=threshold,
                        is_lower_bound=is_lb,
                        severity=severity,
                    )
                )
        return violations

    def compute_penalty(self, violations: list[ConstraintViolation]) -> float:
        """Compute scalar penalty from constraint violations."""
        if not violations:
            return 0.0
        total_severity = sum(v.severity for v in violations)
        return self.config.penalty_weight * total_severity

    def build_dataset(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        obs_dicts: list[dict[str, float]],
        *,
        teacher_constraint_critiques: list[dict[str, bool]] | None = None,
        teacher_severity_default: float = 1.0,
    ) -> ConstitutionalDataset:
        """Build a constitutional dataset with violation annotations.

        When ``teacher_constraint_critiques`` is supplied — one dict per
        sample, mapping constraint name → bool — the rule-derived
        ``violation_matrix`` is OR-merged with the teacher's view. The
        per-sample penalty is recomputed as
        ``max(rule_penalty, teacher_severity_default * num_teacher_flags
        * penalty_weight)`` so teacher critiques can lift penalties on
        samples the rule-based detector missed.
        """
        n = len(observations)
        n_constraints = len(self.constraints)
        violation_matrix = np.zeros((n, n_constraints), dtype=np.float32)
        penalties = np.zeros(n, dtype=np.float32)

        use_teacher = bool(teacher_constraint_critiques)
        if use_teacher and len(teacher_constraint_critiques or []) != n:
            msg = "teacher_constraint_critiques must have the same length as observations"
            raise ValueError(msg)
        name_to_index = {c["name"]: j for j, c in enumerate(self.constraints)}

        for i, obs in enumerate(obs_dicts):
            violations = self.check_violations(obs)
            for v in violations:
                idx = name_to_index.get(v.constraint_name)
                if idx is not None:
                    violation_matrix[i, idx] = 1.0
            rule_penalty = self.compute_penalty(violations)

            if use_teacher:
                critique = (teacher_constraint_critiques or [{}])[i] or {}
                teacher_flags = 0
                for name, flag in critique.items():
                    if not bool(flag):
                        continue
                    idx = name_to_index.get(name)
                    if idx is None:
                        continue
                    if violation_matrix[i, idx] == 0.0:
                        teacher_flags += 1
                    violation_matrix[i, idx] = 1.0
                teacher_penalty = (
                    self.config.penalty_weight * teacher_severity_default * float(teacher_flags)
                )
                penalties[i] = max(rule_penalty, teacher_penalty)
            else:
                penalties[i] = rule_penalty

        return ConstitutionalDataset(
            observations=observations,
            actions=actions,
            rewards=rewards,
            constraint_violations=violation_matrix,
            penalties=penalties,
        )

    def train(self, dataset: ConstitutionalDataset) -> ConstitutionalTrainResult:
        """Train a constraint-aware policy network."""
        rng = np.random.default_rng(self.config.seed)
        state_dim = self._resolve_state_dim(dataset)
        n_actions = self._resolve_action_dim(dataset)

        # Initialize policy weights
        scale = np.sqrt(6.0 / (state_dim + n_actions))
        w_policy = rng.uniform(-scale, scale, (n_actions, state_dim)).astype(np.float32)
        b_policy = np.zeros(n_actions, dtype=np.float32)

        # Value head
        w_value = rng.uniform(-scale, scale, (1, state_dim)).astype(np.float32)
        b_value = np.zeros(1, dtype=np.float32)

        loss_history: list[float] = []

        for epoch in range(self.config.num_epochs):
            indices = rng.permutation(dataset.num_samples)
            epoch_loss = 0.0

            for start in range(0, dataset.num_samples, self.config.batch_size):
                end = min(start + self.config.batch_size, dataset.num_samples)
                batch_idx = indices[start:end]
                x = dataset.observations[batch_idx]
                a = dataset.actions[batch_idx]
                r = dataset.rewards[batch_idx] - dataset.penalties[batch_idx]

                # Policy forward
                logits = x @ w_policy.T + b_policy
                logits_max = logits.max(axis=1, keepdims=True)
                exp_logits = np.exp(logits - logits_max)
                probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

                # Value forward
                values = (x @ w_value.T + b_value).squeeze()

                # Policy gradient loss
                advantages = r - values
                log_probs = np.log(probs[np.arange(len(a)), a] + 1e-8)
                policy_loss = -np.mean(log_probs * advantages)

                # Value loss
                value_loss = np.mean((values - r) ** 2)

                batch_loss = policy_loss + self.config.value_loss_weight * value_loss
                epoch_loss += float(batch_loss) * len(a)

                # Gradient updates (simplified)
                grad = probs.copy()
                grad[np.arange(len(a)), a] -= 1
                grad /= len(a)
                dw_p = grad.T @ x
                w_policy -= self.config.learning_rate * dw_p
                b_policy -= self.config.learning_rate * grad.sum(axis=0)

            epoch_loss /= max(dataset.num_samples, 1)
            loss_history.append(epoch_loss)

            if (epoch + 1) % self.config.log_interval == 0:
                logger.debug(
                    "Constitutional epoch %d/%d: loss=%.4f",
                    epoch + 1,
                    self.config.num_epochs,
                    epoch_loss,
                )

        self._weights = {
            "policy_w": w_policy,
            "policy_b": b_policy,
            "value_w": w_value,
            "value_b": b_value,
        }

        return ConstitutionalTrainResult(
            final_loss=loss_history[-1] if loss_history else 0.0,
            final_violation_rate=dataset.violation_rate,
            epochs_run=self.config.num_epochs,
            loss_history=loss_history,
        )

    def _resolve_state_dim(self, dataset: ConstitutionalDataset) -> int:
        """Resolve the state dimension from data when available, else config."""
        if dataset.num_samples > 0:
            return int(dataset.observations.shape[1])
        return self.config.state_dim

    def _resolve_action_dim(self, dataset: ConstitutionalDataset) -> int:
        """Resolve the action dimension from data when available, else config."""
        if dataset.num_samples > 0:
            return int(dataset.actions.max()) + 1
        return self.config.action_dim

    def export_weights(self, path: str | Path) -> None:
        """Export trained weights as .npz file."""
        if self._weights is None:
            raise RuntimeError("No trained weights. Call train() first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(path), **self._weights)  # type: ignore[arg-type]
        logger.info("Constitutional weights exported to %s", path)
