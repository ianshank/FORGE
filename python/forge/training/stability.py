"""Training stability monitors: early stopping and plateau detection.

Provides:
- EarlyStopping: Stop training when a monitored metric stops improving.
- PlateauDetector: Suggest learning-rate reductions when metric plateaus.

Adapted from AlphaGalerkin's stability module with torch dependency removed
and stdlib logging in place of structlog.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping.

    Attributes:
        patience: Number of evaluations without improvement before stopping.
        min_delta: Minimum change to qualify as improvement.
        mode: ``"max"`` if higher is better, ``"min"`` if lower is better.
    """

    patience: int = 10
    min_delta: float = 0.001
    mode: Literal["max", "min"] = "max"


class EarlyStopping:
    """Early stopping monitor.

    Tracks a scalar metric and signals when training should stop because
    the metric has not improved for ``patience`` consecutive evaluations.

    Example::

        es = EarlyStopping(EarlyStoppingConfig(patience=5))
        for epoch in range(max_epochs):
            val_metric = evaluate()
            if es.step(val_metric):
                break
    """

    def __init__(self, config: EarlyStoppingConfig | None = None) -> None:
        """Initialize early stopping.

        Args:
            config: Early stopping configuration.  Uses defaults when *None*.
        """
        self.config = config or EarlyStoppingConfig()
        self._best_value: float | None = None
        self._counter: int = 0
        self._should_stop: bool = False

    def step(self, metric: float) -> bool:
        """Update with a new metric value.

        Args:
            metric: Current metric value.

        Returns:
            ``True`` if training should stop.
        """
        if self._best_value is None:
            self._best_value = metric
            logger.debug("Early stopping initialized with value %.6f", metric)
            return False

        if self._is_improvement(metric):
            self._best_value = metric
            self._counter = 0
            logger.debug("Early stopping improvement: new best=%.6f", metric)
        else:
            self._counter += 1
            logger.debug(
                "Early stopping no improvement: counter=%d/%d (current=%.6f, best=%.6f)",
                self._counter,
                self.config.patience,
                metric,
                self._best_value,
            )
            if self._counter >= self.config.patience:
                self._should_stop = True
                logger.info(
                    "Early stopping triggered after %d evaluations without improvement (best=%.6f)",
                    self.config.patience,
                    self._best_value,
                )

        return self._should_stop

    def reset(self) -> None:
        """Reset early stopping state."""
        self._best_value = None
        self._counter = 0
        self._should_stop = False

    @property
    def should_stop(self) -> bool:
        """Whether training should stop."""
        return self._should_stop

    @property
    def best_metric(self) -> float:
        """Best metric value seen so far.

        Returns:
            Best value, or ``float('inf')`` / ``float('-inf')`` if no value
            has been recorded yet.
        """
        if self._best_value is None:
            return float("-inf") if self.config.mode == "max" else float("inf")
        return self._best_value

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_improvement(self, value: float) -> bool:
        """Check whether *value* represents an improvement over the best."""
        if self._best_value is None:
            return True
        if self.config.mode == "max":
            return value > self._best_value + self.config.min_delta
        return value < self._best_value - self.config.min_delta


@dataclass
class PlateauDetectorConfig:
    """Configuration for learning-rate plateau detection.

    Attributes:
        patience: Steps without improvement before suggesting a LR reduction.
        cooldown: Steps to wait after a reduction before checking again.
        min_lr: Minimum learning rate (will not suggest below this).
        factor: Multiplicative factor for LR reduction.
        mode: ``"max"`` if higher is better, ``"min"`` if lower is better.
    """

    patience: int = 5
    cooldown: int = 3
    min_lr: float = 1e-6
    factor: float = 0.5
    mode: Literal["max", "min"] = "min"


class PlateauDetector:
    """Detect metric plateaus and suggest learning-rate reductions.

    Unlike PyTorch's ``ReduceLROnPlateau``, this class does **not** hold a
    reference to an optimizer.  Instead it returns the suggested new LR (or
    ``None``) so the caller can apply the change however it likes.

    Example::

        detector = PlateauDetector(PlateauDetectorConfig())
        current_lr = 1e-3
        for step in range(total_steps):
            loss = train_step(current_lr)
            new_lr = detector.step(loss)
            if new_lr is not None:
                current_lr = new_lr
    """

    def __init__(self, config: PlateauDetectorConfig | None = None) -> None:
        """Initialize plateau detector.

        Args:
            config: Plateau detection configuration.  Uses defaults when *None*.
        """
        self.config = config or PlateauDetectorConfig()
        self._best_value: float | None = None
        self._counter: int = 0
        self._cooldown_counter: int = 0
        self._current_lr: float = 1.0  # caller sets via first reduction
        self._num_reductions: int = 0

    def step(self, metric: float, current_lr: float | None = None) -> float | None:
        """Update with a new metric value.

        Args:
            metric: Current metric value.
            current_lr: Current learning rate.  When provided, the returned
                new LR is computed from this value.  When *None*, the detector
                tracks LR internally starting from 1.0.

        Returns:
            Suggested new learning rate if a plateau was detected, otherwise
            ``None``.
        """
        if current_lr is not None:
            self._current_lr = current_lr

        # During cooldown, just tick down and skip checks.
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            return None

        if self._best_value is None:
            self._best_value = metric
            return None

        if self._is_improvement(metric):
            self._best_value = metric
            self._counter = 0
            return None

        self._counter += 1
        if self._counter >= self.config.patience:
            new_lr = max(self._current_lr * self.config.factor, self.config.min_lr)
            if new_lr < self._current_lr:
                logger.info(
                    "Plateau detected: reducing LR %.6g -> %.6g (reduction #%d)",
                    self._current_lr,
                    new_lr,
                    self._num_reductions + 1,
                )
                self._current_lr = new_lr
                self._num_reductions += 1
                self._counter = 0
                self._cooldown_counter = self.config.cooldown
                return new_lr

            # Already at min_lr.
            self._counter = 0
            self._cooldown_counter = self.config.cooldown
            return None

        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_improvement(self, value: float) -> bool:
        """Check whether *value* represents an improvement over the best."""
        if self._best_value is None:
            return True
        if self.config.mode == "min":
            return value < self._best_value
        return value > self._best_value
