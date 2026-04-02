"""Tests for forge.training.stability — EarlyStopping and PlateauDetector."""

from __future__ import annotations

from forge.training.stability import (
    EarlyStopping,
    EarlyStoppingConfig,
    PlateauDetector,
    PlateauDetectorConfig,
)

# ── EarlyStopping ────────────────────────────────────────────────────


def test_early_stopping_triggers_after_patience() -> None:
    """No improvement for ``patience`` steps triggers should_stop."""
    es = EarlyStopping(EarlyStoppingConfig(patience=3, min_delta=0.0))
    es.step(1.0)  # initialise best
    es.step(1.0)  # no improvement (1)
    es.step(1.0)  # no improvement (2)
    result = es.step(1.0)  # no improvement (3) -> triggers
    assert result is True
    assert es.should_stop is True


def test_early_stopping_resets_on_improvement() -> None:
    """Improvement resets the patience counter."""
    es = EarlyStopping(EarlyStoppingConfig(patience=3, min_delta=0.0))
    es.step(1.0)
    es.step(1.0)  # no improvement (1)
    es.step(1.0)  # no improvement (2)
    es.step(2.0)  # improvement -> resets counter
    es.step(2.0)  # no improvement (1)
    assert es.should_stop is False
    assert es.best_metric == 2.0


def test_early_stopping_min_delta() -> None:
    """Improvement smaller than min_delta doesn't count."""
    es = EarlyStopping(EarlyStoppingConfig(patience=2, min_delta=0.1))
    es.step(1.0)
    es.step(1.05)  # below min_delta -> no improvement (1)
    result = es.step(1.05)  # no improvement (2) -> triggers
    assert result is True


def test_early_stopping_mode_min() -> None:
    """mode='min' tracks decreasing metric (for loss)."""
    es = EarlyStopping(EarlyStoppingConfig(patience=2, min_delta=0.0, mode="min"))
    es.step(1.0)
    es.step(0.5)  # improvement (lower)
    assert es.best_metric == 0.5
    es.step(0.6)  # worse (higher) (1)
    result = es.step(0.7)  # worse (2) -> triggers
    assert result is True


def test_early_stopping_reset() -> None:
    """reset() clears all internal state."""
    es = EarlyStopping(EarlyStoppingConfig(patience=1))
    es.step(1.0)
    es.step(1.0)  # triggers
    assert es.should_stop is True
    es.reset()
    assert es.should_stop is False
    assert es.best_metric == float("-inf")


# ── PlateauDetector ──────────────────────────────────────────────────


def test_plateau_detector_detects_plateau() -> None:
    """Flat metric for patience steps returns a new LR."""
    pd = PlateauDetector(
        PlateauDetectorConfig(patience=3, factor=0.5, cooldown=0),
    )
    current_lr = 0.01
    pd.step(1.0, current_lr=current_lr)  # initialise
    pd.step(1.0, current_lr=current_lr)  # no improvement (1)
    pd.step(1.0, current_lr=current_lr)  # no improvement (2)
    new_lr = pd.step(1.0, current_lr=current_lr)  # no improvement (3)
    assert new_lr is not None
    assert abs(new_lr - 0.005) < 1e-9


def test_plateau_detector_cooldown() -> None:
    """No trigger during cooldown period after a reduction."""
    pd = PlateauDetector(
        PlateauDetectorConfig(patience=1, factor=0.5, cooldown=2),
    )
    lr = 0.01
    pd.step(1.0, current_lr=lr)
    new_lr = pd.step(1.0, current_lr=lr)  # triggers reduction
    assert new_lr is not None
    # Now in cooldown for 2 steps
    assert pd.step(1.0, current_lr=new_lr) is None  # cooldown (1)
    assert pd.step(1.0, current_lr=new_lr) is None  # cooldown (2)


def test_plateau_detector_min_lr() -> None:
    """Doesn't reduce below min_lr."""
    pd = PlateauDetector(
        PlateauDetectorConfig(
            patience=1, factor=0.5, min_lr=0.005, cooldown=0,
        ),
    )
    lr = 0.01
    pd.step(1.0, current_lr=lr)
    new_lr = pd.step(1.0, current_lr=lr)  # 0.01 * 0.5 = 0.005
    assert new_lr is not None
    assert abs(new_lr - 0.005) < 1e-9
    # Next plateau: 0.005 * 0.5 = 0.0025 < min_lr -> should not reduce
    pd.step(1.0, current_lr=new_lr)
    result = pd.step(1.0, current_lr=new_lr)
    assert result is None  # already at min_lr
