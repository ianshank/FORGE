"""Tests for forge.training.curriculum — TaskTierCurriculum."""

from __future__ import annotations

from collections import Counter

from forge.training.curriculum import (
    CurriculumStage,
    TaskTierCurriculum,
    TaskTierCurriculumConfig,
)


def test_initial_stage() -> None:
    """step=0 returns the first stage's tiers."""
    curriculum = TaskTierCurriculum()
    stage = curriculum.current_stage(0)
    assert stage.task_tiers == [0, 1]


def test_stage_progression() -> None:
    """step > threshold returns the next stage."""
    curriculum = TaskTierCurriculum()
    stage1 = curriculum.current_stage(999)
    assert stage1.task_tiers == [0, 1]

    stage2 = curriculum.current_stage(1000)
    assert stage2.task_tiers == [0, 1, 2]

    stage3 = curriculum.current_stage(5000)
    assert stage3.task_tiers == [0, 1, 2, 3]


def test_weighted_sampling() -> None:
    """Over many samples, distribution roughly matches weights."""
    config = TaskTierCurriculumConfig(
        stages=[
            CurriculumStage(
                start_step=0,
                task_tiers=[0, 1],
                tier_weights=[0.8, 0.2],
            ),
        ],
        seed=123,
    )
    curriculum = TaskTierCurriculum(config)
    counts: Counter[int] = Counter()
    n_samples = 2000
    for _ in range(n_samples):
        tier = curriculum.sample_tier(0)
        counts[tier] += 1

    ratio_0 = counts[0] / n_samples
    # With 2000 samples, 0.8 +/- 0.05 is very generous.
    assert 0.7 < ratio_0 < 0.9, f"Expected ~0.8, got {ratio_0:.3f}"


def test_deterministic_seed() -> None:
    """Same seed produces the same tier sequence."""
    seq_a = []
    seq_b = []
    for seed_run in (seq_a, seq_b):
        c = TaskTierCurriculum(TaskTierCurriculumConfig(seed=99))
        for step in range(50):
            seed_run.append(c.sample_tier(step))
    assert seq_a == seq_b
