"""Tests for the meta-learner module."""
from __future__ import annotations

from forge.integration.meta_learner import MetaLearner, MetaLearnerConfig


class TestMetaLearnerConfig:
    """Tests for MetaLearnerConfig."""

    def test_defaults(self) -> None:
        config = MetaLearnerConfig()
        assert config.meta_lr == 0.001
        assert config.adaptation_window == 50
        assert config.min_lr == 1e-5
        assert config.max_lr == 1e-2


class TestMetaLearner:
    """Tests for MetaLearner."""

    def test_creation(self) -> None:
        ml = MetaLearner()
        assert ml.current_lr == 3e-4
        assert ml.adaptation_history == []

    def test_creation_with_config(self) -> None:
        config = MetaLearnerConfig(meta_lr=0.01)
        ml = MetaLearner(config=config)
        assert ml.config.meta_lr == 0.01

    def test_record_domain_reward(self) -> None:
        ml = MetaLearner()
        ml.record_domain_reward("navigation", 1.0)
        ml.record_domain_reward("navigation", 2.0)
        assert len(ml._domain_histories["navigation"]) == 2

    def test_adapt_insufficient_data(self) -> None:
        ml = MetaLearner()
        # Not enough data to adapt
        ml.record_domain_reward("nav", 1.0)
        lr = ml.adapt()
        assert lr == ml.current_lr  # unchanged

    def test_adapt_with_sufficient_data(self) -> None:
        config = MetaLearnerConfig(adaptation_window=5)
        ml = MetaLearner(config=config)

        # Generate improving rewards
        for i in range(20):
            ml.record_domain_reward("nav", float(i) * 0.1)

        ml.adapt()
        # Should have adapted
        assert len(ml.adaptation_history) > 0

    def test_adapt_slow_improvement_increases_lr(self) -> None:
        config = MetaLearnerConfig(adaptation_window=5)
        ml = MetaLearner(config=config)

        # Generate flat rewards (no improvement)
        for _ in range(20):
            ml.record_domain_reward("nav", 0.5)

        initial_lr = ml.current_lr
        ml.adapt()
        # Flat performance → slow adaptation → LR should increase
        assert ml.current_lr >= initial_lr

    def test_adapt_fast_improvement_decreases_lr(self) -> None:
        config = MetaLearnerConfig(adaptation_window=5)
        ml = MetaLearner(config=config)

        # Generate rapidly improving rewards
        for i in range(20):
            ml.record_domain_reward("nav", float(i))

        initial_lr = ml.current_lr
        ml.adapt()
        # Fast improvement → LR should decrease
        assert ml.current_lr <= initial_lr

    def test_lr_bounded_above(self) -> None:
        config = MetaLearnerConfig(adaptation_window=5, max_lr=0.01, meta_lr=0.5)
        ml = MetaLearner(config=config)
        ml.current_lr = 0.009

        # Generate flat rewards to trigger LR increase
        for _ in range(20):
            ml.record_domain_reward("nav", 0.5)

        ml.adapt()
        assert ml.current_lr <= config.max_lr

    def test_lr_bounded_below(self) -> None:
        config = MetaLearnerConfig(adaptation_window=5, min_lr=1e-5, meta_lr=0.5)
        ml = MetaLearner(config=config)
        ml.current_lr = 2e-5

        # Generate rapidly improving rewards
        for i in range(20):
            ml.record_domain_reward("nav", float(i))

        ml.adapt()
        assert ml.current_lr >= config.min_lr

    def test_multiple_domains(self) -> None:
        config = MetaLearnerConfig(adaptation_window=5)
        ml = MetaLearner(config=config)

        for i in range(20):
            ml.record_domain_reward("nav", float(i) * 0.1)
            ml.record_domain_reward("combat", float(i) * 0.05)

        ml.adapt()
        assert "nav" in ml._domain_histories
        assert "combat" in ml._domain_histories
