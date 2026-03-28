"""Tests for MangoMAS curiosity weight optimizer."""
from __future__ import annotations

import numpy as np
import pytest
from forge.mangomas.config import CuriosityOptimizerConfig
from forge.mangomas.curiosity_optimizer import (
    DEFAULT_CURIOSITY_CHANNELS,
    DEFAULT_INITIAL_WEIGHTS,
    CuriosityWeightOptimizer,
    CuriosityWeights,
)


class TestCuriosityWeights:
    """Tests for CuriosityWeights dataclass."""

    def test_as_array(self) -> None:
        w = CuriosityWeights(
            weights={"social": 0.4, "epistemic": 0.3, "perceptual": 0.2, "metacognitive": 0.1}
        )
        arr = w.as_array()
        assert arr.shape == (4,)
        assert arr.dtype == np.float32
        assert np.isclose(arr.sum(), 1.0)

    def test_repr(self) -> None:
        w = CuriosityWeights(
            weights={"social": 0.5, "epistemic": 0.5},
            fitness=1.23,
        )
        r = repr(w)
        assert "social=0.500" in r
        assert "fitness=1.2300" in r

    def test_default_fitness(self) -> None:
        w = CuriosityWeights(weights={"a": 1.0})
        assert w.fitness == 0.0
        assert w.iterations == 0


class TestCuriosityWeightOptimizer:
    """Tests for CuriosityWeightOptimizer."""

    def test_init_defaults(self) -> None:
        opt = CuriosityWeightOptimizer()
        assert len(opt.channels) == 4
        assert opt.population_size == 20
        assert opt.sigma == pytest.approx(0.1)

    def test_init_with_config(self) -> None:
        config = CuriosityOptimizerConfig(
            population_size=10, sigma=0.2, learning_rate=0.1
        )
        opt = CuriosityWeightOptimizer(config=config)
        assert opt.population_size == 10
        assert opt.sigma == pytest.approx(0.2)
        assert opt.learning_rate == pytest.approx(0.1)

    def test_init_param_override(self) -> None:
        config = CuriosityOptimizerConfig(population_size=10)
        opt = CuriosityWeightOptimizer(config=config, population_size=50)
        assert opt.population_size == 50

    def test_init_custom_channels(self) -> None:
        opt = CuriosityWeightOptimizer(channels=["a", "b"], initial_weights=[0.5, 0.5])
        assert opt.channels == ["a", "b"]

    def test_normalize(self) -> None:
        opt = CuriosityWeightOptimizer()
        w = np.array([1.0, 2.0, 3.0, 4.0])
        normalized = opt._normalize(w)
        assert np.isclose(normalized.sum(), 1.0)
        assert np.all(normalized >= 0)

    def test_normalize_zeros(self) -> None:
        opt = CuriosityWeightOptimizer()
        w = np.array([0.0, 0.0, 0.0, 0.0])
        normalized = opt._normalize(w)
        assert np.isclose(normalized.sum(), 1.0)
        assert np.allclose(normalized, 0.25)

    def test_normalize_negative(self) -> None:
        opt = CuriosityWeightOptimizer()
        w = np.array([-1.0, 2.0, -3.0, 4.0])
        normalized = opt._normalize(w)
        assert np.isclose(normalized.sum(), 1.0)
        assert np.all(normalized >= 0)

    def test_optimize_simple(self) -> None:
        """Optimize with a simple fitness function that prefers social."""
        opt = CuriosityWeightOptimizer(population_size=10, seed=42)

        def evaluate(weights: dict[str, float]) -> float:
            return weights.get("social", 0.0) * 10.0

        result = opt.optimize(evaluate, num_iterations=5)
        assert isinstance(result, CuriosityWeights)
        assert result.fitness > 0
        assert result.iterations == 5
        assert len(result.weights) == 4

    def test_optimize_weights_sum_to_one(self) -> None:
        opt = CuriosityWeightOptimizer(population_size=10, seed=42)

        def evaluate(weights: dict[str, float]) -> float:
            return sum(weights.values())

        result = opt.optimize(evaluate, num_iterations=3)
        total = sum(result.weights.values())
        assert np.isclose(total, 1.0, atol=0.01)

    def test_optimize_tracks_best(self) -> None:
        opt = CuriosityWeightOptimizer(population_size=10, seed=42)
        rng = np.random.default_rng(99)

        def evaluate(weights: dict[str, float]) -> float:
            return float(rng.random())

        result = opt.optimize(evaluate, num_iterations=10)
        assert result.fitness > 0

    def test_default_constants_preserved(self) -> None:
        """Module-level constants preserved for backwards compatibility."""
        assert len(DEFAULT_CURIOSITY_CHANNELS) == 4
        assert len(DEFAULT_INITIAL_WEIGHTS) == 4
        assert np.isclose(sum(DEFAULT_INITIAL_WEIGHTS), 1.0)
