"""Tests for MangoMAS curiosity weight optimizer."""

from __future__ import annotations

import numpy as np
import pytest

from forge.mangomas.config import (
    DEFAULT_CURIOSITY_CHANNELS as CONFIG_DEFAULT_CURIOSITY_CHANNELS,
)
from forge.mangomas.config import (
    DEFAULT_CURIOSITY_WEIGHTS,
    CuriosityOptimizerConfig,
)
from forge.mangomas.curiosity_optimizer import (
    DEFAULT_CURIOSITY_CHANNELS,
    DEFAULT_INITIAL_WEIGHTS,
    CuriosityWeightOptimizer,
    CuriosityWeights,
)


class TestCuriosityWeights:
    """Tests for CuriosityWeights dataclass."""

    def test_as_array(self) -> None:
        weights = CuriosityWeights(
            weights={"social": 0.4, "epistemic": 0.3, "perceptual": 0.2, "metacognitive": 0.1}
        )
        arr = weights.as_array()
        assert arr.dtype == np.float32
        np.testing.assert_allclose(arr, [0.4, 0.3, 0.2, 0.1])

    def test_repr(self) -> None:
        weights = CuriosityWeights(weights={"social": 0.5, "epistemic": 0.5}, fitness=1.234)
        text = repr(weights)
        assert "social=0.500" in text
        assert "epistemic=0.500" in text
        assert "fitness=1.2340" in text

    def test_default_fitness_and_iterations(self) -> None:
        weights = CuriosityWeights(weights={"a": 1.0})
        assert weights.fitness == 0.0
        assert weights.iterations == 0


class TestCuriosityWeightOptimizer:
    """Tests for CuriosityWeightOptimizer."""

    def test_default_init(self) -> None:
        opt = CuriosityWeightOptimizer(seed=0)
        assert opt.channels == DEFAULT_CURIOSITY_CHANNELS
        assert opt.population_size == 20
        assert opt.sigma == pytest.approx(0.1)

    def test_custom_config(self) -> None:
        cfg = CuriosityOptimizerConfig(
            channels=["a", "b"],
            initial_weights=[0.6, 0.4],
            population_size=10,
            sigma=0.2,
            learning_rate=0.1,
            seed=99,
        )
        opt = CuriosityWeightOptimizer(config=cfg)
        assert opt.channels == ["a", "b"]
        assert opt.population_size == 10
        assert opt.sigma == pytest.approx(0.2)

    def test_parameter_overrides(self) -> None:
        opt = CuriosityWeightOptimizer(
            population_size=5,
            sigma=0.3,
            learning_rate=0.01,
            seed=42,
        )
        assert opt.population_size == 5
        assert opt.sigma == pytest.approx(0.3)
        assert opt.learning_rate == pytest.approx(0.01)

    def test_normalize_positive(self) -> None:
        opt = CuriosityWeightOptimizer(seed=0)
        w = np.array([2.0, 3.0, 5.0, 0.0], dtype=np.float32)
        result = opt._normalize(w)
        assert result.sum() == pytest.approx(1.0)
        assert np.all(result >= 0.0)

    def test_normalize_all_negative(self) -> None:
        opt = CuriosityWeightOptimizer(seed=0)
        w = np.array([-1.0, -2.0, -3.0, -4.0], dtype=np.float32)
        result = opt._normalize(w)
        # All negatives clipped to 0, fallback to uniform
        assert result.sum() == pytest.approx(1.0)
        np.testing.assert_allclose(result, 0.25)

    def test_normalize_all_zero(self) -> None:
        opt = CuriosityWeightOptimizer(seed=0)
        w = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        result = opt._normalize(w)
        assert result.sum() == pytest.approx(1.0)
        np.testing.assert_allclose(result, 0.25)

    def test_optimize_returns_curiosity_weights(self) -> None:
        opt = CuriosityWeightOptimizer(population_size=5, seed=42)

        def evaluate(w: dict[str, float]) -> float:
            return w.get("social", 0.0) * 2.0 + w.get("epistemic", 0.0)

        result = opt.optimize(evaluate, num_iterations=3)
        assert isinstance(result, CuriosityWeights)
        assert result.iterations == 3
        assert len(result.weights) == len(DEFAULT_CURIOSITY_CHANNELS)

    def test_optimize_weights_sum_to_one(self) -> None:
        opt = CuriosityWeightOptimizer(population_size=5, seed=42)

        def evaluate(w: dict[str, float]) -> float:
            return sum(w.values())

        result = opt.optimize(evaluate, num_iterations=5)
        total = sum(result.weights.values())
        assert total == pytest.approx(1.0, abs=1e-5)

    def test_optimize_all_weights_non_negative(self) -> None:
        opt = CuriosityWeightOptimizer(population_size=10, seed=42)

        def evaluate(w: dict[str, float]) -> float:
            return -w.get("social", 0.0)  # penalize social

        result = opt.optimize(evaluate, num_iterations=10)
        for v in result.weights.values():
            assert v >= 0.0

    def test_optimize_improves_fitness(self) -> None:
        opt = CuriosityWeightOptimizer(population_size=10, seed=42)

        def evaluate(w: dict[str, float]) -> float:
            # Optimal: all weight on epistemic
            return w.get("epistemic", 0.0) * 10.0

        result = opt.optimize(evaluate, num_iterations=20)
        # Initial epistemic weight is 0.3 → fitness ~3.0
        # After optimization, should beat initial
        assert result.fitness > 3.0

    def test_optimize_deterministic_with_seed(self) -> None:
        def evaluate(w: dict[str, float]) -> float:
            return w.get("social", 0.0) + w.get("epistemic", 0.0) * 2

        r1 = CuriosityWeightOptimizer(population_size=5, seed=123).optimize(
            evaluate, num_iterations=5
        )
        r2 = CuriosityWeightOptimizer(population_size=5, seed=123).optimize(
            evaluate, num_iterations=5
        )
        np.testing.assert_allclose(r1.as_array(), r2.as_array())
        assert r1.fitness == pytest.approx(r2.fitness)

    def test_optimize_custom_channels(self) -> None:
        opt = CuriosityWeightOptimizer(
            channels=["alpha", "beta"],
            initial_weights=[0.5, 0.5],
            population_size=5,
            seed=42,
        )

        def evaluate(w: dict[str, float]) -> float:
            return w["alpha"] - w["beta"]

        result = opt.optimize(evaluate, num_iterations=5)
        assert "alpha" in result.weights
        assert "beta" in result.weights
        assert len(result.weights) == 2


class TestDefaultConstants:
    """Tests for module-level default constants."""

    def test_default_channels(self) -> None:
        assert len(DEFAULT_CURIOSITY_CHANNELS) == 4
        assert "social" in DEFAULT_CURIOSITY_CHANNELS
        assert "epistemic" in DEFAULT_CURIOSITY_CHANNELS
        assert "perceptual" in DEFAULT_CURIOSITY_CHANNELS
        assert "metacognitive" in DEFAULT_CURIOSITY_CHANNELS

    def test_default_weights(self) -> None:
        assert len(DEFAULT_INITIAL_WEIGHTS) == len(DEFAULT_CURIOSITY_CHANNELS)
        assert sum(DEFAULT_INITIAL_WEIGHTS) == pytest.approx(1.0)
        assert all(w > 0.0 for w in DEFAULT_INITIAL_WEIGHTS)

    def test_default_aliases_match_config_defaults(self) -> None:
        assert list(CONFIG_DEFAULT_CURIOSITY_CHANNELS) == DEFAULT_CURIOSITY_CHANNELS
        assert list(DEFAULT_CURIOSITY_WEIGHTS) == DEFAULT_INITIAL_WEIGHTS
