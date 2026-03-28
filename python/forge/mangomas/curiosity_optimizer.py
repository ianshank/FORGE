"""Curiosity weight optimizer for MangoMAS integration.

Meta-learns optimal curiosity channel weights (social, epistemic,
perceptual, metacognitive) via FORGE multi-agent scenarios.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_CURIOSITY_CHANNELS = ["social", "epistemic", "perceptual", "metacognitive"]
DEFAULT_INITIAL_WEIGHTS = [0.4, 0.3, 0.2, 0.1]


@dataclass
class CuriosityWeights:
    """Optimized curiosity channel weights."""

    weights: dict[str, float]
    fitness: float = 0.0
    iterations: int = 0

    def as_array(self) -> np.ndarray:
        """Return weights as an ordered numpy array."""
        return np.array(
            [self.weights[c] for c in DEFAULT_CURIOSITY_CHANNELS],
            dtype=np.float32,
        )

    def __repr__(self) -> str:
        parts = [f"{k}={v:.3f}" for k, v in self.weights.items()]
        return f"CuriosityWeights({', '.join(parts)}, fitness={self.fitness:.4f})"


class CuriosityWeightOptimizer:
    """Meta-learns optimal curiosity channel weights via evolutionary strategy.

    Uses a simple (μ+λ) evolution strategy over FORGE multi-agent scenarios
    to find the best blend of curiosity channels for MangoMAS transfer.
    """

    def __init__(
        self,
        channels: list[str] | None = None,
        initial_weights: list[float] | None = None,
        population_size: int = 20,
        sigma: float = 0.1,
        learning_rate: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.channels = channels or DEFAULT_CURIOSITY_CHANNELS
        self._initial = np.array(
            initial_weights or DEFAULT_INITIAL_WEIGHTS, dtype=np.float32
        )
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self._rng = np.random.default_rng(seed)
        logger.info(
            "CuriosityWeightOptimizer: %d channels, pop=%d, σ=%.2f",
            len(self.channels),
            population_size,
            sigma,
        )

    def _normalize(self, w: np.ndarray) -> np.ndarray:
        """Ensure weights sum to 1 and are non-negative."""
        w = np.maximum(w, 0.0)
        total = w.sum()
        if total < 1e-8:
            return np.ones_like(w) / len(w)
        return w / total

    def optimize(
        self,
        evaluate_fn: Callable[[dict[str, float]], float],
        num_iterations: int = 50,
    ) -> CuriosityWeights:
        """Optimize curiosity weights using evolution strategy.

        Args:
            evaluate_fn: Callable(weights_dict) -> fitness_score.
                Called with {"social": 0.4, "epistemic": 0.3, ...}.
            num_iterations: Number of ES iterations.

        Returns:
            CuriosityWeights with optimized values.
        """
        current = self._initial.copy()
        best_fitness = float("-inf")
        best_weights = current.copy()

        for iteration in range(num_iterations):
            # Generate population via Gaussian perturbation
            noise = self._rng.normal(0, self.sigma, (self.population_size, len(self.channels)))
            population = np.array([
                self._normalize(current + n) for n in noise
            ])

            # Evaluate fitness
            fitnesses = np.array([
                evaluate_fn(dict(zip(self.channels, w)))
                for w in population
            ])

            # Update via fitness-weighted mean
            advantages = fitnesses - fitnesses.mean()
            std = fitnesses.std() + 1e-8
            advantages /= std

            gradient = np.zeros_like(current)
            for i in range(self.population_size):
                gradient += advantages[i] * noise[i]
            gradient /= self.population_size

            current = self._normalize(current + self.learning_rate * gradient)

            # Track best
            best_idx = np.argmax(fitnesses)
            if fitnesses[best_idx] > best_fitness:
                best_fitness = float(fitnesses[best_idx])
                best_weights = population[best_idx].copy()

            if (iteration + 1) % 10 == 0:
                logger.debug(
                    "Curiosity ES iter %d/%d: best_fitness=%.4f",
                    iteration + 1, num_iterations, best_fitness,
                )

        result = CuriosityWeights(
            weights=dict(zip(self.channels, best_weights.tolist())),
            fitness=best_fitness,
            iterations=num_iterations,
        )
        logger.info("Curiosity optimization complete: %s", result)
        return result
