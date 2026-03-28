"""MCTS hyperparameter sweep runner for MangoMAS integration.

Orchestrates parameter grid sweeps via FORGE's high-throughput simulation,
evaluates configurations, and exports optimal settings.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from forge.mangomas.config import SweepConfig

logger = logging.getLogger(__name__)


@dataclass
class SweepResult:
    """Result from evaluating a single MCTS configuration."""

    config: dict[str, Any]
    mean_reward: float
    std_reward: float
    mean_planning_time_us: float
    episodes_run: int


@dataclass
class SweepReport:
    """Aggregate report from a full parameter sweep."""

    results: list[SweepResult]
    best: SweepResult | None = None
    total_time_secs: float = 0.0

    def summary(self) -> str:
        """Return a human-readable summary."""
        if not self.best:
            return "No results"
        return (
            f"Best config: {self.best.config} "
            f"(mean_reward={self.best.mean_reward:.4f}, "
            f"planning_time={self.best.mean_planning_time_us:.1f}μs) "
            f"from {len(self.results)} configurations in {self.total_time_secs:.1f}s"
        )


@dataclass
class ComparisonReport:
    """PUCT vs UCB1 comparison report."""

    puct_results: list[SweepResult]
    ucb1_results: list[SweepResult]
    puct_best: SweepResult | None = None
    ucb1_best: SweepResult | None = None
    winner: str = ""


class MCTSSweepRunner:
    """Orchestrates MCTS hyperparameter sweep via FORGE.

    Generates parameter grids, runs episodes in parallel via the Rust batch
    runner, and reports optimal configurations for MangoMAS transfer.
    """

    def __init__(self, config: SweepConfig | None = None) -> None:
        self.config = config or SweepConfig()
        self._rng = np.random.default_rng(self.config.seed)
        logger.info("MCTSSweepRunner initialized with seed=%d", self.config.seed)

    def generate_grid(self) -> list[dict[str, Any]]:
        """Generate the parameter grid from config ranges."""
        c_pucts = np.linspace(
            self.config.c_puct_range[0],
            self.config.c_puct_range[1],
            self.config.c_puct_steps,
        )
        sim_budgets = np.linspace(
            self.config.sim_budget_range[0],
            self.config.sim_budget_range[1],
            self.config.sim_budget_steps,
            dtype=int,
        )
        depths = np.linspace(
            self.config.depth_range[0],
            self.config.depth_range[1],
            self.config.depth_steps,
            dtype=int,
        )
        discounts = np.linspace(
            self.config.discount_range[0],
            self.config.discount_range[1],
            self.config.discount_steps,
        )

        grid = [
            {
                "c_puct": float(cp),
                "num_simulations": int(sb),
                "max_depth": int(d),
                "discount": float(disc),
            }
            for cp in c_pucts
            for sb in sim_budgets
            for d in depths
            for disc in discounts
        ]
        logger.info("Generated parameter grid with %d configurations", len(grid))
        return grid

    def run_sweep(
        self,
        evaluate_fn: Callable[[dict[str, Any], int], tuple[float, float, float]],
        param_grid: list[dict[str, Any]] | None = None,
    ) -> SweepReport:
        """Run the sweep, evaluating each config via evaluate_fn.

        Args:
            evaluate_fn: Callable(config, episodes) -> (mean_reward, std_reward, planning_time_us)
            param_grid: Optional explicit grid; generates from config if None.

        Returns:
            SweepReport with all results and best configuration.
        """
        if param_grid is None:
            param_grid = self.generate_grid()

        start = time.monotonic()
        results: list[SweepResult] = []

        for i, config in enumerate(param_grid):
            mean_r, std_r, plan_t = evaluate_fn(
                config, self.config.episodes_per_config
            )
            result = SweepResult(
                config=config,
                mean_reward=mean_r,
                std_reward=std_r,
                mean_planning_time_us=plan_t,
                episodes_run=self.config.episodes_per_config,
            )
            results.append(result)
            if (i + 1) % 10 == 0:
                logger.debug("Sweep progress: %d/%d", i + 1, len(param_grid))

        elapsed = time.monotonic() - start
        best = max(results, key=lambda r: r.mean_reward) if results else None

        report = SweepReport(results=results, best=best, total_time_secs=elapsed)
        logger.info("Sweep complete: %s", report.summary())
        return report

    def compare_puct_vs_ucb1(
        self,
        evaluate_puct_fn: Callable[[dict[str, Any], int], tuple[float, float, float]],
        evaluate_ucb1_fn: Callable[[dict[str, Any], int], tuple[float, float, float]],
        scenarios: list[str] | None = None,
    ) -> ComparisonReport:
        """Compare PUCT vs UCB1 exploration strategies."""
        grid = self.generate_grid()

        puct_results = []
        ucb1_results = []
        for config in grid:
            mr, sr, pt = evaluate_puct_fn(config, self.config.episodes_per_config)
            puct_results.append(
                SweepResult(config=config, mean_reward=mr, std_reward=sr,
                            mean_planning_time_us=pt,
                            episodes_run=self.config.episodes_per_config)
            )
            mr, sr, pt = evaluate_ucb1_fn(config, self.config.episodes_per_config)
            ucb1_results.append(
                SweepResult(config=config, mean_reward=mr, std_reward=sr,
                            mean_planning_time_us=pt,
                            episodes_run=self.config.episodes_per_config)
            )

        puct_best = max(puct_results, key=lambda r: r.mean_reward) if puct_results else None
        ucb1_best = max(ucb1_results, key=lambda r: r.mean_reward) if ucb1_results else None

        winner = "puct"
        if puct_best and ucb1_best:
            winner = "puct" if puct_best.mean_reward >= ucb1_best.mean_reward else "ucb1"

        return ComparisonReport(
            puct_results=puct_results,
            ucb1_results=ucb1_results,
            puct_best=puct_best,
            ucb1_best=ucb1_best,
            winner=winner,
        )

    def export_optimal_config(
        self, report: SweepReport, path: str | Path, fmt: str = "json"
    ) -> None:
        """Export the optimal configuration to a file."""
        if report.best is None:
            logger.warning("No best config to export")
            return

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "optimal_mcts_config": report.best.config,
            "mean_reward": report.best.mean_reward,
            "std_reward": report.best.std_reward,
            "mean_planning_time_us": report.best.mean_planning_time_us,
            "episodes_evaluated": report.best.episodes_run,
            "total_configs_tested": len(report.results),
        }

        if fmt == "json":
            with path.open("w") as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        logger.info("Exported optimal config to %s", path)
