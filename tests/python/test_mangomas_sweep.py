"""Tests for MangoMAS MCTS sweep runner."""

from __future__ import annotations

from typing import Any

import pytest

from forge.mangomas.config import SweepConfig
from forge.mangomas.sweep_runner import MCTSSweepRunner, SweepReport, SweepResult


def _mock_evaluate(config: dict[str, Any], episodes: int) -> tuple[float, float, float]:
    """Mock evaluation: reward scales with c_puct, inversely with sims."""
    reward = config["c_puct"] * 10.0 - config["num_simulations"] * 0.01
    return (reward, 1.0, float(config["num_simulations"]) * 10.0)


class TestMCTSSweepRunner:
    """Tests for MCTSSweepRunner."""

    def test_generate_grid(self) -> None:
        config = SweepConfig(c_puct_steps=2, sim_budget_steps=2, depth_steps=2, discount_steps=2)
        runner = MCTSSweepRunner(config)
        grid = runner.generate_grid()
        assert len(grid) == 2 * 2 * 2 * 2  # 16

    def test_grid_values_in_range(self) -> None:
        config = SweepConfig(c_puct_steps=3, sim_budget_steps=2, depth_steps=2, discount_steps=2)
        runner = MCTSSweepRunner(config)
        grid = runner.generate_grid()
        for g in grid:
            assert 0.5 <= g["c_puct"] <= 3.0
            assert 10 <= g["num_simulations"] <= 500
            assert 10 <= g["max_depth"] <= 100
            assert 0.9 <= g["discount"] <= 0.999

    def test_run_sweep(self) -> None:
        config = SweepConfig(
            c_puct_steps=2,
            sim_budget_steps=2,
            depth_steps=2,
            discount_steps=2,
            episodes_per_config=5,
        )
        runner = MCTSSweepRunner(config)
        report = runner.run_sweep(_mock_evaluate)
        assert len(report.results) == 16
        assert report.best is not None
        assert report.total_time_secs > 0

    def test_best_has_highest_reward(self) -> None:
        config = SweepConfig(
            c_puct_steps=2,
            sim_budget_steps=2,
            depth_steps=2,
            discount_steps=2,
            episodes_per_config=5,
        )
        runner = MCTSSweepRunner(config)
        report = runner.run_sweep(_mock_evaluate)
        best_reward = report.best.mean_reward
        for r in report.results:
            assert r.mean_reward <= best_reward + 1e-6

    def test_run_sweep_custom_grid(self) -> None:
        runner = MCTSSweepRunner()
        grid = [
            {"c_puct": 1.0, "num_simulations": 50, "max_depth": 20, "discount": 0.99},
            {"c_puct": 2.0, "num_simulations": 100, "max_depth": 30, "discount": 0.95},
        ]
        report = runner.run_sweep(_mock_evaluate, param_grid=grid)
        assert len(report.results) == 2

    def test_export_optimal_config(self, tmp_path: Any) -> None:
        runner = MCTSSweepRunner()
        grid = [{"c_puct": 1.5, "num_simulations": 100, "max_depth": 50, "discount": 0.99}]
        report = runner.run_sweep(_mock_evaluate, param_grid=grid)
        out_path = tmp_path / "optimal.json"
        runner.export_optimal_config(report, out_path)
        assert out_path.exists()

    def test_compare_puct_vs_ucb1(self) -> None:
        config = SweepConfig(
            c_puct_steps=2,
            sim_budget_steps=2,
            depth_steps=1,
            discount_steps=1,
            episodes_per_config=3,
        )
        runner = MCTSSweepRunner(config)

        def _puct_eval(c: dict, n: int) -> tuple[float, float, float]:
            return (c["c_puct"] * 5.0, 1.0, 100.0)

        def _ucb1_eval(c: dict, n: int) -> tuple[float, float, float]:
            return (c["c_puct"] * 4.0, 1.0, 80.0)

        report = runner.compare_puct_vs_ucb1(_puct_eval, _ucb1_eval)
        assert report.winner == "puct"
        assert report.puct_best is not None
        assert report.ucb1_best is not None

    def test_summary(self) -> None:
        result = SweepResult(
            config={"c_puct": 1.5},
            mean_reward=10.0,
            std_reward=1.0,
            mean_planning_time_us=500.0,
            episodes_run=50,
        )
        report = SweepReport(results=[result], best=result, total_time_secs=5.0)
        summary = report.summary()
        assert "1.5" in summary
        assert "10.0" in summary

    def test_summary_no_results(self) -> None:
        report = SweepReport(results=[])
        assert report.summary() == "No results"

    def test_export_no_best(self, tmp_path: Any) -> None:
        runner = MCTSSweepRunner()
        report = SweepReport(results=[])
        runner.export_optimal_config(report, tmp_path / "out.json")
        # Should not crash; file should not exist since no best
        assert not (tmp_path / "out.json").exists()

    def test_export_unsupported_format(self, tmp_path: Any) -> None:
        runner = MCTSSweepRunner()
        result = SweepResult(
            config={"c_puct": 1.0},
            mean_reward=1.0,
            std_reward=0.1,
            mean_planning_time_us=50.0,
            episodes_run=10,
        )
        report = SweepReport(results=[result], best=result)
        with pytest.raises(ValueError, match="Unsupported format"):
            runner.export_optimal_config(report, tmp_path / "out.yaml", fmt="yaml")
