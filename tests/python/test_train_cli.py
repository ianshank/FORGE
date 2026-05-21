"""Fast tests for the top-level training CLI parser."""

from __future__ import annotations

import pytest

# scripts/ is placed on sys.path by the root conftest.py (_ensure_importable).
from train import _DEFAULT_EPISODES, _DEFAULT_SEED, parse_args


class TestParseArgs:
    """Tests for the training script argument parser."""

    def test_defaults(self) -> None:
        """Default arguments should match constants."""
        args = parse_args([])
        assert args.agent == "random"
        assert args.episodes == _DEFAULT_EPISODES
        assert args.seed == _DEFAULT_SEED
        assert args.log_level == "INFO"
        assert args.collection_policy == "random"

    def test_custom_args(self) -> None:
        """Custom arguments are parsed correctly."""
        args = parse_args(
            [
                "--agent",
                "mappo",
                "--num-updates",
                "5",
                "--seed",
                "99",
                "--log-level",
                "DEBUG",
            ]
        )
        assert args.agent == "mappo"
        assert args.num_updates == 5
        assert args.seed == 99
        assert args.log_level == "DEBUG"

    def test_mangomas_args(self) -> None:
        """MangoMAS-specific arguments are parsed correctly."""
        args = parse_args(
            [
                "--agent",
                "mangomas",
                "--episodes",
                "12",
                "--collection-policy",
                "mcts",
                "--scenario",
                "patrol",
                "--scenario",
                "escort",
                "--mangomas-config",
                "configs/mangomas/default.toml",
                "--pipeline-run-name",
                "drone-smoke",
                "--pipeline-output-root",
                "artifacts/custom",
                "--collection-report-path",
                "artifacts/custom/collection.json",
            ]
        )
        assert args.agent == "mangomas"
        assert args.episodes == 12
        assert args.collection_policy == "mcts"
        assert args.scenario == ["patrol", "escort"]
        assert args.mangomas_config == ["configs/mangomas/default.toml"]
        assert args.pipeline_run_name == "drone-smoke"
        assert args.pipeline_output_root == "artifacts/custom"
        assert args.collection_report_path == "artifacts/custom/collection.json"

    def test_mangomas_collect_args(self) -> None:
        """Collection-only MangoMAS arguments are parsed correctly."""
        args = parse_args(
            [
                "--agent",
                "mangomas-collect",
                "--episodes",
                "8",
                "--scenario",
                "patrol",
            ]
        )
        assert args.agent == "mangomas-collect"
        assert args.episodes == 8
        assert args.scenario == ["patrol"]
        assert args.collection_policy == "random"

    def test_invalid_agent_raises(self) -> None:
        """Invalid agent type raises SystemExit."""
        with pytest.raises(SystemExit):
            parse_args(["--agent", "nonexistent"])
