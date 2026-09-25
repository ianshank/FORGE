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

    def test_collection_policy_skill_is_accepted(self) -> None:
        args = parse_args(["--collection-policy", "skill"])
        assert args.collection_policy == "skill"

    def test_invalid_collection_policy_raises(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["--collection-policy", "not-a-policy"])

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


class TestDeviceFlag:
    """`--device` overrides [hardware] device and fails fast for mappo."""

    def test_device_defaults_to_none(self) -> None:
        """Absent flag must not clobber the config's [hardware] device."""
        assert parse_args([]).device is None

    def test_device_flag_parsed(self) -> None:
        assert parse_args(["--device", "cuda:0"]).device == "cuda:0"

    @pytest.mark.parametrize("device", ["bogus", "cuda:x"])
    def test_unusable_device_exits_before_env_creation(
        self, device: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unusable device must exit 1 before the env is built."""
        import train

        def _no_env(_config: object) -> None:
            raise AssertionError("env must not be created for an unusable device")

        monkeypatch.setattr(train, "_create_env", _no_env)
        with pytest.raises(SystemExit) as excinfo:
            train.main(["--agent", "mappo", "--config", "configs/dry_run.toml", "--device", device])
        assert excinfo.value.code == 1

    def test_device_flag_overrides_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """--device lands on config.hardware.device, which _train_mappo reads."""
        import train

        seen: dict[str, str] = {}

        class _Env:
            def close(self) -> None:
                pass

        def _capture(_env: object, config: object, _args: object) -> None:
            seen["device"] = config.hardware.device  # type: ignore[attr-defined]

        monkeypatch.setattr(train, "_create_env", lambda _config: _Env())
        monkeypatch.setattr(train, "_train_mappo", _capture)
        train.main(["--agent", "mappo", "--config", "configs/dry_run.toml", "--device", "cpu"])
        assert seen == {"device": "cpu"}


class TestFlatObsAgent:
    """Evaluator passes dict obs; trained agents expect flatten_obs vectors."""

    def test_flattens_dict_obs_in_sorted_key_order(self) -> None:
        import numpy as np
        from train import _FlatObsAgent

        seen: list[object] = []

        class _Recorder:
            def act(self, obs: object) -> tuple[int, None]:
                seen.append(obs)
                return 0, None

        assert _FlatObsAgent(_Recorder()).act({"b": [2.0, 3.0], "a": 1.0}) == (0, None)
        np.testing.assert_array_equal(seen[0], np.array([1.0, 2.0, 3.0], dtype=np.float32))

    def test_passes_flat_obs_through(self) -> None:
        from train import _FlatObsAgent

        class _Echo:
            def act(self, obs: object) -> object:
                return obs

        flat = [0.5, 1.5]
        assert _FlatObsAgent(_Echo()).act(flat) is flat


def test_mappo_with_eval_interval_runs_end_to_end(tmp_path: object) -> None:
    """Regression: `--agent mappo --eval-interval N` crashed on the first
    evaluation (`TypeError: must be real number, not dict`)."""
    pytest.importorskip("torch")
    pytest.importorskip("gymnasium")
    pytest.importorskip("forge_env.forge_env")
    import train

    train.main(
        [
            "--agent", "mappo",
            "--config", "configs/dry_run.toml",
            "--dry-run",
            "--num-updates", "1",
            "--eval-interval", "1",
            "--eval-episodes", "1",
            "--device", "cpu",
            "--checkpoint-dir", str(tmp_path),
        ]
    )  # fmt: skip
    assert any(tmp_path.iterdir())  # type: ignore[attr-defined]
