"""Smoke tests for the new --collection-policy llm CLI surface in train.py."""

from __future__ import annotations

from pathlib import Path

import pytest

# scripts/ is placed on sys.path by the root conftest.py (_ensure_importable).
import train

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_cli_accepts_llm_policy_choice() -> None:
    # We use train._COLLECTION_POLICY_CHOICES indirectly via parse_args.
    args = train.parse_args(
        ["--agent", "mangomas-collect", "--collection-policy", "llm", "--episodes", "1"]
    )
    assert args.collection_policy == "llm"
    assert args.teacher_concurrency is None


def test_cli_teacher_overrides_threaded(tmp_path: Path) -> None:
    from forge.mangomas.config import DEFAULT_TEACHER_BASE_URL, MangoMASBridgeConfig

    args = train.parse_args(
        [
            "--agent",
            "mangomas-collect",
            "--collection-policy",
            "llm",
            "--episodes",
            "1",
            "--teacher-model",
            "qwen2.5-14b-instruct",
            "--teacher-base-url",
            DEFAULT_TEACHER_BASE_URL,
            "--teacher-concurrency",
            "8",
            "--teacher-output-root",
            str(tmp_path),
        ]
    )

    bridge = MangoMASBridgeConfig()
    train._apply_mangomas_cli_overrides(bridge, args)
    assert bridge.teacher.model == "qwen2.5-14b-instruct"
    assert bridge.teacher.base_url == DEFAULT_TEACHER_BASE_URL
    assert bridge.teacher.concurrency == 8
    assert bridge.teacher.output_root == str(tmp_path)


def test_cli_teacher_config_loads_preset(tmp_path: Path) -> None:
    preset_path = (
        _REPO_ROOT / "configs" / "cognitive" / "qwen14b_teacher.toml"
    )
    if not preset_path.exists():
        pytest.skip("preset TOML missing")
    args = train.parse_args(
        [
            "--agent",
            "mangomas-collect",
            "--collection-policy",
            "llm",
            "--episodes",
            "1",
            "--teacher-config",
            str(preset_path),
        ]
    )

    from forge.mangomas.config import MangoMASBridgeConfig

    bridge = MangoMASBridgeConfig()
    train._apply_mangomas_cli_overrides(bridge, args)
    assert bridge.teacher.provider == "lmstudio"
    assert bridge.teacher.model == "qwen2.5-14b-instruct"


def test_cli_random_policy_unaffected_by_new_flags() -> None:
    """Regression: random policy still parses with default flag values."""
    args = train.parse_args(
        ["--agent", "mangomas-collect", "--collection-policy", "random", "--episodes", "1"]
    )
    assert args.collection_policy == "random"
    assert args.teacher_model is None
    assert args.teacher_concurrency is None


def test_cli_omits_bc_train_flag() -> None:
    """Regression: --bc-train-after-collect was removed (was inert).

    Forces a parser error because the flag no longer exists. The BC stage
    decision lives in MangoMASPipeline._run_bc_stage and keys off the
    presence of teacher data in CollectedTrainingData.
    """
    import pytest

    with pytest.raises(SystemExit):
        train.parse_args(
            [
                "--agent",
                "mangomas-collect",
                "--collection-policy",
                "llm",
                "--episodes",
                "1",
                "--bc-train-after-collect",
            ]
        )
