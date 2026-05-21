"""Local LM Studio demo runner.

Usage:
    # Ping the configured LM Studio endpoint and print round-trip latency.
    python scripts/run_lmstudio_demo.py --check

    # Use an explicit preset.
    python scripts/run_lmstudio_demo.py --check \\
        --config configs/cognitive/gemma_e4b_teacher.toml

All numeric/string values flow through MangoMASBridgeConfig — no hard-coded
model ids, URLs, or timeouts live in this script. For full collection runs
(scenario manifests, episode loops, trace shards) use scripts/train.py with
``--collection-policy llm --mangomas-config <preset.toml>``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from forge.cognitive import CompletionConfig, create_provider
from forge.mangomas.config import MangoMASBridgeConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

    from forge.cognitive.providers import CognitiveProvider
    from forge.mangomas.config import TeacherConfig

logger = logging.getLogger("run_lmstudio_demo")

#: Demo helper default. Points at the gemma preset because that is the
#: ``MangoMASBridgeConfig``-shape TOML this script expects (``[teacher]``
#: section). ``configs/cognitive/default.toml`` uses a different
#: ``[cognitive]`` schema consumed by the cognitive provider factory, not
#: by ``MangoMASBridgeConfig.from_toml`` — pointing at it would silently
#: load an empty TeacherConfig and ping the mock provider. Override with
#: ``--config`` to use any other ``[teacher]``-shaped preset (e.g. qwen14b).
DEFAULT_CONFIG_PATH = Path("configs/cognitive/gemma_e4b_teacher.toml")
# Single source for the ping-call max_tokens — kept tiny so the helper is fast.
PING_MAX_TOKENS: int = 64
PING_PROMPT: str = 'Reply with the JSON object {"ok": true}. No prose.'


def _build_provider(teacher_cfg: TeacherConfig) -> CognitiveProvider:
    """Build a CognitiveProvider from a TeacherConfig — no inline literals."""
    logger.debug(
        "_build_provider provider=%s base_url=%s model=%s",
        teacher_cfg.provider,
        teacher_cfg.base_url,
        teacher_cfg.model,
    )
    provider: CognitiveProvider = create_provider(
        teacher_cfg.provider,
        base_url=teacher_cfg.base_url,
        model=teacher_cfg.model,
        timeout_secs=teacher_cfg.timeout_secs,
        max_retries=teacher_cfg.max_retries,
        retry_backoff_secs=teacher_cfg.retry_backoff_secs,
    )
    return provider


def _ping(provider: CognitiveProvider, teacher_cfg: TeacherConfig) -> int:
    logger.info(
        "ping provider=%s base_url=%s model=%s",
        teacher_cfg.provider,
        teacher_cfg.base_url,
        teacher_cfg.model,
    )
    try:
        resp = provider.complete(
            PING_PROMPT,
            CompletionConfig(
                model=teacher_cfg.model,
                temperature=teacher_cfg.temperature,
                max_tokens=PING_MAX_TOKENS,
                seed=teacher_cfg.seed,
                timeout_secs=teacher_cfg.timeout_secs,
            ),
        )
    except Exception:
        # `logger.exception` records the traceback at ERROR level so connection
        # / auth / serialisation issues are immediately diagnosable from the
        # log line (`logger.error("err=%s", exc)` would have dropped it).
        logger.exception("ping failed")
        return 1
    logger.info(
        "ping ok tokens_in=%d tokens_out=%d latency_ms=%.1f",
        resp.input_tokens,
        resp.output_tokens,
        resp.latency_ms,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_lmstudio_demo",
        description="Smoke-test a local LM Studio instance against a preset.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=(
            f"MangoMASBridgeConfig TOML preset (defaults to {DEFAULT_CONFIG_PATH}). "
            "Must contain a [teacher] section."
        ),
    )
    parser.add_argument("--check", action="store_true", help="Ping the endpoint and print latency.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = parser.parse_args(argv)

    level = getattr(logging, args.log_level)
    # Configure stdlib root once; do NOT use force=True because it removes
    # handlers attached by pytest's caplog fixture and tests would see no
    # records. Setting the named logger's level directly is sufficient
    # whether or not basicConfig has already been called.
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    logger.setLevel(level)
    logger.debug("argv parsed config=%s check=%s", args.config, args.check)

    if not args.check:
        parser.error("specify --check to ping the configured LM Studio endpoint")

    bridge_cfg = MangoMASBridgeConfig.from_toml(args.config)
    teacher_cfg = bridge_cfg.teacher
    logger.info(
        "loaded config path=%s provider=%s model=%s",
        args.config,
        teacher_cfg.provider,
        teacher_cfg.model,
    )

    provider = _build_provider(teacher_cfg)
    return _ping(provider, teacher_cfg)


if __name__ == "__main__":  # pragma: no cover - entry point
    sys.exit(main())
