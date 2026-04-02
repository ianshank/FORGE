#!/usr/bin/env python3
"""End-to-end dry-run pipeline: train all agents, evaluate, report.

Uses REAL native environments when available, RealisticFakeEnv otherwise. NO MagicMock.

Usage::
    python scripts/run_e2e_dry_run.py
    python scripts/run_e2e_dry_run.py --config configs/dry_run.toml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from forge.agents.base_agent import AgentConfig
from forge.agents.random_agent import RandomAgent
from forge.config import DryRunConfig, ForgeConfig
from forge.evaluation.evaluator import EvalConfig, Evaluator
from forge.testing.env_factory import NATIVE_AVAILABLE, FakeEnvConfig, create_env

logger = logging.getLogger("forge.e2e")

_AGENT_FACTORIES: dict[str, Any] = {
    "random": lambda cfg, n: RandomAgent(cfg, action_space_size=n),
}


def run_e2e(config: ForgeConfig) -> dict[str, Any]:
    """Run full E2E pipeline. Returns results dict."""
    effective = config.effective_simulation()
    logger.info(
        "E2E pipeline: grid=%d, max_ep_len=%d, episodes=%d, native=%s",
        effective.grid_size,
        effective.max_episode_length,
        config.dry_run.max_episodes,
        NATIVE_AVAILABLE,
    )

    fake_cfg = FakeEnvConfig(
        max_episode_length=effective.max_episode_length,
        seed=config.dry_run.seed,
    )
    try:
        env = create_env(
            config=fake_cfg,
            native_config=config.to_rust_config() if NATIVE_AVAILABLE else None,
        )
    except (ImportError, RuntimeError) as exc:
        logger.warning("Native env unavailable (%s) — falling back to RealisticFakeEnv", exc)
        env = create_env(config=fake_cfg, force_fake=True)
    action_n = env.action_space.n

    results: dict[str, Any] = {"native_env": NATIVE_AVAILABLE}
    for agent_name, factory in _AGENT_FACTORIES.items():
        logger.info("--- Evaluating agent: %s ---", agent_name)
        agent_config = AgentConfig(name=agent_name)
        agent = factory(agent_config, action_n)

        evaluator = Evaluator(
            EvalConfig(
                num_episodes=config.dry_run.max_episodes,
                seed=config.dry_run.seed,
                log_per_episode=True,
            )
        )
        result = evaluator.evaluate(env, agent)
        results[agent_name] = result.to_dict()
        logger.info(
            "%s: reward=%.2f+/-%.2f, throughput=%.0f steps/s",
            agent_name,
            result.reward_mean,
            result.reward_std,
            result.steps_per_second,
        )

    return results


def main() -> None:
    """Entrypoint for the E2E dry-run pipeline."""
    parser = argparse.ArgumentParser(description="E2E dry-run pipeline")
    parser.add_argument("--config", default="configs/dry_run.toml")
    parser.add_argument("--output", default="e2e_results.json")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = ForgeConfig.from_file(args.config)
    if not config.dry_run.enabled:
        config = ForgeConfig(
            hardware=config.hardware,
            simulation=config.simulation,
            training=config.training,
            dry_run=DryRunConfig(enabled=True),
        )

    t0 = time.perf_counter()
    results = run_e2e(config)
    elapsed = time.perf_counter() - t0

    output = {"results": results, "elapsed_seconds": elapsed}
    Path(args.output).write_text(json.dumps(output, indent=2))
    logger.info("E2E complete in %.1fs. Results: %s", elapsed, args.output)


if __name__ == "__main__":
    main()
