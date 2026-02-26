"""Multi-agent cooperative scenario for FORGE.

Demonstrates how to set up and run a multi-agent cooperative environment
using the PettingZoo-compatible parallel API provided by FORGE. Each agent
takes random actions while we track per-agent rewards and positions over
time. This serves as a starting point for building cooperative multi-agent
reinforcement learning experiments.
"""

import argparse
import random

import numpy as np

try:
    from forge_env.pettingzoo_env import ForgeParallelEnv
except ImportError:
    ForgeParallelEnv = None
    print(
        "WARNING: Could not import ForgeParallelEnv from forge_env.pettingzoo_env.\n"
        "Make sure the forge-python crate is built and installed:\n"
        "  cd crates/forge-python && maturin develop\n"
    )


def run_cooperative_scenario(num_steps=100, num_agents=2):
    """Run a multi-agent cooperative scenario with random actions.

    Args:
        num_steps: Number of environment steps to run.
        num_agents: Number of cooperative agents in the environment.
    """
    if ForgeParallelEnv is None:
        print("ForgeParallelEnv is not available. Exiting.")
        return

    config = {
        "world": {
            "width": 32,
            "height": 32,
        },
        "agents": {
            "num_agents": num_agents,
            "comm_vocab_size": 8,
        },
    }

    env = ForgeParallelEnv(config=config)
    observations, infos = env.reset()

    agent_ids = list(observations.keys())
    print(f"Environment created with {len(agent_ids)} agents: {agent_ids}")
    print(f"Running for {num_steps} steps...\n")

    total_rewards = {agent_id: 0.0 for agent_id in agent_ids}

    for step in range(1, num_steps + 1):
        # Sample random actions for each agent
        actions = {}
        for agent_id in agent_ids:
            action_space = env.action_space(agent_id)
            actions[agent_id] = action_space.sample()

        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Accumulate rewards
        for agent_id in agent_ids:
            total_rewards[agent_id] += rewards.get(agent_id, 0.0)

        # Report every 20 steps
        if step % 20 == 0:
            print(f"--- Step {step} ---")
            for agent_id in agent_ids:
                reward = rewards.get(agent_id, 0.0)
                position = infos.get(agent_id, {}).get("position", "unknown")
                print(
                    f"  Agent '{agent_id}': "
                    f"reward={reward:.3f}, "
                    f"cumulative={total_rewards[agent_id]:.3f}, "
                    f"position={position}"
                )
            print()

        # Check if all agents are done
        if all(terminations.get(a, False) or truncations.get(a, False) for a in agent_ids):
            print(f"All agents finished at step {step}.")
            break

    env.close()

    # Print summary
    print("=" * 50)
    print("Summary")
    print("=" * 50)
    for agent_id in agent_ids:
        print(f"  Agent '{agent_id}': total reward = {total_rewards[agent_id]:.3f}")
    mean_reward = np.mean(list(total_rewards.values()))
    print(f"  Mean reward across agents: {mean_reward:.3f}")
    print(f"  Steps completed: {step}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a multi-agent cooperative scenario in FORGE."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of environment steps to run (default: 100).",
    )
    parser.add_argument(
        "--num_agents",
        type=int,
        default=2,
        help="Number of cooperative agents (default: 2).",
    )
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    run_cooperative_scenario(num_steps=args.steps, num_agents=args.num_agents)
