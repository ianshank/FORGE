"""MCTS planning concept demonstration for FORGE.

Illustrates the core idea behind Monte Carlo Tree Search (MCTS) planning in
the context of a FORGE environment. The production MCTS implementation lives
in the forge-agent Rust crate, where it benefits from fast deterministic
simulation and zero-allocation stepping. This Python script demonstrates the
*concept* using simplified random rollouts.

Since we cannot cheaply clone or snapshot the full environment state from
Python, this demo uses a simplified approach:
  1. At each decision step, try every legal action once.
  2. For each candidate action, perform several random rollouts of fixed
     depth starting from the resulting state.
  3. Pick the action whose rollouts yield the highest average cumulative
     reward.
  4. Compare the planned policy against a purely random baseline.

For true MCTS with proper state cloning and tree reuse, use the Rust
forge-agent crate directly or call it through the Python bindings.
"""

import argparse
import time

import numpy as np

try:
    from forge_env.gymnasium_env import ForgeGymnasiumEnv
except ImportError:
    ForgeGymnasiumEnv = None
    print(
        "WARNING: Could not import ForgeGymnasiumEnv from forge_env.gymnasium_env.\n"
        "Make sure the forge-python crate is built and installed:\n"
        "  cd crates/forge-python && maturin develop\n"
    )


def simple_rollout(env, depth=10):
    """Execute a random rollout for a fixed number of steps.

    Takes random actions in the environment for up to *depth* steps and
    returns the cumulative reward collected during the rollout.

    Args:
        env: A Gymnasium-compatible environment (already stepped into a
            candidate action before calling this function).
        depth: Maximum number of random steps to take.

    Returns:
        The cumulative reward accumulated over the rollout.
    """
    cumulative_reward = 0.0
    for _ in range(depth):
        action = env.action_space.sample()
        _obs, reward, terminated, truncated, _info = env.step(action)
        cumulative_reward += reward
        if terminated or truncated:
            break
    return cumulative_reward


def plan_action(env, n_simulations=50, depth=10):
    """Select the best immediate action via random-rollout planning.

    For each possible action in the environment's discrete action space,
    perform *n_simulations* random rollouts of the given *depth*. The action
    with the highest mean cumulative reward across rollouts is returned.

    Note: Because we cannot clone the Python environment state, each
    simulation resets the environment to a fresh episode. This means the
    rollouts do not branch from the *current* state -- they illustrate the
    planning concept without full state cloning. For accurate lookahead,
    use the Rust forge-agent MCTS implementation.

    Args:
        env: A Gymnasium-compatible FORGE environment.
        n_simulations: Number of rollout simulations per candidate action.
        depth: Rollout horizon (number of steps to simulate).

    Returns:
        The integer action with the highest average rollout reward.
    """
    num_actions = env.action_space.n
    action_values = np.zeros(num_actions)

    for action in range(num_actions):
        rollout_rewards = []
        for _ in range(n_simulations):
            # Reset to get a consistent starting point for each rollout
            env.reset()
            # Take the candidate action
            _obs, immediate_reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                rollout_rewards.append(immediate_reward)
                continue
            # Continue with a random rollout
            rollout_reward = immediate_reward + simple_rollout(env, depth=depth - 1)
            rollout_rewards.append(rollout_reward)

        action_values[action] = np.mean(rollout_rewards)

    return int(np.argmax(action_values))


def run_planned_agent(env, num_steps, n_simulations, depth):
    """Run the planning agent for a given number of steps.

    Args:
        env: A Gymnasium-compatible FORGE environment.
        num_steps: Number of decision steps to execute.
        n_simulations: Simulations per action in the planner.
        depth: Rollout depth for each simulation.

    Returns:
        A list of per-step rewards collected by the planned agent.
    """
    _obs, _info = env.reset()
    rewards = []

    for step in range(1, num_steps + 1):
        action = plan_action(env, n_simulations=n_simulations, depth=depth)

        # After planning, reset and replay to maintain consistent state
        # (This is the simplified-demo workaround for lacking env cloning.)
        _obs, _info = env.reset()
        _obs, reward, terminated, truncated, _info = env.step(action)
        rewards.append(reward)

        if step % 50 == 0:
            mean_r = np.mean(rewards[-50:])
            print(f"  [Planned] Step {step}: last-50 mean reward = {mean_r:.4f}")

        if terminated or truncated:
            _obs, _info = env.reset()

    return rewards


def run_random_agent(env, num_steps):
    """Run a purely random agent for comparison.

    Args:
        env: A Gymnasium-compatible FORGE environment.
        num_steps: Number of steps to execute.

    Returns:
        A list of per-step rewards collected by the random agent.
    """
    _obs, _info = env.reset()
    rewards = []

    for step in range(1, num_steps + 1):
        action = env.action_space.sample()
        _obs, reward, terminated, truncated, _info = env.step(action)
        rewards.append(reward)

        if step % 50 == 0:
            mean_r = np.mean(rewards[-50:])
            print(f"  [Random]  Step {step}: last-50 mean reward = {mean_r:.4f}")

        if terminated or truncated:
            _obs, _info = env.reset()

    return rewards


def main(n_simulations=50, depth=10, num_steps=200):
    """Run the MCTS concept demo comparing planned vs random agents.

    Args:
        n_simulations: Number of rollout simulations per candidate action.
        depth: Rollout horizon per simulation.
        num_steps: Total decision steps for each agent.
    """
    if ForgeGymnasiumEnv is None:
        print("ForgeGymnasiumEnv is not available. Exiting.")
        return

    config = {
        "world": {
            "width": 32,
            "height": 32,
        },
        "agents": {
            "num_agents": 1,
        },
    }

    print("=" * 60)
    print("MCTS Planning Concept Demo")
    print("=" * 60)
    print(f"  Simulations per action : {n_simulations}")
    print(f"  Rollout depth          : {depth}")
    print(f"  Decision steps         : {num_steps}")
    print()
    print(
        "NOTE: The production MCTS implementation lives in the Rust\n"
        "forge-agent crate. This script demonstrates the concept using\n"
        "simplified random rollouts from Python.\n"
    )

    # --- Planned agent ------------------------------------------------------
    print("Running planned agent...")
    env_planned = ForgeGymnasiumEnv(config=config, seed=42)
    t0 = time.time()
    planned_rewards = run_planned_agent(env_planned, num_steps, n_simulations, depth)
    planned_time = time.time() - t0
    env_planned.close()

    # --- Random baseline ----------------------------------------------------
    print("\nRunning random baseline...")
    env_random = ForgeGymnasiumEnv(config=config, seed=42)
    t0 = time.time()
    random_rewards = run_random_agent(env_random, num_steps)
    random_time = time.time() - t0
    env_random.close()

    # --- Comparison ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print("  Planned agent:")
    print(f"    Total reward : {np.sum(planned_rewards):.3f}")
    print(f"    Mean reward  : {np.mean(planned_rewards):.4f}")
    print(f"    Std reward   : {np.std(planned_rewards):.4f}")
    print(f"    Wall time    : {planned_time:.2f}s")
    print()
    print("  Random baseline:")
    print(f"    Total reward : {np.sum(random_rewards):.3f}")
    print(f"    Mean reward  : {np.mean(random_rewards):.4f}")
    print(f"    Std reward   : {np.std(random_rewards):.4f}")
    print(f"    Wall time    : {random_time:.2f}s")
    print()

    improvement = np.sum(planned_rewards) - np.sum(random_rewards)
    print(f"  Planned vs Random improvement: {improvement:+.3f} total reward")
    if planned_time > 0:
        print(
            f"  Planning overhead: {planned_time / max(random_time, 1e-9):.1f}x slower than random"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "MCTS planning concept demo for FORGE. Compares a simplified "
            "rollout-based planner against a random baseline."
        )
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=50,
        help="Number of rollout simulations per candidate action (default: 50).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=10,
        help="Rollout depth (number of steps to simulate ahead, default: 10).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Total decision steps for each agent (default: 200).",
    )
    args = parser.parse_args()

    main(
        n_simulations=args.simulations,
        depth=args.depth,
        num_steps=args.steps,
    )
