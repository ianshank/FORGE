"""Basic navigation demo for the FORGE environment.

Demonstrates the simplest possible interaction with FORGE: creating a small
world, taking random actions, and observing how the agent's position and
reward change over time. This is a good starting point for understanding the
Gymnasium-style API that FORGE exposes.

Actions used (from the discrete action space):
    0: Noop
    1: Move Up
    2: Move Down
    3: Move Left
    4: Move Right
    5: Pick Up
    31: Interact

Usage:
    python basic_navigation.py
    python basic_navigation.py --steps 500 --seed 42 --world-size 64
"""

import argparse
import random
import sys

try:
    from forge_env import ForgeEnv
except ImportError:
    print(
        "ERROR: forge_env native module not found.\n"
        "Build and install the FORGE Python bindings first:\n"
        "    cd /path/to/FORGE && pip install -e . \n"
        "    (requires maturin: pip install maturin)"
    )
    sys.exit(1)


# Human-readable names for the basic movement actions.
ACTION_NAMES = {
    0: "Noop",
    1: "Move Up",
    2: "Move Down",
    3: "Move Left",
    4: "Move Right",
    5: "Pick Up",
    31: "Interact",
}

# Only use movement and noop actions for this navigation demo.
NAVIGATION_ACTIONS = [0, 1, 2, 3, 4]


def parse_args():
    parser = argparse.ArgumentParser(
        description="FORGE basic navigation demo: random walk in a small world."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of environment steps to run (default: 200).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=32,
        help="Width and height of the world grid (default: 32).",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=50,
        help="Print status every N steps (default: 50).",
    )
    return parser.parse_args()


def run_navigation(steps, seed, world_size, print_every):
    """Run a random navigation loop and collect statistics."""

    # Configure a small world with default physics.
    config = {
        "world": {
            "width": world_size,
            "height": world_size,
        },
        "agents": {
            "num_agents": 1,
        },
        "task": {
            "max_episode_length": steps + 100,  # allow room beyond our step count
            "dense_rewards": True,
        },
    }

    print(f"Creating ForgeEnv with {world_size}x{world_size} world ...")
    env = ForgeEnv(config=config)

    print(f"Resetting environment with seed={seed} ...")
    obs, info = env.reset(seed=seed)

    # Seed Python's RNG for reproducible action selection.
    rng = random.Random(seed)

    total_reward = 0.0
    episode_count = 0
    steps_taken = 0
    start_position = _extract_position(obs)

    print(f"Starting position: {start_position}")
    print(f"Running {steps} steps ...\n")

    for step_idx in range(1, steps + 1):
        # Pick a random navigation action.
        action = rng.choice(NAVIGATION_ACTIONS)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps_taken += 1

        # Periodic status report.
        if step_idx % print_every == 0:
            pos = _extract_position(obs)
            health = _extract_scalar(obs, "health")
            stamina = _extract_scalar(obs, "stamina")
            tick = info.get("tick", step_idx) if isinstance(info, dict) else step_idx
            print(
                f"  Step {step_idx:>5d} | "
                f"Position: ({pos[0]:>3d}, {pos[1]:>3d}) | "
                f"Reward: {reward:>+7.3f} | "
                f"Total: {total_reward:>+8.3f} | "
                f"Health: {health:.2f} | "
                f"Stamina: {stamina:.2f} | "
                f"Tick: {tick}"
            )

        # Handle episode boundaries.
        if terminated or truncated:
            episode_count += 1
            reason = "terminated" if terminated else "truncated"
            print(f"\n  >> Episode ended ({reason}) at step {step_idx}. Resetting ...")
            obs, info = env.reset(seed=seed + episode_count)

    # Final summary.
    final_position = _extract_position(obs)
    print("\n" + "=" * 60)
    print("Navigation Demo Complete")
    print("=" * 60)
    print(f"  Total steps taken : {steps_taken}")
    print(f"  Episodes completed: {episode_count}")
    print(f"  Total reward      : {total_reward:+.4f}")
    print(f"  Average reward    : {total_reward / max(steps_taken, 1):+.6f}")
    print(f"  Start position    : {start_position}")
    print(f"  Final position    : {final_position}")
    print(f"  World size        : {world_size}x{world_size}")
    print(f"  Seed              : {seed}")

    env.close()
    print("\nEnvironment closed.")


def _extract_position(obs):
    """Extract (x, y) position from the observation dict."""
    if isinstance(obs, dict):
        pos = obs.get("position", (0, 0))
        # The position may be a numpy array or a tuple.
        try:
            return (int(pos[0]), int(pos[1]))
        except (IndexError, TypeError):
            return (0, 0)
    return (0, 0)


def _extract_scalar(obs, key):
    """Extract a scalar float from the observation dict."""
    if isinstance(obs, dict):
        val = obs.get(key, 0.0)
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


if __name__ == "__main__":
    args = parse_args()
    run_navigation(
        steps=args.steps,
        seed=args.seed,
        world_size=args.world_size,
        print_every=args.print_every,
    )
