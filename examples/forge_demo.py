"""FORGE Interactive Demo — showcases all major capabilities.

A unified, visually compelling demonstration of the FORGE simulation platform.
Runs through 8 progressive scenarios with ASCII world rendering, demonstrating
world generation, navigation, resource gathering, crafting, multi-agent
cooperation, day/night cycle, determinism, and raw performance.

Usage:
    python forge_demo.py                       # Full demo with animation
    python forge_demo.py --quick               # Fast mode (no delays, for CI)
    python forge_demo.py --section navigation  # Run one section only
    python forge_demo.py --no-color            # Disable ANSI colors
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from typing import Any

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy is required. Install with: pip install numpy")
    sys.exit(1)

try:
    from forge_env import ForgeEnv
except ImportError:
    print(
        "ERROR: forge_env native module not found.\n"
        "Build and install the FORGE Python bindings first:\n"
        "    maturin build --release -m crates/forge-python/Cargo.toml\n"
        "    pip install target/wheels/*.whl"
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTION_NOOP = 0
ACTION_MOVE_UP = 1
ACTION_MOVE_DOWN = 2
ACTION_MOVE_LEFT = 3
ACTION_MOVE_RIGHT = 4
ACTION_PICKUP = 5
ACTION_CRAFT_BASE = 26

MOVE_ACTIONS = [ACTION_MOVE_UP, ACTION_MOVE_DOWN, ACTION_MOVE_LEFT, ACTION_MOVE_RIGHT]

ITEM_NAMES: dict[int, str] = {
    0: "Wood",
    1: "Stone",
    2: "Ore",
    3: "Fish",
    4: "Fiber",
    5: "Clay",
    10: "Axe",
    11: "Pickaxe",
    12: "Sword",
    13: "Shield",
    14: "Plank",
    15: "Bridge",
    16: "Rope",
    17: "Brick",
    18: "Key",
    19: "Torch",
    30: "CookedFish",
    31: "Bread",
}

DAY_PHASE_NAMES = {0: "Dawn", 1: "Day", 2: "Dusk", 3: "Night"}

TERRAIN_LEGEND = (
    ". Ground  ~ Water  # Wall  T Forest  M Mountain  "
    "S Sand  I Ice  L Lava  A Agent  R Resource  O Object"
)

RESET = "\033[0m"
TERRAIN_COLORS: dict[str, str] = {
    ".": "\033[37m",  # white
    "~": "\033[34m",  # blue
    "#": "\033[90m",  # dark gray
    "T": "\033[32m",  # green
    "M": "\033[33m",  # yellow
    "S": "\033[93m",  # bright yellow
    "I": "\033[96m",  # cyan
    "L": "\033[91m",  # red
    "A": "\033[1;91m",  # bold red
    "R": "\033[95m",  # magenta
    "O": "\033[93m",  # bright yellow
}

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_use_color = True
_quick_mode = False

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def print_header(title: str) -> None:
    """Print a prominent section header."""
    width = 60
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)
    print()


def print_subheader(title: str) -> None:
    """Print a subsection header."""
    print(f"--- {title} ---")


def colorize_grid(grid_str: str) -> str:
    """Add ANSI color codes to ASCII grid characters."""
    if not _use_color:
        return grid_str
    result: list[str] = []
    for ch in grid_str:
        color = TERRAIN_COLORS.get(ch)
        if color:
            result.append(f"{color}{ch}{RESET}")
        else:
            result.append(ch)
    return "".join(result)


def print_grid(grid_str: str, label: str = "", max_width: int = 0) -> None:
    """Print an ASCII grid, optionally cropped to max_width columns."""
    lines = grid_str.strip().split("\n")
    if max_width > 0:
        lines = [line[:max_width] for line in lines]
    if label:
        print(f"  {label}:")
    for line in lines:
        print(f"    {colorize_grid(line)}")


def print_grids_side_by_side(
    grids: list[tuple[str, str]],
    spacing: int = 3,
) -> None:
    """Print multiple grids horizontally with labels."""
    # grids is a list of (label, grid_str) tuples
    parsed = []
    for label, grid_str in grids:
        lines = grid_str.strip().split("\n")
        width = max(len(line) for line in lines) if lines else 0
        parsed.append((label, lines, width))

    # Print labels
    label_line = ""
    for label, _lines, width in parsed:
        label_line += f"  {label:<{width}}" + " " * spacing
    print(label_line)

    # Print grid rows
    max_rows = max(len(lines) for _, lines, _ in parsed)
    for row in range(max_rows):
        row_str = ""
        for _, lines, width in parsed:
            line = lines[row] if row < len(lines) else ""
            row_str += f"  {colorize_grid(line):<{width + (len(colorize_grid(line)) - len(line))}}"
            row_str += " " * spacing
        # Simpler approach: print each grid's row with padding
        parts: list[str] = []
        for _, lines, width in parsed:
            line = lines[row] if row < len(lines) else ""
            padded = line + " " * (width - len(line))
            parts.append(colorize_grid(padded))
        print("  " + (" " * spacing).join(parts))
    print()


def pause(seconds: float = 0.5) -> None:
    """Sleep unless in quick mode."""
    if not _quick_mode:
        time.sleep(seconds)


def extract_position(obs: dict[str, Any]) -> tuple[int, int]:
    """Extract (x, y) position from observation dict."""
    pos = obs.get("position", (0, 0))
    try:
        return (int(pos[0]), int(pos[1]))
    except (IndexError, TypeError):
        return (0, 0)


def extract_inventory(obs: dict[str, Any]) -> dict[int, int]:
    """Parse inventory from observation dict into {item_id: count}."""
    inv: dict[int, int] = {}
    inv_data = obs.get("inventory")
    if inv_data is None:
        return inv
    try:
        for slot_idx in range(len(inv_data)):
            item_type = int(inv_data[slot_idx][0])
            count = int(inv_data[slot_idx][1])
            if item_type != 255 and count > 0:
                inv[item_type] = inv.get(item_type, 0) + count
    except (IndexError, TypeError, ValueError):
        pass
    return inv


def format_inventory(inv: dict[int, int]) -> str:
    """Format an inventory dict as a human-readable string."""
    if not inv:
        return "(empty)"
    parts = []
    for item_id, count in sorted(inv.items()):
        name = ITEM_NAMES.get(item_id, f"Item#{item_id}")
        parts.append(f"{name} x{count}")
    return ", ".join(parts)


def obs_hash(obs: dict[str, Any]) -> str:
    """Compute a deterministic hash of observation data."""
    h = hashlib.sha256()
    for key in sorted(obs.keys()):
        val = obs[key]
        if isinstance(val, np.ndarray):
            h.update(val.tobytes())
        else:
            h.update(str(val).encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Demo sections
# ---------------------------------------------------------------------------


def demo_world_generation(seed: int) -> bool:
    """Section 1: Procedural world generation with different seeds."""
    print_header("1. Procedural World Generation")

    print("FORGE generates unique worlds from seeds using Perlin noise.")
    print(f"Legend: {TERRAIN_LEGEND}")
    print()

    grids: list[tuple[str, str]] = []
    for s in [seed, seed + 100, seed + 200]:
        config = {"world": {"width": 20, "height": 12, "seed": s}}
        env = ForgeEnv(config=config)
        env.reset(seed=s)
        grid_str = env.render()
        grids.append((f"Seed {s}", grid_str))
        env.close()

    print_grids_side_by_side(grids)
    print("  Each seed produces a deterministic, unique world.")
    pause(1.0)
    return True


def demo_navigation(seed: int) -> bool:
    """Section 2: Animated agent navigation."""
    print_header("2. Agent Navigation")

    config = {"world": {"width": 16, "height": 12}}
    env = ForgeEnv(config=config)
    obs, _info = env.reset(seed=seed)

    pos = extract_position(obs)
    print(f"Agent starts at position {pos}")
    print("Moving: Up, Up, Right, Right, Down, Down, Left, Left\n")

    actions = [
        (ACTION_MOVE_UP, "Up"),
        (ACTION_MOVE_UP, "Up"),
        (ACTION_MOVE_RIGHT, "Right"),
        (ACTION_MOVE_RIGHT, "Right"),
        (ACTION_MOVE_DOWN, "Down"),
        (ACTION_MOVE_DOWN, "Down"),
        (ACTION_MOVE_LEFT, "Left"),
        (ACTION_MOVE_LEFT, "Left"),
    ]

    for action_id, action_name in actions:
        obs, _rew, _term, _trunc, info = env.step(action_id)
        pos = extract_position(obs)
        health = float(obs.get("health", 0))
        stamina = float(obs.get("stamina", 0))
        tick = info.get("tick", 0) if isinstance(info, dict) else 0

        print(
            f"  Move {action_name:<5s} -> pos=({pos[0]:>2d},{pos[1]:>2d})  "
            f"health={health:.2f}  stamina={stamina:.2f}  tick={tick}"
        )
        pause(0.2)

    print()
    print_grid(env.render(), "Final world state")
    env.close()
    pause(0.5)
    return True


def demo_resource_gathering(seed: int) -> bool:
    """Section 3: Resource gathering with inventory tracking."""
    print_header("3. Resource Gathering")

    config = {
        "world": {"width": 24, "height": 16, "resource_density": 0.8},
        "crafting": {"enabled": True},
    }
    env = ForgeEnv(config=config)
    obs, _info = env.reset(seed=seed)

    before_inv = extract_inventory(obs)
    print(f"Initial inventory: {format_inventory(before_inv)}")
    print()
    print("Strategy: move around and pick up resources at every step...")
    print()

    rng = __import__("random").Random(seed)
    pickup_count = 0

    for step in range(1, 61):
        # Alternate between move and pickup
        action = ACTION_PICKUP if step % 2 == 0 else rng.choice(MOVE_ACTIONS)

        obs, _rew, term, trunc, _info = env.step(action)

        if term or trunc:
            obs, _info = env.reset(seed=seed)
            continue

        current_inv = extract_inventory(obs)
        if current_inv != before_inv and action == ACTION_PICKUP:
            # Something was picked up
            for item_id, count in current_inv.items():
                old_count = before_inv.get(item_id, 0)
                if count > old_count:
                    name = ITEM_NAMES.get(item_id, f"Item#{item_id}")
                    pickup_count += 1
                    print(f"  [Step {step:>3d}] Picked up: {name} x{count - old_count}")
            before_inv = current_inv

    final_inv = extract_inventory(obs)
    print()
    print(f"Final inventory: {format_inventory(final_inv)}")
    print(f"Total pickups: {pickup_count}")
    print()
    print_grid(env.render(), "World after gathering")
    env.close()
    pause(0.5)
    return True


def demo_crafting(seed: int) -> bool:
    """Section 4: Crafting system demonstration."""
    print_header("4. Crafting System")

    print("Default recipes:")
    print("  Axe     : 2 Wood + 1 Stone  -> 1 Axe")
    print("  Pickaxe : 2 Wood + 2 Stone  -> 1 Pickaxe")
    print("  Plank   : 2 Wood            -> 2 Plank")
    print("  Rope    : 3 Fiber           -> 1 Rope")
    print("  Torch   : 1 Wood + 1 Fiber  -> 1 Torch")
    print()

    config = {
        "world": {"width": 24, "height": 16, "resource_density": 0.9},
        "crafting": {"enabled": True},
    }
    env = ForgeEnv(config=config)
    obs, _info = env.reset(seed=seed)

    # Gather resources first
    rng = __import__("random").Random(seed)
    for _ in range(100):
        action = ACTION_PICKUP if rng.random() < 0.4 else rng.choice(MOVE_ACTIONS)
        obs, _rew, term, trunc, _info = env.step(action)
        if term or trunc:
            obs, _info = env.reset(seed=seed)

    before_inv = extract_inventory(obs)
    print(f"After gathering: {format_inventory(before_inv)}")
    print()

    # Try crafting each recipe
    recipes = [(0, "Axe"), (2, "Plank"), (7, "Torch"), (5, "Rope"), (1, "Pickaxe")]
    for recipe_id, recipe_name in recipes:
        inv_before = extract_inventory(obs)
        obs, _rew, term, trunc, _info = env.step(ACTION_CRAFT_BASE + recipe_id)
        if term or trunc:
            break
        inv_after = extract_inventory(obs)

        if inv_after != inv_before:
            gained = {
                k: v - inv_before.get(k, 0)
                for k, v in inv_after.items()
                if v > inv_before.get(k, 0)
            }
            consumed = {
                k: v - inv_after.get(k, 0) for k, v in inv_before.items() if v > inv_after.get(k, 0)
            }
            gained_str = format_inventory(gained)
            consumed_str = format_inventory(consumed)
            print(f"  Crafted {recipe_name}: consumed [{consumed_str}] -> gained [{gained_str}]")
        else:
            print(f"  {recipe_name}: insufficient materials (need more resources)")

    final_inv = extract_inventory(obs)
    print()
    print(f"Final inventory: {format_inventory(final_inv)}")
    env.close()
    pause(0.5)
    return True


def demo_multi_agent(seed: int) -> bool:
    """Section 5: Multi-agent environment."""
    print_header("5. Multi-Agent Cooperation")

    try:
        from forge_env.pettingzoo_env import ForgeParallelEnv
    except ImportError:
        print("  ForgeParallelEnv not available, skipping.")
        return False

    config = {
        "world": {"width": 20, "height": 12},
        "agents": {"num_agents": 2, "comm_vocab_size": 8},
    }

    env = ForgeParallelEnv(config=config)
    observations, _infos = env.reset()
    agent_ids = list(observations.keys())

    print(f"Created environment with {len(agent_ids)} agents: {agent_ids}")
    print()

    rng = __import__("random").Random(seed)
    total_rewards: dict[str, float] = dict.fromkeys(agent_ids, 0.0)

    for step in range(1, 11):
        actions = {agent_id: rng.choice(MOVE_ACTIONS) for agent_id in agent_ids}
        observations, rewards, _terms, _truncs, infos = env.step(actions)

        for agent_id in agent_ids:
            total_rewards[agent_id] += rewards.get(agent_id, 0.0)

        if step % 5 == 0:
            print(f"  Step {step}:")
            for agent_id in agent_ids:
                pos = infos.get(agent_id, {}).get("position", "?")
                rew = rewards.get(agent_id, 0.0)
                print(
                    f"    {agent_id}: pos={pos}, reward={rew:+.3f}, total={total_rewards[agent_id]:+.3f}"
                )

    print()
    # Render the grid (through the inner env if available)
    try:
        grid_str = env.render()
        if grid_str:
            print_grid(grid_str, "Multi-agent world")
    except (AttributeError, TypeError):
        pass

    print("  Summary:")
    for agent_id in agent_ids:
        print(f"    {agent_id}: total reward = {total_rewards[agent_id]:+.4f}")

    env.close()
    pause(0.5)
    return True


def demo_day_night(seed: int) -> bool:
    """Section 6: Day/night cycle."""
    print_header("6. Day/Night Cycle")

    config = {
        "world": {"width": 16, "height": 10, "day_night_cycle_length": 20},
        "agents": {"default_vision_radius": 4},
    }
    env = ForgeEnv(config=config)
    obs, info = env.reset(seed=seed)

    print("Stepping through the day/night cycle (cycle length = 20 ticks):")
    print()

    prev_phase = -1
    for step in range(1, 81):
        obs, _rew, term, trunc, info = env.step(ACTION_NOOP)
        if term or trunc:
            obs, info = env.reset(seed=seed)

        tick = info.get("tick", step) if isinstance(info, dict) else step
        phase = info.get("day_phase", obs.get("day_phase", 0))
        phase = int(phase) if not isinstance(phase, int) else phase
        phase_name = DAY_PHASE_NAMES.get(phase, f"Phase {phase}")

        if phase != prev_phase:
            bar = "#" * (step % 20 or 20)
            print(f"  Tick {tick:>3d}: {phase_name:<6s}  [{bar:<20s}]")
            prev_phase = phase
            pause(0.3)

    print()
    print("  The cycle repeats: Dawn -> Day -> Dusk -> Night -> Dawn ...")
    env.close()
    pause(0.5)
    return True


def demo_determinism(seed: int) -> bool:
    """Section 7: Determinism proof."""
    print_header("7. Deterministic Simulation")

    print("Same seed + same actions = byte-identical results.")
    print()

    config = {"world": {"width": 32, "height": 32}}
    num_steps = 100
    actions = [i % 5 for i in range(num_steps)]  # Cycle through Noop + 4 moves

    hashes: list[str] = []
    for run in range(2):
        env = ForgeEnv(config=config)
        obs, _info = env.reset(seed=seed)
        for action in actions:
            obs, _rew, term, trunc, _info = env.step(action)
            if term or trunc:
                obs, _info = env.reset(seed=seed)
        h = obs_hash(obs)
        hashes.append(h)
        env.close()
        print(f"  Run {run + 1}: {num_steps} steps with seed={seed} -> hash={h}")

    print()
    if hashes[0] == hashes[1]:
        if _use_color:
            print("  \033[1;32mPASS: Deterministic - observations are byte-identical\033[0m")
        else:
            print("  PASS: Deterministic - observations are byte-identical")
    else:
        print("  FAIL: Observations differ!")
        return False

    pause(0.5)
    return True


def demo_performance(seed: int) -> bool:
    """Section 8: Performance benchmark."""
    print_header("8. Performance Benchmark")

    config = {"world": {"width": 64, "height": 64}}
    n_steps = 10_000

    # Benchmark world creation
    t0 = time.perf_counter()
    env = ForgeEnv(config=config)
    _obs, _info = env.reset(seed=seed)
    creation_time = time.perf_counter() - t0

    print(f"  World creation (64x64): {creation_time * 1000:.1f} ms")
    print(f"  Running {n_steps:,} steps...")
    print()

    # Benchmark stepping
    t0 = time.perf_counter()
    for i in range(n_steps):
        _obs, _rew, term, trunc, _info = env.step(i % 5)
        if term or trunc:
            env.reset(seed=seed)
    elapsed = time.perf_counter() - t0

    fps = n_steps / elapsed if elapsed > 0 else float("inf")
    us_per_step = (elapsed / n_steps) * 1_000_000

    print("  Results:")
    print(f"    Steps/second  : {fps:>12,.0f}")
    print(f"    us/step       : {us_per_step:>12.2f}")
    print(f"    Total time    : {elapsed:>12.3f} s")
    print(f"    Steps         : {n_steps:>12,}")
    print()

    target_met = us_per_step < 10.0  # generous target for Python overhead
    if target_met:
        if _use_color:
            print("  \033[1;32mPerformance target met (<10 us/step from Python)\033[0m")
        else:
            print("  Performance target met (<10 us/step from Python)")
    else:
        print(f"  Note: {us_per_step:.1f} us/step (includes Python + PyO3 overhead)")

    env.close()
    pause(0.5)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SECTIONS = {
    "worldgen": ("World Generation", demo_world_generation),
    "navigation": ("Navigation", demo_navigation),
    "gathering": ("Resource Gathering", demo_resource_gathering),
    "crafting": ("Crafting", demo_crafting),
    "multiagent": ("Multi-Agent", demo_multi_agent),
    "daynight": ("Day/Night Cycle", demo_day_night),
    "determinism": ("Determinism", demo_determinism),
    "performance": ("Performance", demo_performance),
}


def _run_section(
    name: str,
    func: Any,
    seed: int,
) -> bool:
    """Run a single demo section with error handling."""
    try:
        return func(seed)
    except Exception as e:
        print(f"\n  ERROR in {name}: {e}")
        return False


def main() -> None:
    """Run the FORGE interactive demo."""
    global _use_color, _quick_mode  # noqa: PLW0603

    parser = argparse.ArgumentParser(
        description="FORGE Interactive Demo - showcases all major capabilities.",
    )
    parser.add_argument("--quick", action="store_true", help="Skip animation delays (CI mode)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI color output")
    parser.add_argument(
        "--section",
        choices=list(SECTIONS.keys()),
        help="Run only a specific section",
    )
    args = parser.parse_args()

    _use_color = not args.no_color
    _quick_mode = args.quick

    # Title
    if _use_color:
        print("\n\033[1;36m" + "=" * 60 + "\033[0m")
        print("\033[1;36m   FORGE - Fast Open-source Runtime for Generalist Envs\033[0m")
        print("\033[1;36m" + "=" * 60 + "\033[0m")
    else:
        print("\n" + "=" * 60)
        print("   FORGE - Fast Open-source Runtime for Generalist Envs")
        print("=" * 60)
    print(f"\n  Seed: {args.seed}  |  Quick: {args.quick}  |  Color: {_use_color}")

    demo_start = time.perf_counter()
    results: dict[str, bool] = {}

    if args.section:
        name, func = SECTIONS[args.section]
        results[name] = func(args.seed)
    else:
        for name, func in SECTIONS.values():
            results[name] = _run_section(name, func, args.seed)

    total_time = time.perf_counter() - demo_start

    # Summary
    print_header("Summary")
    for name, passed in results.items():
        if _use_color:
            status = "\033[1;32mPASS\033[0m" if passed else "\033[1;31mFAIL\033[0m"
        else:
            status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    passed_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    print(f"\n  {passed_count}/{total_count} sections passed in {total_time:.1f}s")
    print()


if __name__ == "__main__":
    main()
