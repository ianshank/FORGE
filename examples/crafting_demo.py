"""Crafting system demonstration for the FORGE environment.

Shows how the crafting subsystem works by running an agent that explores
the world, gathers raw resources (Wood, Stone, Ore, Fiber, etc.), and
attempts to craft items using the built-in recipe book.

FORGE discrete action encoding (relevant subset):
    0     : Noop
    1-4   : Move (Up, Down, Left, Right)
    5     : Pick Up (gather resource at current tile)
    16-25 : Use item from inventory slot 0-9
    26    : Craft recipe index 0 (Axe: 2 Wood + 1 Stone)
    31    : Interact

Default recipes (from RecipeBook::default_recipes):
    0 - Axe      : 2 Wood + 1 Stone  -> 1 Axe       (level 1, no station)
    1 - Pickaxe  : 2 Wood + 2 Stone  -> 1 Pickaxe    (level 1, no station)
    2 - Plank    : 2 Wood            -> 2 Plank      (level 1, no station)
    3 - Bridge   : 4 Plank           -> 1 Bridge     (level 2, no station)
    4 - Sword    : 1 Wood + 2 Ore    -> 1 Sword      (level 2, station)
    5 - Rope     : 3 Fiber           -> 1 Rope       (level 1, no station)
    6 - Brick    : 2 Clay            -> 1 Brick      (level 1, station)
    7 - Torch    : 1 Wood + 1 Fiber  -> 1 Torch      (level 1, no station)
    8 - Shield   : 2 Plank + 1 Ore   -> 1 Shield     (level 2, station)

Usage:
    python crafting_demo.py
    python crafting_demo.py --steps 500 --seed 123
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
        "    cd /path/to/FORGE && pip install -e .\n"
        "    (requires maturin: pip install maturin)"
    )
    sys.exit(1)


# Item type IDs (matching forge_types::resource::ItemType repr(u8) values).
ITEM_NAMES = {
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
    255: "(empty)",
}

# Actions for gathering and crafting.
ACTION_NOOP = 0
ACTION_MOVE_UP = 1
ACTION_MOVE_DOWN = 2
ACTION_MOVE_LEFT = 3
ACTION_MOVE_RIGHT = 4
ACTION_PICKUP = 5
ACTION_CRAFT_BASE = 26  # Craft recipe 0; recipe index is encoded as (26 + recipe_id)
ACTION_INTERACT = 31

MOVE_ACTIONS = [ACTION_MOVE_UP, ACTION_MOVE_DOWN, ACTION_MOVE_LEFT, ACTION_MOVE_RIGHT]


def parse_args():
    parser = argparse.ArgumentParser(
        description="FORGE crafting demo: gather resources and craft items."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of environment steps to run (default: 500).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducibility (default: 7).",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=48,
        help="Width and height of the world grid (default: 48).",
    )
    return parser.parse_args()


def read_inventory(obs):
    """Parse the inventory from the observation dict.

    Returns a dict mapping item_type_id -> total count across all slots.
    """
    inventory = {}
    if not isinstance(obs, dict):
        return inventory

    inv_data = obs.get("inventory")
    if inv_data is None:
        return inventory

    try:
        # inv_data is a numpy array of shape (capacity, 2): [[item_type, count], ...]
        for slot_idx in range(len(inv_data)):
            item_type = int(inv_data[slot_idx][0])
            count = int(inv_data[slot_idx][1])
            if item_type != 255 and count > 0:
                inventory[item_type] = inventory.get(item_type, 0) + count
    except (IndexError, TypeError, ValueError):
        pass

    return inventory


def format_inventory(inv):
    """Format an inventory dict as a human-readable string."""
    if not inv:
        return "(empty)"
    parts = []
    for item_id, count in sorted(inv.items()):
        name = ITEM_NAMES.get(item_id, f"Item#{item_id}")
        parts.append(f"{name} x{count}")
    return ", ".join(parts)


def inventory_diff(before, after):
    """Compute what changed between two inventory snapshots.

    Returns (gained, lost) where each is a dict of item_id -> delta count.
    """
    all_keys = set(before.keys()) | set(after.keys())
    gained = {}
    lost = {}
    for k in all_keys:
        old = before.get(k, 0)
        new = after.get(k, 0)
        if new > old:
            gained[k] = new - old
        elif old > new:
            lost[k] = old - new
    return gained, lost


def run_crafting_demo(steps, seed, world_size):  # noqa: PLR0912, PLR0915
    """Run the crafting demonstration."""

    config = {
        "world": {
            "width": world_size,
            "height": world_size,
            "resource_density": 0.5,  # more resources to find
        },
        "agents": {
            "num_agents": 1,
        },
        "crafting": {
            "enabled": True,
            "procedural_recipes": False,
            "require_discovery": False,
        },
        "task": {
            "max_episode_length": steps + 100,
            "dense_rewards": True,
        },
    }

    print("=" * 60)
    print("FORGE Crafting Demo")
    print("=" * 60)
    print(f"World size     : {world_size}x{world_size}")
    print("Resource density: 0.5 (high)")
    print("Crafting       : enabled")
    print(f"Seed           : {seed}")
    print(f"Steps          : {steps}")
    print()

    env = ForgeEnv(config=config)
    obs, _info = env.reset(seed=seed)
    rng = random.Random(seed)

    total_reward = 0.0
    items_collected = {}  # cumulative items gained
    items_crafted = {}    # cumulative crafted items gained
    craft_attempts = 0
    craft_successes = 0
    pickup_attempts = 0

    prev_inventory = read_inventory(obs)
    print(f"Initial inventory: {format_inventory(prev_inventory)}")
    print(f"Initial position : {_extract_position(obs)}")
    print()

    # Strategy: alternate between exploring (moving + picking up) and crafting.
    # Every ~20 steps, attempt to craft recipe 0 (Axe) or recipe 7 (Torch).
    craft_recipes_to_try = [0, 1, 2, 5, 7]  # Axe, Pickaxe, Plank, Rope, Torch

    for step_idx in range(1, steps + 1):
        # Decide action: mostly explore, periodically try to craft.
        if step_idx % 20 == 0:
            # Attempt crafting a random recipe from our list.
            recipe_id = rng.choice(craft_recipes_to_try)
            action = ACTION_CRAFT_BASE + recipe_id
            craft_attempts += 1
        elif step_idx % 3 == 0:
            # Try to pick up whatever is on the ground.
            action = ACTION_PICKUP
            pickup_attempts += 1
        else:
            # Random movement.
            action = rng.choice(MOVE_ACTIONS)

        obs, reward, terminated, truncated, _info = env.step(action)
        total_reward += reward

        # Check for inventory changes.
        current_inventory = read_inventory(obs)
        gained, lost = inventory_diff(prev_inventory, current_inventory)

        if gained or lost:
            # Determine if this was a craft action (inputs consumed, output gained).
            is_craft = bool(lost) and bool(gained)
            if is_craft:
                craft_successes += 1
                for item_id, count in gained.items():
                    name = ITEM_NAMES.get(item_id, f"Item#{item_id}")
                    items_crafted[item_id] = items_crafted.get(item_id, 0) + count
                    lost_str = format_inventory(lost)
                    print(
                        f"  [Step {step_idx:>4d}] CRAFTED: {name} x{count} "
                        f"(consumed: {lost_str})"
                    )
            elif gained and not lost:
                for item_id, count in gained.items():
                    name = ITEM_NAMES.get(item_id, f"Item#{item_id}")
                    items_collected[item_id] = items_collected.get(item_id, 0) + count
                    print(f"  [Step {step_idx:>4d}] PICKED UP: {name} x{count}")
            elif lost and not gained:
                for item_id, count in lost.items():
                    name = ITEM_NAMES.get(item_id, f"Item#{item_id}")
                    print(f"  [Step {step_idx:>4d}] LOST: {name} x{count}")

        prev_inventory = current_inventory

        # Handle episode end.
        if terminated or truncated:
            reason = "terminated" if terminated else "truncated"
            print(f"\n  >> Episode {reason} at step {step_idx}. Resetting ...\n")
            obs, _info = env.reset(seed=seed)
            prev_inventory = read_inventory(obs)

    # Final summary.
    final_inventory = read_inventory(obs)

    print()
    print("=" * 60)
    print("Crafting Demo Results")
    print("=" * 60)
    print(f"  Steps run        : {steps}")
    print(f"  Total reward     : {total_reward:+.4f}")
    print(f"  Pickup attempts  : {pickup_attempts}")
    print(f"  Craft attempts   : {craft_attempts}")
    print(f"  Craft successes  : {craft_successes}")
    print()
    print("  Resources collected:")
    if items_collected:
        for item_id, count in sorted(items_collected.items()):
            name = ITEM_NAMES.get(item_id, f"Item#{item_id}")
            print(f"    {name:>12s} : {count}")
    else:
        print("    (none)")
    print()
    print("  Items crafted:")
    if items_crafted:
        for item_id, count in sorted(items_crafted.items()):
            name = ITEM_NAMES.get(item_id, f"Item#{item_id}")
            print(f"    {name:>12s} : {count}")
    else:
        print("    (none -- try increasing --steps or --world-size)")
    print()
    print(f"  Final inventory: {format_inventory(final_inventory)}")
    print(f"  Final position : {_extract_position(obs)}")

    env.close()
    print("\nEnvironment closed.")


def _extract_position(obs):
    """Extract (x, y) position from the observation dict."""
    if isinstance(obs, dict):
        pos = obs.get("position", (0, 0))
        try:
            return (int(pos[0]), int(pos[1]))
        except (IndexError, TypeError):
            return (0, 0)
    return (0, 0)


if __name__ == "__main__":
    args = parse_args()
    run_crafting_demo(
        steps=args.steps,
        seed=args.seed,
        world_size=args.world_size,
    )
