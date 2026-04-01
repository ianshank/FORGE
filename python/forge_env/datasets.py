"""Offline training dataset utilities for forge_env.

Provides thin wrappers over Minari, MineRL, and the Strategic Game Maze dataset
that return batches in the same observation/action numpy array format used by
ForgeGymnasiumEnv and ForgeParallelEnv.

Supported sources
-----------------
- **Minari** (D4RL Maze2D, Adroit, Kitchen) — Apache 2.0
  https://minari.farama.org/
- **MineRL** (resource gathering, crafting) — MIT-like
  https://zenodo.org/records/12659939
- **Strategic Game Maze** (350K BFS-solved mazes) — Open / LAION
  https://huggingface.co/datasets/laion/strategic_game_maze
- **FORGE native JSONL** — any file exported by ``forge-replay``

Each loader returns a :class:`ForgeDataset` which implements the PyTorch
``Dataset`` interface (``__len__`` + ``__getitem__``) and also has a
``to_arrays()`` helper that returns plain NumPy arrays suitable for use with
JAX / Flax / StableBaselines3 offline RL methods.

Usage
-----
::

    from forge_env.datasets import load_minari, load_minerl, load_maze_jsonl

    # Load Minari PointMaze trajectories
    ds = load_minari("D4RL_pointmaze-umaze-v0", max_episodes=100)
    obs, actions, rewards, dones = ds.to_arrays()

    # Load a MineRL JSONL export (produced by scripts/export_minerl_to_jsonl.py)
    ds = load_minerl("data/minerl_export.jsonl", max_episodes=50)

    # Load the Strategic Game Maze JSONL
    ds = load_maze_jsonl("data/maze.jsonl", max_mazes=1000)

    # Load FORGE-native JSONL (from forge-replay export)
    ds = load_forge_jsonl("data/trajectories.jsonl")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step record (one transition)
# ---------------------------------------------------------------------------

@dataclass
class ForgeStep:
    """A single transition in an offline trajectory.

    All observation fields match the output of ``ForgeGymnasiumEnv.reset()``
    / ``step()``.
    """
    # Observation components
    grid_view: np.ndarray          # shape (view_h, view_w, 7), dtype uint8
    inventory: np.ndarray          # shape (10, 2), dtype uint16
    health: float
    stamina: float
    position: Tuple[int, int]
    day_phase: int
    task_progress: np.ndarray      # shape (n_predicates,), dtype float32

    # Transition
    action: int                    # discrete action id
    reward: float
    terminated: bool
    truncated: bool


# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------

class ForgeDataset:
    """Container for offline trajectory data.

    Implements ``len`` and ``__getitem__`` for PyTorch DataLoader compatibility,
    and provides :meth:`to_arrays` for bulk NumPy export.
    """

    def __init__(self, steps: List[ForgeStep], source: str = "unknown") -> None:
        self._steps = steps
        self.source = source

    def __len__(self) -> int:
        return len(self._steps)

    def __getitem__(self, idx: int) -> ForgeStep:
        return self._steps[idx]

    def __iter__(self) -> Iterator[ForgeStep]:
        return iter(self._steps)

    def to_arrays(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Export dataset as flat NumPy arrays.

        Returns
        -------
        observations : np.ndarray, shape (N, obs_dim)
            Flattened observation vector (grid_view + inventory + scalars).
        actions : np.ndarray, shape (N,), dtype int32
        rewards : np.ndarray, shape (N,), dtype float32
        dones : np.ndarray, shape (N,), dtype bool
            ``True`` on terminal or truncated steps.
        """
        n = len(self._steps)
        if n == 0:
            return (
                np.zeros((0, 1), dtype=np.float32),
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=bool),
            )

        # Build flat obs from first step to determine dimension
        sample_obs = self._flatten_obs(self._steps[0])
        obs_dim = sample_obs.shape[0]

        obs_arr = np.zeros((n, obs_dim), dtype=np.float32)
        act_arr = np.zeros(n, dtype=np.int32)
        rew_arr = np.zeros(n, dtype=np.float32)
        done_arr = np.zeros(n, dtype=bool)

        for i, step in enumerate(self._steps):
            obs_arr[i] = self._flatten_obs(step)
            act_arr[i] = step.action
            rew_arr[i] = step.reward
            done_arr[i] = step.terminated or step.truncated

        return obs_arr, act_arr, rew_arr, done_arr

    @staticmethod
    def _flatten_obs(step: ForgeStep) -> np.ndarray:
        """Flatten all observation fields into a 1-D float32 vector."""
        parts = [
            step.grid_view.flatten().astype(np.float32) / 255.0,
            step.inventory.flatten().astype(np.float32) / 64.0,  # max stack = 64
            np.array([step.health, step.stamina], dtype=np.float32),
            np.array(step.position, dtype=np.float32),
            np.array([step.day_phase / 3.0], dtype=np.float32),
            step.task_progress.astype(np.float32),
        ]
        return np.concatenate(parts)

    def __repr__(self) -> str:
        return f"ForgeDataset(source={self.source!r}, steps={len(self)})"


# ---------------------------------------------------------------------------
# Shared JSONL step parser
# ---------------------------------------------------------------------------

# Default observation shapes when fields are missing
_DEFAULT_VIEW_SIZE = 11
_DEFAULT_INV_SLOTS = 10


def _parse_forge_jsonl_step(raw: dict, tick: int) -> Optional[ForgeStep]:
    """Parse one line of a forge-replay JSONL export into a :class:`ForgeStep`.

    Fields that are absent receive safe defaults so the loader is lenient
    about partial exports.
    """
    try:
        obs_list = raw.get("observations", [{}])
        obs = obs_list[0] if obs_list else {}
        actions = raw.get("actions", [0])
        rewards = raw.get("rewards", [0.0])

        grid_raw = obs.get("grid_view", [])
        vw = obs.get("view_width", _DEFAULT_VIEW_SIZE)
        vh = obs.get("view_height", _DEFAULT_VIEW_SIZE)
        if grid_raw:
            flat = []
            for tile in grid_raw:
                if isinstance(tile, dict):
                    flat.extend([
                        tile.get("terrain", 0),
                        int(tile.get("has_agent", False)),
                        int(tile.get("has_object", False)),
                        int(tile.get("has_resource", False)),
                        tile.get("elevation", 0),
                        tile.get("object_type", 255),
                        tile.get("resource_type", 255),
                    ])
                elif isinstance(tile, (list, tuple)):
                    flat.extend(tile)
                else:
                    flat.append(int(tile))
            grid_view = np.array(flat, dtype=np.uint8).reshape(vh, vw, 7)
        else:
            grid_view = np.zeros((vh, vw, 7), dtype=np.uint8)

        inv_raw = obs.get("inventory", {}).get("slots", [])
        if inv_raw:
            inv = np.array(inv_raw, dtype=np.uint16).reshape(-1, 2)
            inv = inv[:_DEFAULT_INV_SLOTS]
            if len(inv) < _DEFAULT_INV_SLOTS:
                pad = np.zeros((_DEFAULT_INV_SLOTS - len(inv), 2), dtype=np.uint16)
                inv = np.vstack([inv, pad])
        else:
            inv = np.zeros((_DEFAULT_INV_SLOTS, 2), dtype=np.uint16)

        pos_raw = obs.get("position", [0, 0])
        position = (int(pos_raw[0]), int(pos_raw[1]))

        task_prog = obs.get("task_progress", [])
        task_progress = np.array(task_prog, dtype=np.float32) if task_prog else np.zeros(1, dtype=np.float32)

        return ForgeStep(
            grid_view=grid_view,
            inventory=inv,
            health=float(obs.get("health", 1.0)),
            stamina=float(obs.get("stamina", 1.0)),
            position=position,
            day_phase=int(obs.get("day_phase", 1)),
            task_progress=task_progress,
            action=int(actions[0]) if actions else 0,
            reward=float(rewards[0]) if rewards else 0.0,
            terminated=bool(raw.get("terminated", False)),
            truncated=bool(raw.get("truncated", False)),
        )
    except Exception as exc:
        logger.debug("Skipping malformed step at tick %d: %s", tick, exc)
        return None


# ---------------------------------------------------------------------------
# FORGE-native JSONL loader
# ---------------------------------------------------------------------------

def load_forge_jsonl(path: str, max_steps: int = 0) -> ForgeDataset:
    """Load a FORGE ``forge-replay`` JSONL export.

    Parameters
    ----------
    path : str
        Path to a ``.jsonl`` file produced by ``forge_replay::export::export_to_jsonl``
        or :meth:`OfflineDataset.export_jsonl`.
    max_steps : int
        Maximum steps to load (0 = unlimited).

    Returns
    -------
    ForgeDataset
    """
    steps: List[ForgeStep] = []
    with open(path) as f:
        for tick, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            step = _parse_forge_jsonl_step(raw, tick)
            if step is not None:
                steps.append(step)
            if max_steps > 0 and len(steps) >= max_steps:
                break

    logger.info("Loaded %d steps from FORGE JSONL %s", len(steps), path)
    return ForgeDataset(steps, source=f"forge-jsonl:{path}")


# ---------------------------------------------------------------------------
# Minari loader
# ---------------------------------------------------------------------------

def load_minari(
    dataset_id: str,
    max_episodes: int = 0,
    max_steps: int = 0,
) -> ForgeDataset:
    """Load a Minari offline RL dataset.

    Requires ``minari`` to be installed::

        pip install minari

    Popular dataset IDs:
        - ``"D4RL_pointmaze-umaze-v0"``   (navigation, small maze)
        - ``"D4RL_pointmaze-medium-v0"``  (navigation, medium maze)
        - ``"D4RL_pointmaze-large-v0"``   (navigation, large maze)

    All datasets are available at https://minari.farama.org/

    Parameters
    ----------
    dataset_id : str
        Minari dataset identifier.
    max_episodes : int
        Maximum episodes to load (0 = unlimited).
    max_steps : int
        Maximum steps to load (0 = unlimited).

    Returns
    -------
    ForgeDataset
    """
    try:
        import minari  # type: ignore
    except ImportError:
        raise ImportError(
            "minari is required to load Minari datasets. "
            "Install with: pip install minari"
        )

    logger.info("Loading Minari dataset %s...", dataset_id)
    dataset = minari.load_dataset(dataset_id, download=True)

    steps: List[ForgeStep] = []
    ep_count = 0

    for episode in dataset.iterate_episodes():
        for t in range(len(episode.actions)):
            obs = episode.observations
            # Minari stores obs as a dict of arrays; extract position if available.
            position = (0, 0)
            if isinstance(obs, dict) and "observation" in obs:
                xy = obs["observation"][t]
                position = (int(xy[0] * 10), int(xy[1] * 10))  # scale to tile coords

            health = 1.0
            if isinstance(obs, dict) and "health" in obs:
                health = float(obs["health"][t])

            action = int(episode.actions[t]) if np.isscalar(episode.actions[t]) else 0
            reward = float(episode.rewards[t])
            terminated = bool(episode.terminations[t]) if hasattr(episode, "terminations") else False
            truncated = bool(episode.truncations[t]) if hasattr(episode, "truncations") else False

            step = ForgeStep(
                grid_view=np.zeros((_DEFAULT_VIEW_SIZE, _DEFAULT_VIEW_SIZE, 7), dtype=np.uint8),
                inventory=np.zeros((_DEFAULT_INV_SLOTS, 2), dtype=np.uint16),
                health=health,
                stamina=1.0,
                position=position,
                day_phase=1,
                task_progress=np.zeros(1, dtype=np.float32),
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
            )
            steps.append(step)

            if max_steps > 0 and len(steps) >= max_steps:
                break

        ep_count += 1
        if max_episodes > 0 and ep_count >= max_episodes:
            break
        if max_steps > 0 and len(steps) >= max_steps:
            break

    logger.info("Loaded %d steps from Minari %s", len(steps), dataset_id)
    return ForgeDataset(steps, source=f"minari:{dataset_id}")


# ---------------------------------------------------------------------------
# MineRL JSONL loader
# ---------------------------------------------------------------------------

# MineRL action name → FORGE discrete action id (comm_vocab=0, no drone)
_MINERL_ACTION_MAP: dict[str, int] = {
    "noop": 0,
    "move_forward": 1,   # Move(Up)
    "move_back": 2,      # Move(Down)
    "move_left": 3,      # Move(Left)
    "move_right": 4,     # Move(Right)
    "pickup": 5,         # PickUp
    "attack": 16,        # Use(slot 0) — weapon
    "interact": 39,      # Interact
}


def _minerl_action_to_discrete(action: dict) -> int:
    """Convert a MineRL action dict to a FORGE discrete action id."""
    if action.get("no_op"):
        return 0
    craft = action.get("craft")
    if craft:
        # Craft recipe mapping (same as minerl.rs CRAFT_MAP, pickaxe before axe)
        craft_lower = craft.lower()
        if "pickaxe" in craft_lower:
            return 27  # Craft(1) = pickaxe
        if "axe" in craft_lower:
            return 26  # Craft(0) = axe
        if "plank" in craft_lower:
            return 28  # Craft(2) = plank
        if "sword" in craft_lower:
            return 30  # Craft(4) = sword
        if "torch" in craft_lower:
            return 33  # Craft(7) = torch
        return 0
    if action.get("attack"):
        return 16  # Use(0)
    if action.get("use"):
        return 39  # Interact
    if action.get("pickup"):
        return 5   # PickUp
    if action.get("forward"):
        return 1   # Move(Up)
    if action.get("back"):
        return 2   # Move(Down)
    if action.get("left"):
        return 3   # Move(Left)
    if action.get("right"):
        return 4   # Move(Right)
    return 0  # Noop


def load_minerl(
    path: str,
    max_episodes: int = 0,
    max_steps: int = 0,
) -> ForgeDataset:
    """Load a MineRL JSONL export into a :class:`ForgeDataset`.

    The JSONL file should be produced by ``scripts/export_minerl_to_jsonl.py``.
    Each line contains one step with ``action``, ``reward``, ``terminated``,
    ``truncated``, and optional ``obs`` fields.

    Parameters
    ----------
    path : str
        Path to the MineRL JSONL export.
    max_episodes : int
        Maximum episodes (0 = unlimited).
    max_steps : int
        Maximum steps (0 = unlimited).

    Returns
    -------
    ForgeDataset
    """
    steps: List[ForgeStep] = []
    ep_count = 0

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)

            action_dict = raw.get("action", {})
            action_id = _minerl_action_to_discrete(action_dict)
            reward = float(raw.get("reward", 0.0))
            terminated = bool(raw.get("terminated", False))
            truncated = bool(raw.get("truncated", False))

            obs_dict = raw.get("obs", {}) or {}
            health_raw = obs_dict.get("health", 20.0)
            health = min(1.0, float(health_raw) / 20.0)  # Minecraft health: max 20

            pos_raw = obs_dict.get("position", [0.0, 0.0, 0.0])
            position = (int(pos_raw[0]), int(pos_raw[2]))  # x, z → FORGE (x, y)

            step = ForgeStep(
                grid_view=np.zeros((_DEFAULT_VIEW_SIZE, _DEFAULT_VIEW_SIZE, 7), dtype=np.uint8),
                inventory=np.zeros((_DEFAULT_INV_SLOTS, 2), dtype=np.uint16),
                health=health,
                stamina=1.0,
                position=position,
                day_phase=1,
                task_progress=np.zeros(1, dtype=np.float32),
                action=action_id,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
            )
            steps.append(step)

            if terminated or truncated:
                ep_count += 1
                if max_episodes > 0 and ep_count >= max_episodes:
                    break

            if max_steps > 0 and len(steps) >= max_steps:
                break

    logger.info("Loaded %d steps from MineRL export %s", len(steps), path)
    return ForgeDataset(steps, source=f"minerl:{path}")


# ---------------------------------------------------------------------------
# Strategic Game Maze JSONL loader
# ---------------------------------------------------------------------------

_MAZE_DIR_TO_ACTION = {"U": 1, "D": 2, "L": 3, "R": 4}  # Move(Up/Down/Left/Right)


def load_maze_jsonl(
    path: str,
    max_mazes: int = 0,
    max_solution_length: int = 0,
) -> ForgeDataset:
    """Load the Strategic Game Maze JSONL dataset.

    Each maze record is converted to a sequence of :class:`ForgeStep` entries
    where every BFS move is one ``Move`` action with a small intermediate reward
    and a final reward of 1.0 on the last step.

    Parameters
    ----------
    path : str
        Path to the JSONL export from the HuggingFace dataset.
    max_mazes : int
        Maximum mazes to load (0 = unlimited).
    max_solution_length : int
        Skip mazes whose solution is longer than this (0 = no limit).

    Returns
    -------
    ForgeDataset
    """
    steps: List[ForgeStep] = []
    maze_count = 0

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            solution: str = record.get("solution", "")
            if max_solution_length > 0 and len(solution) > max_solution_length:
                continue

            start = record.get("start", [0, 0])
            pos = [int(start[0]), int(start[1])]
            n = len(solution)

            for i, ch in enumerate(solution.upper()):
                action_id = _MAZE_DIR_TO_ACTION.get(ch, 0)
                is_last = i == n - 1
                reward = 1.0 if is_last else 0.01

                step = ForgeStep(
                    grid_view=np.zeros((_DEFAULT_VIEW_SIZE, _DEFAULT_VIEW_SIZE, 7), dtype=np.uint8),
                    inventory=np.zeros((_DEFAULT_INV_SLOTS, 2), dtype=np.uint16),
                    health=1.0,
                    stamina=1.0,
                    position=(pos[0], pos[1]),
                    day_phase=1,
                    task_progress=np.array([float(i + 1) / n], dtype=np.float32),
                    action=action_id,
                    reward=reward,
                    terminated=is_last,
                    truncated=False,
                )
                steps.append(step)

                # Advance position
                if ch == "U":
                    pos[1] = max(0, pos[1] - 1)
                elif ch == "D":
                    pos[1] += 1
                elif ch == "L":
                    pos[0] = max(0, pos[0] - 1)
                elif ch == "R":
                    pos[0] += 1

            maze_count += 1
            if max_mazes > 0 and maze_count >= max_mazes:
                break

    logger.info("Loaded %d steps from %d mazes in %s", len(steps), maze_count, path)
    return ForgeDataset(steps, source=f"maze-jsonl:{path}")
