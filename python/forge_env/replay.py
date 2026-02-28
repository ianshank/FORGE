"""forge_env/replay.py — Load and play back ``.forge`` episode replay files.

The ``.forge`` format is a versioned JSON schema that stores seed, config,
actions, observations, rewards, and step timestamps for a single episode.

CLI usage::

    forge-replay episode.forge
    forge-replay episode.forge --fps 30
    forge-replay episode.forge --export-gif out.gif --fps 10
    forge-replay episode.forge --start-frame 50

Python API::

    from forge_env.replay import load_replay, play_replay, export_gif

    data = load_replay("episode.forge")
    play_replay(data, fps=10)
    export_gif(data, "out.gif", fps=10)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

logger = logging.getLogger(__name__)

# Current supported format version
_SUPPORTED_FORMAT_VERSION: int = 1

# Max frames for GIF export (memory guard)
_GIF_MAX_FRAMES: int = 2_000


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayData:
    """Typed view of a ``.forge`` replay file.

    All fields match the JSON schema exactly.  ``observations`` and
    ``timestamps_ms`` are optional — they may be absent in files written with
    ``store_observations=False`` or produced by older tool versions.
    """

    forge_version: str
    format_version: int
    seed: int | None
    config: dict[str, Any]
    actions: list[int]
    rewards: list[float]
    terminated_at: int
    observations: list[list[float]] = field(default_factory=list)
    timestamps_ms: list[float] = field(default_factory=list)

    @property
    def num_steps(self) -> int:
        """Total number of recorded steps."""
        return len(self.actions)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_replay(path: str | Path) -> ReplayData:
    """Load a ``.forge`` replay file from disk.

    Parameters
    ----------
    path:
        Path to the ``.forge`` JSON file.

    Returns
    -------
    ReplayData:
        Parsed, validated replay data.

    Raises
    ------
    FileNotFoundError:
        If the file does not exist.
    ValueError:
        If the file is not valid JSON or has an unsupported format version.
    """
    fpath = Path(path)
    if not fpath.exists():
        raise FileNotFoundError(f"Replay file not found: {fpath}")

    try:
        raw = json.loads(fpath.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in replay file {fpath}: {exc}") from exc

    fmt_ver = raw.get("format_version", 1)
    if fmt_ver > _SUPPORTED_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported .forge format version {fmt_ver}. "
            f"This tool supports up to version {_SUPPORTED_FORMAT_VERSION}."
        )

    replay_ver = raw.get("forge_version", "unknown")
    try:
        from importlib.metadata import version  # noqa: PLC0415

        current_ver = version("forge-env")
        if current_ver != replay_ver:
            logger.warning(
                "Version mismatch: replay was recorded with forge-env %s, "
                "current version is %s. Proceeding anyway.",
                replay_ver,
                current_ver,
            )
    except Exception:
        pass

    return ReplayData(
        forge_version=replay_ver,
        format_version=fmt_ver,
        seed=raw.get("seed"),
        config=raw.get("config", {}),
        actions=raw.get("actions", []),
        rewards=raw.get("rewards", []),
        terminated_at=raw.get("terminated_at", len(raw.get("actions", []))),
        observations=raw.get("observations", []),
        timestamps_ms=raw.get("timestamps_ms", []),
    )


# ---------------------------------------------------------------------------
# Terminal playback
# ---------------------------------------------------------------------------

#: ASCII characters used to render cells (space = empty, heavy block = agent)
_CELL_CHARS: tuple[str, ...] = (" ", "·", "○", "A", "█")


def play_replay(
    data: ReplayData,
    fps: float = 10.0,
    stream: TextIO = sys.stdout,
    start_frame: int = 0,
) -> None:
    """Render the replay step-by-step in the terminal.

    Parameters
    ----------
    data:
        Loaded replay data.
    fps:
        Target playback frames per second.
    stream:
        Output stream.  Defaults to ``sys.stdout``.
    start_frame:
        First frame index to render (0-indexed).
    """
    delay = 1.0 / max(fps, 0.1)
    width = data.config.get("world", {}).get("width", 16)
    height = data.config.get("world", {}).get("height", 16)

    _print = lambda msg: print(msg, file=stream, flush=True)  # noqa: E731
    _clear = "\033[2J\033[H"  # ANSI clear screen + move cursor to top

    _print(f"FORGE Replay  seed={data.seed}  steps={data.num_steps}  fps={fps}")
    _print(f"Config: {json.dumps(data.config, separators=(',', ':'))}")
    _print("─" * (width + 2))

    for step_idx, action in enumerate(data.actions):
        if step_idx < start_frame:
            continue

        reward = data.rewards[step_idx] if step_idx < len(data.rewards) else 0.0
        obs = data.observations[step_idx] if step_idx < len(data.observations) else []

        # Build a simple ASCII grid — obs values normalised to char index
        if obs:
            cells = [_CELL_CHARS[min(int(abs(v) * len(_CELL_CHARS)), len(_CELL_CHARS) - 1)] for v in obs]
            rows = [
                "│"
                + "".join(cells[r * width : (r + 1) * width])
                + "│"
                for r in range(height)
            ]
            grid = "\n".join(rows)
        else:
            grid = "(no observation data stored)"

        _print(_clear)
        _print(f"Step {step_idx + 1}/{data.num_steps}  action={action}  reward={reward:+.3f}")
        _print(f"┌{'─' * width}┐")
        _print(grid)
        _print(f"└{'─' * width}┘")

        time.sleep(delay)

    _print("\nReplay complete.")


# ---------------------------------------------------------------------------
# GIF export
# ---------------------------------------------------------------------------


def export_gif(
    data: ReplayData,
    out_path: str | Path,
    fps: float = 10.0,
    cell_size_px: int = 8,
    max_frames: int = _GIF_MAX_FRAMES,
) -> None:
    """Export the replay as an animated GIF using Pillow.

    Parameters
    ----------
    data:
        Loaded replay data.
    out_path:
        Output ``.gif`` file path.
    fps:
        Frames per second (converted to GIF frame duration).
    cell_size_px:
        Pixel size of each world cell.
    max_frames:
        Maximum number of frames to export (memory guard).

    Raises
    ------
    ImportError:
        If Pillow is not installed.
    ValueError:
        If the replay has no stored observations.
    """
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for GIF export. Install: pip install Pillow"
        ) from exc

    if not data.observations:
        raise ValueError("GIF export requires stored observations. Re-record with store_observations=True.")

    width = data.config.get("world", {}).get("width", 16)
    height = data.config.get("world", {}).get("height", 16)
    duration_ms = int(1000 / max(fps, 1))

    frames: list[Image.Image] = []
    for obs in data.observations[:max_frames]:
        # Map observation values to greyscale pixels
        pixels = [
            min(255, max(0, int((v + 1.0) / 2.0 * 255)))
            for v in obs[: width * height]
        ]
        # Pad if shorter than grid
        pixels += [0] * (width * height - len(pixels))

        img = Image.new("L", (width * cell_size_px, height * cell_size_px), color=0)
        px = img.load()
        for r in range(height):
            for c in range(width):
                grey = pixels[r * width + c]
                for dr in range(cell_size_px):
                    for dc in range(cell_size_px):
                        px[c * cell_size_px + dc, r * cell_size_px + dr] = grey  # type: ignore[index]

        frames.append(img.convert("P"))

    if not frames:
        raise ValueError("No frames to export.")

    out = Path(out_path)
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    logger.info("GIF exported → %s (%d frames)", out, len(frames))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    """Entry point for the ``forge-replay`` command."""
    parser = argparse.ArgumentParser(
        prog="forge-replay",
        description="Play back a FORGE .forge episode replay file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("path", help="Path to the .forge replay file")
    parser.add_argument(
        "--fps", type=float, default=10.0, help="Playback speed in frames/sec (default: 10)"
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        metavar="N",
        help="Start playback from frame N (0-indexed)",
    )
    parser.add_argument(
        "--export-gif",
        metavar="OUT.gif",
        help="Export an animated GIF instead of terminal playback",
    )
    parser.add_argument(
        "--cell-size", type=int, default=8, metavar="PX", help="GIF cell size in pixels (default: 8)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=_GIF_MAX_FRAMES,
        help=f"Max frames for GIF export (default: {_GIF_MAX_FRAMES})",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        data = load_replay(args.path)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    if args.export_gif:
        try:
            export_gif(
                data,
                out_path=args.export_gif,
                fps=args.fps,
                cell_size_px=args.cell_size,
                max_frames=args.max_frames,
            )
        except (ImportError, ValueError) as exc:
            logger.error("%s", exc)
            return 1
    else:
        play_replay(data, fps=args.fps, start_frame=args.start_frame)

    return 0


if __name__ == "__main__":
    sys.exit(_cli())
