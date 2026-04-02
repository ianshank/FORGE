"""Checkpoint management for saving and loading agent state."""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forge.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

DEFAULT_MAX_CHECKPOINTS = 5
CHECKPOINT_SCHEMA_VERSION = 1


class CheckpointManager:
    """Manages saving and loading agent checkpoints with rotation."""

    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = DEFAULT_MAX_CHECKPOINTS,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "CheckpointManager initialized: dir=%s, max=%d",
            checkpoint_dir,
            max_checkpoints,
        )

    def save(
        self,
        agent: BaseAgent,
        episode: int,
        metrics: dict[str, float],
    ) -> None:
        """Save an agent checkpoint with metadata."""
        timestamp = int(time.time() * 1000)
        checkpoint_name = f"checkpoint_{episode}_{timestamp}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Save agent state
        agent.save(str(checkpoint_path / "agent.json"))

        # Save metadata with schema version for backwards compatibility
        metadata = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "episode": episode,
            "timestamp": timestamp,
            "metrics": metrics,
            "step_count": agent.step_count,
        }
        meta_path = checkpoint_path / "metadata.json"
        with meta_path.open("w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Saved checkpoint at episode %d to %s", episode, checkpoint_path)
        self._rotate_checkpoints()

    def load_latest(self, agent: BaseAgent) -> dict[str, object] | None:
        """Load the most recent checkpoint. Returns metadata or None."""
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            logger.info("No checkpoints found")
            return None

        latest = checkpoints[-1]
        checkpoint_path = Path(str(latest["path"]))
        agent.load(str(checkpoint_path / "agent.json"))
        logger.info("Loaded latest checkpoint from %s", checkpoint_path)
        return latest

    def list_checkpoints(self) -> list[dict[str, object]]:
        """List all available checkpoints sorted by episode."""
        checkpoints: list[dict[str, object]] = []
        if not self.checkpoint_dir.exists():
            return checkpoints

        for entry in sorted(self.checkpoint_dir.iterdir()):
            meta_path = entry / "metadata.json"
            if meta_path.exists():
                try:
                    with meta_path.open() as f:
                        metadata = json.load(f)
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning("Skipping corrupt checkpoint %s: %s", entry, exc)
                    continue
                metadata = self._migrate_metadata(metadata)
                metadata["path"] = str(entry)
                checkpoints.append(metadata)

        checkpoints.sort(key=lambda c: int(str(c.get("episode", 0))))
        return checkpoints

    @staticmethod
    def _migrate_metadata(metadata: dict[str, object]) -> dict[str, object]:
        """Migrate checkpoint metadata from older schema versions."""
        version = metadata.get("schema_version", 0)
        if isinstance(version, int) and version < 1:
            metadata.setdefault("schema_version", CHECKPOINT_SCHEMA_VERSION)
            metadata.setdefault("step_count", 0)
            logger.debug(
                "Migrated checkpoint metadata from v%s to v%d", version, CHECKPOINT_SCHEMA_VERSION
            )
        return metadata

    def _rotate_checkpoints(self) -> None:
        """Remove old checkpoints if we exceed max_checkpoints."""
        checkpoints = self.list_checkpoints()
        while len(checkpoints) > self.max_checkpoints:
            oldest = checkpoints.pop(0)
            oldest_path = Path(str(oldest["path"]))
            if oldest_path.exists():
                shutil.rmtree(oldest_path)
                logger.info("Removed old checkpoint: %s", oldest_path)
