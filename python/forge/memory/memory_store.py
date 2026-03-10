"""Memory store: persistent agent memory with semantic, episodic, and preference layers.

This module provides a Python-native memory store that mirrors the Rust
InMemoryStore API, enabling agents to accumulate experience across episodes.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SemanticFact:
    """A learned fact about the world."""

    key: str
    value: str
    confidence: float = 1.0
    source_tick: int = 0
    strength: float = 1.0
    reinforcement_count: int = 0


@dataclass
class Episode:
    """A recorded episode of agent experience."""

    tick_start: int = 0
    tick_end: int = 0
    agent_ids: list[int] = field(default_factory=list)
    location: tuple[int, int] = (0, 0)
    event_summaries: list[str] = field(default_factory=list)
    outcome: str = "neutral"
    reward: float = 0.0
    tags: list[str] = field(default_factory=list)
    strength: float = 1.0


@dataclass
class Preference:
    """A learned action preference for a context."""

    context_key: str = ""
    action_weights: dict[int, float] = field(default_factory=dict)
    update_count: int = 0
    strength: float = 1.0

    def update(self, action_id: int, reward: float, lr: float = 0.1) -> None:
        """Update the weight for an action using exponential moving average."""
        if action_id in self.action_weights:
            old = self.action_weights[action_id]
            self.action_weights[action_id] = old * (1 - lr) + reward * lr
        else:
            self.action_weights[action_id] = reward
        self.update_count += 1

    def preferred_action(self) -> int | None:
        """Return the action with the highest weight."""
        if not self.action_weights:
            return None
        return max(self.action_weights, key=self.action_weights.get)  # type: ignore[arg-type]


@dataclass
class MemoryStoreConfig:
    """Configuration for the memory store."""

    semantic_capacity: int = 10_000
    episodic_capacity: int = 5_000
    preference_capacity: int = 1_000
    decay_rate: float = 0.001
    min_strength: float = 0.01


class MemoryStore:
    """Python-native persistent memory store for a single agent."""

    def __init__(self, agent_id: int, config: MemoryStoreConfig | None = None) -> None:
        self.agent_id = agent_id
        self.config = config or MemoryStoreConfig()
        self._semantic: list[SemanticFact] = []
        self._episodic: list[Episode] = []
        self._preferences: dict[str, Preference] = {}
        logger.info("MemoryStore created for agent %d", agent_id)

    def store_fact(self, fact: SemanticFact) -> None:
        """Store a semantic fact, replacing existing facts with the same key."""
        for i, f in enumerate(self._semantic):
            if f.key == fact.key:
                self._semantic[i] = fact
                return
        if len(self._semantic) >= self.config.semantic_capacity:
            weakest = min(range(len(self._semantic)), key=lambda i: self._semantic[i].strength)
            self._semantic.pop(weakest)
        self._semantic.append(fact)

    def store_episode(self, episode: Episode) -> None:
        """Store an episodic memory."""
        if len(self._episodic) >= self.config.episodic_capacity:
            weakest = min(range(len(self._episodic)), key=lambda i: self._episodic[i].strength)
            self._episodic.pop(weakest)
        self._episodic.append(episode)

    def get_preference(self, context_key: str) -> Preference:
        """Get or create a preference for the given context."""
        if context_key not in self._preferences:
            self._preferences[context_key] = Preference(context_key=context_key)
        return self._preferences[context_key]

    def query_facts(self, prefix: str = "", min_strength: float = 0.0) -> list[SemanticFact]:
        """Query semantic facts by key prefix and minimum strength."""
        return [
            f
            for f in self._semantic
            if f.key.startswith(prefix) and f.strength >= min_strength
        ]

    def query_episodes(
        self, agent_id: int | None = None, tag: str | None = None
    ) -> list[Episode]:
        """Query episodes by agent ID or tag."""
        results = self._episodic
        if agent_id is not None:
            results = [e for e in results if agent_id in e.agent_ids]
        if tag is not None:
            results = [e for e in results if tag in e.tags]
        return results

    def recent_episodes(self, n: int = 10) -> list[Episode]:
        """Return the most recent episodes."""
        return self._episodic[-n:]

    def tick_decay(self) -> None:
        """Apply decay to all memories and prune weak ones."""
        rate = self.config.decay_rate
        threshold = self.config.min_strength
        for f in self._semantic:
            f.strength = max(0.0, f.strength - rate)
        self._semantic = [f for f in self._semantic if f.strength >= threshold]
        for e in self._episodic:
            e.strength = max(0.0, e.strength - rate)
        self._episodic = [e for e in self._episodic if e.strength >= threshold]

    def total_entries(self) -> int:
        """Return total memory entries across all subsystems."""
        return len(self._semantic) + len(self._episodic) + len(self._preferences)

    def save(self, path: str) -> None:
        """Save the memory store to a JSON file."""
        data: dict[str, Any] = {
            "agent_id": self.agent_id,
            "semantic": [
                {"key": f.key, "value": f.value, "confidence": f.confidence,
                 "source_tick": f.source_tick, "strength": f.strength,
                 "reinforcement_count": f.reinforcement_count}
                for f in self._semantic
            ],
            "episodic": [
                {"tick_start": e.tick_start, "tick_end": e.tick_end,
                 "agent_ids": e.agent_ids, "location": list(e.location),
                 "event_summaries": e.event_summaries, "outcome": e.outcome,
                 "reward": e.reward, "tags": e.tags, "strength": e.strength}
                for e in self._episodic
            ],
            "preferences": [
                {"context_key": p.context_key,
                 "action_weights": {str(k): v for k, v in p.action_weights.items()},
                 "update_count": p.update_count, "strength": p.strength}
                for p in self._preferences.values()
            ],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("w") as f:
            json.dump(data, f)
        logger.info("MemoryStore saved to %s", path)

    def load(self, path: str) -> None:
        """Load the memory store from a JSON file."""
        with Path(path).open() as f:
            data = json.load(f)
        self.agent_id = data["agent_id"]

        known_top_keys = {"agent_id", "semantic", "episodic", "preferences"}
        for key in data:
            if key not in known_top_keys:
                logger.warning("Unknown field '%s' in memory store file %s", key, path)

        semantic_fields = set(SemanticFact.__dataclass_fields__)
        self._semantic = [
            SemanticFact(**{k: v for k, v in f.items() if k in semantic_fields})
            for f in data.get("semantic", [])
        ]
        episodic_fields = set(Episode.__dataclass_fields__)
        self._episodic = [
            Episode(**{k: v for k, v in e.items() if k in episodic_fields})
            for e in data.get("episodic", [])
        ]
        self._preferences = {}
        for p in data.get("preferences", []):
            pref = Preference(
                context_key=p["context_key"],
                action_weights={int(k): v for k, v in p.get("action_weights", {}).items()},
                update_count=p.get("update_count", 0),
                strength=p.get("strength", 1.0),
            )
            self._preferences[pref.context_key] = pref

        logger.info("MemoryStore loaded from %s", path)
