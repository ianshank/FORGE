"""Tests for the memory store module."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from forge.memory.memory_store import (
    Episode,
    MemoryStore,
    MemoryStoreConfig,
    Preference,
    SemanticFact,
)


class TestSemanticFact:
    """Tests for SemanticFact dataclass."""

    def test_default_values(self) -> None:
        fact = SemanticFact(key="agent.role", value="scout")
        assert fact.confidence == 1.0
        assert fact.strength == 1.0
        assert fact.reinforcement_count == 0
        assert fact.source_tick == 0

    def test_custom_values(self) -> None:
        fact = SemanticFact(key="k", value="v", confidence=0.5, source_tick=42)
        assert fact.confidence == 0.5
        assert fact.source_tick == 42


class TestEpisode:
    """Tests for Episode dataclass."""

    def test_default_values(self) -> None:
        ep = Episode()
        assert ep.tick_start == 0
        assert ep.tick_end == 0
        assert ep.outcome == "neutral"
        assert ep.reward == 0.0
        assert ep.strength == 1.0

    def test_custom_values(self) -> None:
        ep = Episode(
            tick_start=10,
            tick_end=20,
            agent_ids=[1, 2],
            outcome="success",
            reward=5.0,
            tags=["combat"],
        )
        assert ep.tick_start == 10
        assert ep.agent_ids == [1, 2]
        assert ep.tags == ["combat"]


class TestPreference:
    """Tests for Preference dataclass."""

    def test_update(self) -> None:
        pref = Preference(context_key="combat")
        pref.update(1, 1.0)
        pref.update(2, 0.5)
        assert pref.preferred_action() == 1
        assert pref.update_count == 2

    def test_update_ema(self) -> None:
        pref = Preference(context_key="ctx")
        pref.update(0, 1.0, lr=0.5)
        pref.update(0, 0.0, lr=0.5)
        # After two updates: first=1.0, second=1.0*0.5 + 0.0*0.5 = 0.5
        assert abs(pref.action_weights[0] - 0.5) < 0.01

    def test_preferred_action_empty(self) -> None:
        pref = Preference(context_key="empty")
        assert pref.preferred_action() is None

    def test_update_new_action(self) -> None:
        pref = Preference(context_key="ctx")
        pref.update(5, 0.8)
        assert pref.action_weights[5] == 0.8


class TestMemoryStoreConfig:
    """Tests for MemoryStoreConfig."""

    def test_defaults(self) -> None:
        config = MemoryStoreConfig()
        assert config.semantic_capacity == 10_000
        assert config.episodic_capacity == 5_000
        assert config.preference_capacity == 1_000
        assert config.decay_rate == 0.001
        assert config.min_strength == 0.01


class TestMemoryStore:
    """Tests for MemoryStore."""

    def test_creation(self) -> None:
        store = MemoryStore(agent_id=0)
        assert store.agent_id == 0
        assert store.total_entries() == 0

    def test_store_and_query_fact(self) -> None:
        store = MemoryStore(agent_id=0)
        store.store_fact(SemanticFact(key="agent_1.role", value="scout"))
        store.store_fact(SemanticFact(key="agent_2.role", value="builder"))

        results = store.query_facts(prefix="agent_1")
        assert len(results) == 1
        assert results[0].value == "scout"

    def test_replace_existing_fact(self) -> None:
        store = MemoryStore(agent_id=0)
        store.store_fact(SemanticFact(key="k", value="v1"))
        store.store_fact(SemanticFact(key="k", value="v2"))
        assert store.total_entries() == 1
        assert store.query_facts(prefix="k")[0].value == "v2"

    def test_semantic_eviction_at_capacity(self) -> None:
        config = MemoryStoreConfig(semantic_capacity=2)
        store = MemoryStore(agent_id=0, config=config)
        store.store_fact(SemanticFact(key="a", value="1", strength=0.1))
        store.store_fact(SemanticFact(key="b", value="2"))
        store.store_fact(SemanticFact(key="c", value="3"))
        facts = store.query_facts()
        assert len(facts) == 2
        keys = {f.key for f in facts}
        assert "a" not in keys  # weakest evicted

    def test_store_and_query_episodes(self) -> None:
        store = MemoryStore(agent_id=0)
        store.store_episode(Episode(agent_ids=[0, 1], tags=["combat"]))
        store.store_episode(Episode(agent_ids=[0, 2], tags=["crafting"]))

        assert len(store.query_episodes(agent_id=0)) == 2
        assert len(store.query_episodes(agent_id=1)) == 1
        assert len(store.query_episodes(tag="combat")) == 1

    def test_recent_episodes(self) -> None:
        store = MemoryStore(agent_id=0)
        for i in range(5):
            store.store_episode(Episode(tick_start=i * 10))
        recent = store.recent_episodes(n=3)
        assert len(recent) == 3

    def test_episodic_eviction(self) -> None:
        config = MemoryStoreConfig(episodic_capacity=2)
        store = MemoryStore(agent_id=0, config=config)
        store.store_episode(Episode(strength=0.1))
        store.store_episode(Episode(tick_start=10))
        store.store_episode(Episode(tick_start=20))
        assert len(store.query_episodes()) == 2

    def test_get_preference(self) -> None:
        store = MemoryStore(agent_id=0)
        pref = store.get_preference("combat.low_health")
        pref.update(3, 1.0)
        retrieved = store.get_preference("combat.low_health")
        assert retrieved.preferred_action() == 3

    def test_tick_decay(self) -> None:
        config = MemoryStoreConfig(decay_rate=0.5, min_strength=0.4)
        store = MemoryStore(agent_id=0, config=config)
        store.store_fact(SemanticFact(key="weak", value="v", strength=0.45))
        store.store_fact(SemanticFact(key="strong", value="v", strength=1.0))
        store.tick_decay()
        # weak: 0.45 - 0.5 = 0.0 < 0.4 → pruned
        # strong: 1.0 - 0.5 = 0.5 >= 0.4 → kept
        facts = store.query_facts()
        assert len(facts) == 1
        assert facts[0].key == "strong"

    def test_total_entries(self) -> None:
        store = MemoryStore(agent_id=0)
        store.store_fact(SemanticFact(key="k", value="v"))
        store.store_episode(Episode())
        store.get_preference("ctx")
        assert store.total_entries() == 3

    def test_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test_store.json")
            store = MemoryStore(agent_id=7)
            store.store_fact(SemanticFact(key="k", value="v", confidence=0.9))
            store.store_episode(
                Episode(tick_start=0, tick_end=10, outcome="success", tags=["nav"])
            )
            pref = store.get_preference("combat")
            pref.update(5, 1.0)
            store.save(path)

            loaded = MemoryStore(agent_id=0)
            loaded.load(path)
            assert loaded.agent_id == 7
            assert len(loaded.query_facts()) == 1
            assert len(loaded.query_episodes()) == 1
            # Verify preferences were saved and loaded
            loaded_pref = loaded.get_preference("combat")
            assert loaded_pref.preferred_action() == 5
            assert loaded_pref.update_count == 1

    def test_save_creates_parent_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "nested" / "dir" / "store.json")
            store = MemoryStore(agent_id=0)
            store.save(path)
            assert Path(path).exists()

    def test_query_facts_by_min_strength(self) -> None:
        store = MemoryStore(agent_id=0)
        store.store_fact(SemanticFact(key="a", value="1", strength=0.3))
        store.store_fact(SemanticFact(key="b", value="2", strength=0.8))
        results = store.query_facts(min_strength=0.5)
        assert len(results) == 1
        assert results[0].key == "b"


class TestMemoryStoreEdgeCases:
    """Edge case tests for MemoryStore."""

    def test_empty_store_queries(self) -> None:
        store = MemoryStore(agent_id=0)
        assert store.query_facts() == []
        assert store.query_episodes() == []
        assert store.recent_episodes() == []

    def test_load_file_not_found(self) -> None:
        store = MemoryStore(agent_id=0)
        with pytest.raises(FileNotFoundError):
            store.load("/tmp/nonexistent_forge_test_12345.json")

    def test_query_episodes_combined_filters(self) -> None:
        store = MemoryStore(agent_id=0)
        store.store_episode(Episode(agent_ids=[1], tags=["combat"]))
        store.store_episode(Episode(agent_ids=[2], tags=["combat"]))
        store.store_episode(Episode(agent_ids=[1], tags=["nav"]))
        results = store.query_episodes(agent_id=1, tag="combat")
        assert len(results) == 1
