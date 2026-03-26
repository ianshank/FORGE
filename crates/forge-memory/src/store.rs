//! Memory store trait and in-memory implementation.
//!
//! The [`InMemoryStore`] provides a complete in-memory implementation with
//! optional persistence to disk via bincode serialization.

use std::path::Path;

use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::config::MemoryConfig;
use crate::episodic::{Episode, EpisodicMemory};
use crate::error::MemoryError;
use crate::preference::PreferenceMemory;
use crate::semantic::{SemanticFact, SemanticMemory};

/// Query parameters for memory retrieval.
#[derive(Debug, Clone)]
pub enum MemoryQuery {
    /// Query semantic facts by key prefix.
    SemanticByKey(String),
    /// Query episodes involving a specific agent.
    EpisodicByAgent(u32),
    /// Query episodes by tag.
    EpisodicByTag(String),
    /// Query episodes near a location.
    EpisodicByLocation {
        /// X coordinate.
        x: u16,
        /// Y coordinate.
        y: u16,
        /// Search radius.
        radius: u16,
    },
    /// Query the preferred action for a context.
    PreferenceByContext(String),
}

/// A retrieved memory entry.
#[derive(Debug, Clone)]
pub enum MemoryEntry {
    /// A semantic fact.
    Semantic(SemanticFact),
    /// An episodic memory.
    Episodic(Episode),
    /// A preferred action ID for a context.
    PreferredAction {
        /// Context key.
        context: String,
        /// Preferred action ID.
        action_id: u32,
    },
}

/// Complete in-memory store combining semantic, episodic, and preference memories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InMemoryStore {
    /// Semantic memory subsystem.
    pub semantic: SemanticMemory,
    /// Episodic memory subsystem.
    pub episodic: EpisodicMemory,
    /// Preference memory subsystem.
    pub preferences: PreferenceMemory,
    /// Agent identifier this store belongs to.
    pub agent_id: u32,
}

impl InMemoryStore {
    /// Creates a new in-memory store for the given agent with the provided config.
    #[instrument(skip_all)]
    pub fn new(agent_id: u32, config: &MemoryConfig) -> Self {
        Self {
            semantic: SemanticMemory::new(config.semantic_capacity),
            episodic: EpisodicMemory::new(config.episodic_capacity),
            preferences: PreferenceMemory::new(config.preference_capacity),
            agent_id,
        }
    }

    /// Queries the store and returns matching entries.
    #[instrument(skip_all)]
    pub fn query(&self, query: &MemoryQuery, top_k: usize) -> Vec<MemoryEntry> {
        match query {
            MemoryQuery::SemanticByKey(prefix) => self
                .semantic
                .query_by_prefix(prefix, top_k)
                .into_iter()
                .map(|f| MemoryEntry::Semantic(f.clone()))
                .collect(),
            MemoryQuery::EpisodicByAgent(agent_id) => self
                .episodic
                .query_by_agent(*agent_id)
                .into_iter()
                .take(top_k)
                .map(|e| MemoryEntry::Episodic(e.clone()))
                .collect(),
            MemoryQuery::EpisodicByTag(tag) => self
                .episodic
                .query_by_tag(tag)
                .into_iter()
                .take(top_k)
                .map(|e| MemoryEntry::Episodic(e.clone()))
                .collect(),
            MemoryQuery::EpisodicByLocation { x, y, radius } => self
                .episodic
                .query_by_location(*x, *y, *radius)
                .into_iter()
                .take(top_k)
                .map(|e| MemoryEntry::Episodic(e.clone()))
                .collect(),
            MemoryQuery::PreferenceByContext(context) => self
                .preferences
                .get(context)
                .and_then(|p| p.preferred_action().map(|a| (context.clone(), a)))
                .into_iter()
                .map(|(ctx, action_id)| MemoryEntry::PreferredAction {
                    context: ctx,
                    action_id,
                })
                .collect(),
        }
    }

    /// Applies decay to all memory subsystems.
    #[instrument(skip_all)]
    pub fn tick_decay(&mut self, config: &MemoryConfig) {
        self.semantic
            .tick_decay(config.decay_rate, config.min_strength);
        self.episodic
            .tick_decay(config.decay_rate, config.min_strength);
        self.preferences
            .tick_decay(config.decay_rate, config.min_strength);
    }

    /// Saves the store to disk as bincode.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::Serialize`] if serialization fails,
    /// or [`MemoryError::Io`] if the file cannot be written.
    #[instrument(skip(self))]
    pub fn save_to_file(&self, path: &Path) -> Result<(), MemoryError> {
        let data = bincode::serialize(self).map_err(|e| MemoryError::Serialize(e.to_string()))?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// Loads the store from a bincode file.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::Io`] if the file cannot be read,
    /// or [`MemoryError::Deserialize`] if deserialization fails.
    #[instrument]
    pub fn load_from_file(path: &Path) -> Result<Self, MemoryError> {
        let data = std::fs::read(path)?;
        let store =
            bincode::deserialize(&data).map_err(|e| MemoryError::Deserialize(e.to_string()))?;
        Ok(store)
    }

    /// Returns total memory entries across all subsystems.
    pub fn total_entries(&self) -> usize {
        self.semantic.len() + self.episodic.len() + self.preferences.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::episodic::EpisodeOutcome;

    fn test_config() -> MemoryConfig {
        MemoryConfig {
            enabled: true,
            semantic_capacity: 100,
            episodic_capacity: 50,
            preference_capacity: 20,
            ..MemoryConfig::default()
        }
    }

    #[test]
    fn test_new_store() {
        let store = InMemoryStore::new(0, &test_config());
        assert_eq!(store.agent_id, 0);
        assert_eq!(store.total_entries(), 0);
    }

    #[test]
    fn test_semantic_query() {
        let mut store = InMemoryStore::new(0, &test_config());
        store.semantic.store(SemanticFact::new(
            "agent_1.role".into(),
            "scout".into(),
            0.9,
            1,
        ));
        store.semantic.store(SemanticFact::new(
            "agent_2.role".into(),
            "builder".into(),
            0.8,
            2,
        ));

        let results = store.query(&MemoryQuery::SemanticByKey("agent_1".into()), 10);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_episodic_query() {
        let mut store = InMemoryStore::new(0, &test_config());
        let ep = Episode::new((0, 10), vec![0, 1], (5, 5), EpisodeOutcome::Success, 1.0);
        store.episodic.store(ep);

        let results = store.query(&MemoryQuery::EpisodicByAgent(0), 10);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_preference_query() {
        let mut store = InMemoryStore::new(0, &test_config());
        let pref = store.preferences.get_or_create("combat");
        pref.update_action(5, 1.0, 0.5, 0.05);

        let results = store.query(&MemoryQuery::PreferenceByContext("combat".into()), 10);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let dir = std::env::temp_dir().join("forge_memory_test_store");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_store.bin");

        let mut store = InMemoryStore::new(7, &test_config());
        store
            .semantic
            .store(SemanticFact::new("key".into(), "val".into(), 0.9, 1));
        store.save_to_file(&path).unwrap();

        let loaded = InMemoryStore::load_from_file(&path).unwrap();
        assert_eq!(loaded.agent_id, 7);
        assert_eq!(loaded.total_entries(), 1);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_load_nonexistent_file() {
        let result = InMemoryStore::load_from_file(Path::new("/tmp/nonexistent_forge_test.bin"));
        assert!(result.is_err());
    }

    #[test]
    fn test_tick_decay_reduces_entries() {
        let mut config = test_config();
        config.decay_rate = 0.1; // faster decay for test
        config.min_strength = 0.01;
        let mut store = InMemoryStore::new(0, &config);

        // Add a fact with low strength
        store
            .semantic
            .store(SemanticFact::new("weak".into(), "value".into(), 0.05, 1));
        assert_eq!(store.total_entries(), 1);

        // Decay enough times to drop below min_strength
        for _ in 0..100 {
            store.tick_decay(&config);
        }
        // After enough decay, the weak fact should be pruned
        assert_eq!(store.total_entries(), 0);
    }

    #[test]
    fn test_episodic_query_by_tag() {
        let mut store = InMemoryStore::new(0, &test_config());
        let mut ep = Episode::new((0, 5), vec![0], (3, 3), EpisodeOutcome::Success, 1.0);
        ep.tags.push("combat".into());
        store.episodic.store(ep);

        let results = store.query(&MemoryQuery::EpisodicByTag("combat".into()), 10);
        assert_eq!(results.len(), 1);

        let results = store.query(&MemoryQuery::EpisodicByTag("explore".into()), 10);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_episodic_query_by_location() {
        let mut store = InMemoryStore::new(0, &test_config());
        let ep = Episode::new((0, 5), vec![0], (10, 10), EpisodeOutcome::Success, 1.0);
        store.episodic.store(ep);

        // Nearby query should find it
        let results = store.query(
            &MemoryQuery::EpisodicByLocation {
                x: 10,
                y: 10,
                radius: 5,
            },
            10,
        );
        assert_eq!(results.len(), 1);

        // Far-away query should not
        let results = store.query(
            &MemoryQuery::EpisodicByLocation {
                x: 100,
                y: 100,
                radius: 5,
            },
            10,
        );
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_empty_query_returns_empty() {
        let store = InMemoryStore::new(0, &test_config());
        let results = store.query(&MemoryQuery::SemanticByKey("nonexistent".into()), 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_empty_store_all_query_types_return_empty() {
        let store = InMemoryStore::new(0, &test_config());

        assert!(store
            .query(&MemoryQuery::SemanticByKey("any".into()), 10)
            .is_empty());
        assert!(store
            .query(&MemoryQuery::EpisodicByAgent(0), 10)
            .is_empty());
        assert!(store
            .query(&MemoryQuery::EpisodicByTag("combat".into()), 10)
            .is_empty());
        assert!(store
            .query(
                &MemoryQuery::EpisodicByLocation {
                    x: 5,
                    y: 5,
                    radius: 10,
                },
                10,
            )
            .is_empty());
        assert!(store
            .query(&MemoryQuery::PreferenceByContext("ctx".into()), 10)
            .is_empty());
    }

    #[test]
    fn test_save_load_roundtrip_preserves_all_subsystems() {
        let dir = std::env::temp_dir().join("forge_memory_test_full");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("full_store.bin");

        let mut store = InMemoryStore::new(3, &test_config());
        store
            .semantic
            .store(SemanticFact::new("role".into(), "scout".into(), 0.9, 1));
        let ep = Episode::new((0, 5), vec![3], (7, 7), EpisodeOutcome::Failure, 0.5);
        store.episodic.store(ep);
        let pref = store.preferences.get_or_create("navigate");
        pref.update_action(2, 1.0, 0.5, 0.05);

        assert_eq!(store.total_entries(), 3);

        store.save_to_file(&path).unwrap();
        let loaded = InMemoryStore::load_from_file(&path).unwrap();

        assert_eq!(loaded.agent_id, 3);
        assert_eq!(loaded.total_entries(), 3);
        assert_eq!(loaded.semantic.len(), 1);
        assert_eq!(loaded.episodic.len(), 1);
        assert_eq!(loaded.preferences.len(), 1);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_decay_with_very_small_rate() {
        let mut config = test_config();
        config.decay_rate = 1e-7;
        config.min_strength = 0.01;
        let mut store = InMemoryStore::new(0, &config);

        store
            .semantic
            .store(SemanticFact::new("fact".into(), "val".into(), 1.0, 1));
        let ep = Episode::new((0, 5), vec![0], (3, 3), EpisodeOutcome::Success, 1.0);
        store.episodic.store(ep);
        let pref = store.preferences.get_or_create("ctx");
        pref.update_action(1, 1.0, 0.5, 0.05);

        assert_eq!(store.total_entries(), 3);

        // Even after many ticks, tiny decay should not prune strong memories.
        for _ in 0..1000 {
            store.tick_decay(&config);
        }
        // 1.0 - 1000 * 1e-7 = 0.9999 — well above min_strength.
        assert_eq!(store.total_entries(), 3);
    }

    #[test]
    fn test_total_entries_across_all_subsystems() {
        let mut store = InMemoryStore::new(0, &test_config());
        assert_eq!(store.total_entries(), 0);

        store
            .semantic
            .store(SemanticFact::new("s1".into(), "v".into(), 1.0, 1));
        store
            .semantic
            .store(SemanticFact::new("s2".into(), "v".into(), 1.0, 2));
        assert_eq!(store.total_entries(), 2);

        let ep = Episode::new((0, 5), vec![0], (0, 0), EpisodeOutcome::Neutral, 0.0);
        store.episodic.store(ep);
        assert_eq!(store.total_entries(), 3);

        store.preferences.get_or_create("pref1");
        store.preferences.get_or_create("pref2");
        assert_eq!(store.total_entries(), 5);
    }

    #[test]
    fn test_save_load_roundtrip_large_data() {
        let dir = std::env::temp_dir().join("forge_memory_test_large");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("large_store.bin");

        let config = MemoryConfig {
            enabled: true,
            semantic_capacity: 500,
            episodic_capacity: 500,
            preference_capacity: 500,
            ..MemoryConfig::default()
        };
        let mut store = InMemoryStore::new(42, &config);

        // Populate with many entries.
        for i in 0..200 {
            store.semantic.store(SemanticFact::new(
                format!("key_{i}"),
                format!("value_{i}"),
                (i as f32) / 200.0,
                i as u64,
            ));
        }
        for i in 0..150 {
            let mut ep = Episode::new(
                (i as u64, i as u64 + 10),
                vec![i as u32 % 10],
                ((i as u16) % 100, (i as u16) % 100),
                EpisodeOutcome::Success,
                i as f32 * 0.1,
            );
            ep.tags.push(format!("tag_{}", i % 5));
            ep.event_summaries.push(format!("event_{i}"));
            store.episodic.store(ep);
        }
        for i in 0..50 {
            let pref = store.preferences.get_or_create(&format!("ctx_{i}"));
            pref.update_action(i as u32, 1.0, 0.5, 0.05);
        }

        let total_before = store.total_entries();
        assert_eq!(total_before, 200 + 150 + 50);

        store.save_to_file(&path).unwrap();
        let loaded = InMemoryStore::load_from_file(&path).unwrap();

        assert_eq!(loaded.agent_id, 42);
        assert_eq!(loaded.total_entries(), total_before);
        assert_eq!(loaded.semantic.len(), 200);
        assert_eq!(loaded.episodic.len(), 150);
        assert_eq!(loaded.preferences.len(), 50);

        // Spot-check a few entries survived the roundtrip.
        assert!(loaded.semantic.get("key_0").is_some());
        assert!(loaded.semantic.get("key_199").is_some());
        assert_eq!(
            loaded.preferences.get("ctx_0").unwrap().preferred_action(),
            Some(0)
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}
