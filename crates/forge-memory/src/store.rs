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
                .query_by_strength(0.0)
                .into_iter()
                .filter(|f| f.key.starts_with(prefix.as_str()))
                .take(top_k)
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
}
