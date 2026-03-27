//! Configuration for the agent memory system.
//!
//! All memory parameters are configurable through [`MemoryConfig`].
//! No hard-coded values — defaults are provided via `Default` trait.

use serde::{Deserialize, Serialize};

/// Default maximum number of semantic facts.
const DEFAULT_SEMANTIC_CAPACITY: usize = 10_000;
/// Default maximum number of episodic memories.
const DEFAULT_EPISODIC_CAPACITY: usize = 5_000;
/// Default maximum number of preference entries.
const DEFAULT_PREFERENCE_CAPACITY: usize = 1_000;
/// Default memory decay rate per tick.
const DEFAULT_DECAY_RATE: f32 = 0.001;
/// Default number of memories to retrieve per query.
const DEFAULT_RETRIEVAL_TOP_K: usize = 10;
/// Default minimum strength threshold for retrieval.
const DEFAULT_MIN_STRENGTH: f32 = 0.01;
/// Default reinforcement strength increment per access (semantic facts).
const DEFAULT_REINFORCEMENT_INCREMENT: f32 = 0.1;
/// Default preference reinforcement strength increment per update.
const DEFAULT_PREFERENCE_REINFORCEMENT_INCREMENT: f32 = 0.05;

/// Configuration for the persistent agent memory system.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MemoryConfig {
    /// Whether the memory system is enabled.
    pub enabled: bool,
    /// Maximum number of semantic facts stored.
    pub semantic_capacity: usize,
    /// Maximum number of episodic memories stored.
    pub episodic_capacity: usize,
    /// Maximum number of preference entries stored.
    pub preference_capacity: usize,
    /// Memory strength decay rate per tick (0.0 = no decay).
    pub decay_rate: f32,
    /// Number of memories returned per query.
    pub retrieval_top_k: usize,
    /// Minimum strength threshold; memories below this are pruned.
    pub min_strength: f32,
    /// Strength increment when a semantic fact is reinforced.
    pub reinforcement_increment: f32,
    /// Strength increment when a preference is updated.
    pub preference_reinforcement_increment: f32,
    /// File path for persistent storage. `None` = in-memory only.
    pub persistence_path: Option<String>,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            semantic_capacity: DEFAULT_SEMANTIC_CAPACITY,
            episodic_capacity: DEFAULT_EPISODIC_CAPACITY,
            preference_capacity: DEFAULT_PREFERENCE_CAPACITY,
            decay_rate: DEFAULT_DECAY_RATE,
            retrieval_top_k: DEFAULT_RETRIEVAL_TOP_K,
            min_strength: DEFAULT_MIN_STRENGTH,
            reinforcement_increment: DEFAULT_REINFORCEMENT_INCREMENT,
            preference_reinforcement_increment: DEFAULT_PREFERENCE_REINFORCEMENT_INCREMENT,
            persistence_path: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = MemoryConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.semantic_capacity, DEFAULT_SEMANTIC_CAPACITY);
        assert_eq!(config.episodic_capacity, DEFAULT_EPISODIC_CAPACITY);
        assert_eq!(config.preference_capacity, DEFAULT_PREFERENCE_CAPACITY);
        assert!(config.persistence_path.is_none());
    }

    #[test]
    fn test_serialization_roundtrip() {
        forge_types::assert_config_serde_roundtrip!(MemoryConfig);
    }

    #[test]
    fn test_defaults_valid() {
        forge_types::assert_config_defaults_valid!(MemoryConfig);
    }
}
