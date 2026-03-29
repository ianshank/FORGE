//! Compact deterministic replay format.
//!
//! Given FORGE's byte-identical determinism (same seed + config + actions =
//! identical state), a compact replay stores only the minimal information
//! needed to reconstruct the full simulation:
//!
//! - The [`ForgeConfig`] (for self-contained replays)
//! - A config hash (for quick validation)
//! - The seed
//! - Per-tick action IDs for each agent
//!
//! A 10,000-tick episode with 4 agents compresses to ~160 KB in bincode
//! vs. ~50 MB for full trajectory data.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use forge_core::WorldState;
use forge_types::agent_interface::AgentMetadata;
use forge_types::config::ForgeConfig;
use forge_types::observation::StepResult;
use forge_types::Action;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument, warn};

/// Current compact replay format version.
pub const FORMAT_VERSION: u32 = 1;

/// Compact deterministic replay.
///
/// Contains enough information to reconstruct the full simulation state
/// at any tick by replaying actions from the seed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactReplay {
    /// Format version for forward compatibility.
    pub format_version: u32,
    /// Hash of the ForgeConfig for quick validation.
    pub config_hash: u64,
    /// Full config for self-contained replay.
    pub config: ForgeConfig,
    /// Seed used for world generation and RNG.
    pub seed: u64,
    /// Per-tick action IDs: `actions[tick][agent_idx]`.
    pub actions: Vec<Vec<u32>>,
    /// Metadata about the replay.
    pub metadata: ReplayMetadata,
}

/// Metadata about a replay recording.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReplayMetadata {
    /// Names of agents involved.
    pub agent_names: Vec<String>,
    /// Metadata for each agent (type, model, version).
    pub agent_metadata: Vec<AgentMetadata>,
    /// Total ticks recorded.
    pub total_ticks: u64,
    /// Final per-agent rewards.
    pub final_rewards: Vec<f32>,
    /// IDs of tasks completed during the episode.
    pub completed_tasks: Vec<u64>,
    /// ISO 8601 timestamp of recording.
    pub timestamp: String,
    /// Optional scenario identifier.
    pub scenario_id: Option<String>,
}

impl CompactReplay {
    /// Creates a new compact replay builder.
    #[instrument(skip_all)]
    pub fn builder(config: ForgeConfig, seed: u64) -> CompactReplayBuilder {
        CompactReplayBuilder::new(config, seed)
    }

    /// Serializes to bincode bytes.
    ///
    /// Returns an error if serialization fails (e.g., extremely large replays).
    #[instrument(skip_all)]
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        bincode::serialize(self).map_err(|e| {
            warn!(error = %e, "Bincode serialization failed");
            format!("serialization failed: {e}")
        })
    }

    /// Deserializes from bincode bytes.
    #[instrument(skip_all)]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let replay: Self =
            bincode::deserialize(bytes).map_err(|e| format!("deserialization failed: {e}"))?;
        debug!(
            format_version = replay.format_version,
            ticks = replay.metadata.total_ticks,
            "Deserialized compact replay"
        );
        Ok(replay)
    }

    /// Serializes to pretty-printed JSON string.
    #[instrument(skip_all)]
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| {
            warn!(error = %e, "JSON serialization failed");
            format!("JSON serialization failed: {e}")
        })
    }

    /// Deserializes from JSON string.
    #[instrument(skip_all)]
    pub fn from_json(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| format!("JSON deserialization failed: {e}"))
    }

    /// Validates that the config hash matches the stored config.
    #[instrument(skip_all)]
    pub fn validate_config(&self) -> bool {
        hash_config(&self.config) == self.config_hash
    }

    /// Replays the episode, yielding `StepResult` at each tick.
    ///
    /// Returns `None` if the config doesn't match or world creation fails.
    #[instrument(skip_all)]
    pub fn replay(&self) -> Option<ReplayIterator<'_>> {
        if !self.validate_config() {
            warn!("Config hash mismatch during replay");
            return None;
        }

        if self.config.world.seed != self.seed {
            warn!(
                config_seed = self.config.world.seed,
                replay_seed = self.seed,
                "Compact replay stored mismatched config and replay seed; replay will use replay seed"
            );
        }

        let mut replay_config = self.config.clone();
        replay_config.world.seed = self.seed;

        let world = match WorldState::new(replay_config) {
            Ok(w) => w,
            Err(e) => {
                warn!(error = %e, "Failed to create world for replay");
                return None;
            }
        };

        info!(
            ticks = self.actions.len(),
            seed = self.seed,
            "Starting replay"
        );

        Some(ReplayIterator {
            world,
            actions: &self.actions,
            tick: 0,
            comm_vocab: self.config.agents.comm_vocab_size,
            drone_enabled: self.config.drone.enabled,
        })
    }
}

/// Iterator that replays a compact replay tick-by-tick.
pub struct ReplayIterator<'a> {
    world: WorldState,
    actions: &'a [Vec<u32>],
    tick: usize,
    comm_vocab: u16,
    drone_enabled: bool,
}

impl<'a> Iterator for ReplayIterator<'a> {
    type Item = StepResult;

    fn next(&mut self) -> Option<StepResult> {
        if self.tick >= self.actions.len() {
            return None;
        }

        if self.world.terminated || self.world.truncated {
            return None;
        }

        let action_ids = &self.actions[self.tick];
        let actions: Vec<Action> = action_ids
            .iter()
            .map(|&id| {
                Action::from_discrete(id, self.comm_vocab, self.drone_enabled)
                    .unwrap_or(Action::Noop)
            })
            .collect();

        let result = self.world.step(&actions);
        self.tick += 1;
        Some(result)
    }
}

impl<'a> ReplayIterator<'a> {
    /// Returns the current tick.
    pub fn current_tick(&self) -> u64 {
        self.tick as u64
    }

    /// Returns a reference to the current world state.
    pub fn world(&self) -> &WorldState {
        &self.world
    }
}

/// Builder for constructing compact replays during an episode.
pub struct CompactReplayBuilder {
    config: ForgeConfig,
    seed: u64,
    actions: Vec<Vec<u32>>,
    metadata: ReplayMetadata,
}

impl CompactReplayBuilder {
    /// Creates a new builder.
    fn new(mut config: ForgeConfig, seed: u64) -> Self {
        config.world.seed = seed;

        Self {
            config,
            seed,
            actions: Vec::new(),
            metadata: ReplayMetadata::default(),
        }
    }

    /// Records one tick of actions.
    pub fn record_tick(&mut self, action_ids: Vec<u32>) {
        self.actions.push(action_ids);
    }

    /// Sets agent names.
    pub fn agent_names(mut self, names: Vec<String>) -> Self {
        self.metadata.agent_names = names;
        self
    }

    /// Sets agent metadata.
    pub fn agent_metadata(mut self, metadata: Vec<AgentMetadata>) -> Self {
        self.metadata.agent_metadata = metadata;
        self
    }

    /// Sets final rewards.
    pub fn final_rewards(mut self, rewards: Vec<f32>) -> Self {
        self.metadata.final_rewards = rewards;
        self
    }

    /// Sets scenario ID.
    pub fn scenario_id(mut self, id: String) -> Self {
        self.metadata.scenario_id = Some(id);
        self
    }

    /// Builds the compact replay.
    #[instrument(skip_all)]
    pub fn build(mut self) -> CompactReplay {
        self.metadata.total_ticks = self.actions.len() as u64;
        self.metadata.timestamp = chrono::Utc::now().to_rfc3339();

        debug!(
            ticks = self.metadata.total_ticks,
            agents = self.metadata.agent_names.len(),
            "Building compact replay"
        );

        CompactReplay {
            format_version: FORMAT_VERSION,
            config_hash: hash_config(&self.config),
            config: self.config,
            seed: self.seed,
            actions: self.actions,
            metadata: self.metadata,
        }
    }
}

/// Computes a deterministic hash of a ForgeConfig.
fn hash_config(config: &ForgeConfig) -> u64 {
    let json = serde_json::to_string(config).unwrap_or_default();
    let mut hasher = DefaultHasher::new();
    json.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> ForgeConfig {
        let mut config = ForgeConfig::default();
        config.world.width = 16;
        config.world.height = 16;
        config.world.seed = 42;
        config.agents.num_agents = 1;
        config.agents.comm_vocab_size = 0;
        config.task.max_episode_length = 100;
        config
    }

    #[test]
    fn test_compact_replay_builder() {
        let config = test_config();
        let mut builder = CompactReplay::builder(config, 42);
        builder.record_tick(vec![0]); // Noop
        builder.record_tick(vec![1]); // Move Up

        let replay = builder
            .agent_names(vec!["TestAgent".to_string()])
            .final_rewards(vec![1.5])
            .build();

        assert_eq!(replay.format_version, FORMAT_VERSION);
        assert_eq!(replay.seed, 42);
    assert_eq!(replay.config.world.seed, 42);
        assert_eq!(replay.actions.len(), 2);
        assert_eq!(replay.metadata.total_ticks, 2);
        assert_eq!(replay.metadata.agent_names, vec!["TestAgent"]);
        assert_eq!(replay.metadata.final_rewards, vec![1.5]);
        assert!(!replay.metadata.timestamp.is_empty());
    }

    #[test]
    fn test_config_hash_validation() {
        let config = test_config();
        let replay = CompactReplay::builder(config, 42).build();
        assert!(replay.validate_config());
    }

    #[test]
    fn test_config_hash_mismatch() {
        let config = test_config();
        let mut replay = CompactReplay::builder(config, 42).build();
        // Corrupt the hash
        replay.config_hash = 0;
        assert!(!replay.validate_config());
    }

    #[test]
    fn test_bincode_roundtrip() {
        let config = test_config();
        let mut builder = CompactReplay::builder(config, 42);
        builder.record_tick(vec![0]);
        builder.record_tick(vec![1]);
        let replay = builder.build();

        let bytes = replay.to_bytes().unwrap();
        let deserialized = CompactReplay::from_bytes(&bytes).unwrap();

        assert_eq!(deserialized.seed, 42);
        assert_eq!(deserialized.actions.len(), 2);
        assert_eq!(deserialized.config_hash, replay.config_hash);
        assert!(deserialized.validate_config());
    }

    #[test]
    fn test_json_roundtrip() {
        let config = test_config();
        let mut builder = CompactReplay::builder(config, 42);
        builder.record_tick(vec![0, 1]);
        let replay = builder.build();

        let json = replay.to_json().unwrap();
        let deserialized = CompactReplay::from_json(&json).unwrap();

        assert_eq!(deserialized.seed, replay.seed);
        assert_eq!(deserialized.actions, replay.actions);
    }

    #[test]
    fn test_builder_normalizes_config_seed() {
        let mut config = test_config();
        config.world.seed = 7;

        let replay = CompactReplay::builder(config, 42).build();

        assert_eq!(replay.seed, 42);
        assert_eq!(replay.config.world.seed, 42);
        assert!(replay.validate_config());
    }

    #[test]
    fn test_replay_uses_stored_replay_seed() {
        let config = test_config();
        let mut replay = CompactReplay::builder(config, 42).build();

        replay.config.world.seed = 7;
        replay.config_hash = hash_config(&replay.config);

        let replay_iter = replay.replay().unwrap();
        assert_eq!(replay_iter.world().config.world.seed, 42);
    }

    #[test]
    fn test_replay_iterator() {
        let config = test_config();
        let mut world = WorldState::new(config.clone()).unwrap();

        // Record a real episode
        let mut builder = CompactReplay::builder(config, 42);
        let actions_sequence = vec![
            vec![0u32], // Noop
            vec![1],    // Move Up
            vec![0],    // Noop
        ];

        for action_ids in &actions_sequence {
            builder.record_tick(action_ids.clone());
            let actions: Vec<Action> = action_ids
                .iter()
                .map(|&id| Action::from_discrete(id, 0, false).unwrap_or(Action::Noop))
                .collect();
            world.step(&actions);
        }

        let replay = builder.build();

        // Replay and verify we get the same number of steps
        let results: Vec<_> = replay.replay().unwrap().collect();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_replay_determinism() {
        let config = test_config();

        // Record episode
        let mut world1 = WorldState::new(config.clone()).unwrap();
        let mut builder = CompactReplay::builder(config, 42);

        let action_seq = vec![vec![0u32], vec![1], vec![2], vec![0]];
        let mut original_positions = vec![];

        for action_ids in &action_seq {
            builder.record_tick(action_ids.clone());
            let actions: Vec<Action> = action_ids
                .iter()
                .map(|&id| Action::from_discrete(id, 0, false).unwrap_or(Action::Noop))
                .collect();
            world1.step(&actions);
            original_positions.push(world1.agents[0].position);
        }

        let replay = builder.build();

        // Replay and verify positions match
        let mut replay_iter = replay.replay().unwrap();
        for (i, result) in replay_iter.by_ref().enumerate() {
            let replay_pos = result.observations[0].position;
            let orig_pos = original_positions[i];
            assert_eq!(
                replay_pos,
                (orig_pos.x, orig_pos.y),
                "Position mismatch at tick {i}"
            );
        }
    }

    #[test]
    fn test_empty_replay() {
        let config = test_config();
        let replay = CompactReplay::builder(config, 0).build();
        assert_eq!(replay.actions.len(), 0);
        assert_eq!(replay.metadata.total_ticks, 0);

        let results: Vec<_> = replay.replay().unwrap().collect();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_replay_metadata_scenario_id() {
        let config = test_config();
        let replay = CompactReplay::builder(config, 42)
            .scenario_id("patrol_basic".to_string())
            .build();
        assert_eq!(replay.metadata.scenario_id.as_deref(), Some("patrol_basic"));
    }

    #[test]
    fn test_bincode_size_efficiency() {
        let config = test_config();
        let mut builder = CompactReplay::builder(config, 42);

        // Record 100 ticks of noop
        for _ in 0..100 {
            builder.record_tick(vec![0]);
        }

        let replay = builder.build();
        let bytes = replay.to_bytes().unwrap();

        // Bincode should be much smaller than JSON
        let json = replay.to_json().unwrap();
        assert!(
            bytes.len() < json.len(),
            "Bincode ({} bytes) should be smaller than JSON ({} bytes)",
            bytes.len(),
            json.len()
        );
    }

    #[test]
    fn test_invalid_bincode() {
        let result = CompactReplay::from_bytes(&[0, 1, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_json() {
        let result = CompactReplay::from_json("not json");
        assert!(result.is_err());
    }

    #[test]
    fn test_format_version_constant() {
        let config = test_config();
        let replay = CompactReplay::builder(config, 42).build();
        assert_eq!(replay.format_version, FORMAT_VERSION);
    }

    #[test]
    fn test_multi_agent_replay() {
        let mut config = test_config();
        config.agents.num_agents = 2;
        let mut builder = CompactReplay::builder(config, 42);
        builder.record_tick(vec![0, 1]);
        builder.record_tick(vec![1, 0]);

        let replay = builder
            .agent_names(vec!["Agent1".into(), "Agent2".into()])
            .build();

        assert_eq!(replay.actions.len(), 2);
        assert_eq!(replay.actions[0].len(), 2);
        assert_eq!(replay.metadata.agent_names.len(), 2);
    }

    #[test]
    fn test_replay_config_hash_determinism() {
        let c1 = test_config();
        let c2 = test_config();
        assert_eq!(hash_config(&c1), hash_config(&c2));
    }

    #[test]
    fn test_replay_iterator_world_access() {
        let config = test_config();
        let mut builder = CompactReplay::builder(config, 42);
        builder.record_tick(vec![0]);
        let replay = builder.build();

        let iter = replay.replay().unwrap();
        assert_eq!(iter.current_tick(), 0);
        assert!(!iter.world().terminated);
    }
}
