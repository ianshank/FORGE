//! BDI (Belief-Desire-Intention) episode collector.
//!
//! Maps FORGE actions to MangoMAS BDI intention classes and collects
//! episodes in a format suitable for training the BDI GRU+MLP model.

use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::adapters::action_adapter::ActionCategory;
use crate::batch_runner::Episode;
use crate::config::TransferConfig;
use crate::error::MangoMasResult;
use forge_types::action::Action;

/// Maps FORGE actions to MangoMAS BDI intention classes (0-7).
///
/// Default mapping:
/// - 0: Navigate (Move, Ascend, Descend, TakeOff, Land)
/// - 1: Gather (PickUp, Drop, DropPayload)
/// - 2: Plan (Craft)
/// - 3: Manipulate (Push, Use, Interact)
/// - 4: Cooperate (Communicate)
/// - 5: Evade (combat-related, contextual)
/// - 6: Track (Scan)
/// - 7: Idle (Noop, Hover)
pub struct BdiIntentionMapper {
    config: TransferConfig,
}

impl BdiIntentionMapper {
    /// Creates a new mapper.
    #[instrument(skip_all)]
    pub fn new(config: TransferConfig) -> Self {
        Self { config }
    }

    /// Returns the number of intention classes.
    pub fn num_intentions(&self) -> u8 {
        self.config.num_bdi_intentions
    }

    /// Map a FORGE action to a BDI intention class (0-7).
    pub fn map_action(&self, action: &Action) -> u8 {
        // Check for overrides first
        let discrete_id = action.to_discrete_full(16); // Standard vocab size
        for &(action_id, intention) in &self.config.bdi_mapping_overrides {
            if action_id == discrete_id {
                return intention.min(self.config.num_bdi_intentions - 1);
            }
        }

        // Fall back to semantic category mapping
        ActionCategory::from_action(action).as_intention_index()
    }

    /// Map a FORGE discrete action ID to a BDI intention class.
    pub fn map_action_id(&self, action_id: u32, comm_vocab: u16, drone_enabled: bool) -> u8 {
        match Action::from_discrete(action_id, comm_vocab, drone_enabled) {
            Some(action) => self.map_action(&action),
            None => 7, // Unknown → Idle
        }
    }
}

/// A single BDI training sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BdiSample {
    /// Flattened observation state vector.
    pub state: Vec<f32>,
    /// BDI intention class (0-7).
    pub intention: u8,
    /// Reward received after taking the action.
    pub reward: f32,
}

/// Dataset of BDI training samples extracted from FORGE episodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BdiTrainingData {
    /// Training samples.
    pub samples: Vec<BdiSample>,
    /// Number of intention classes.
    pub num_intentions: u8,
    /// Number of episodes the data was collected from.
    pub source_episodes: u32,
}

/// Collects FORGE episodes and converts them to BDI training data.
pub struct BdiEpisodeCollector {
    mapper: BdiIntentionMapper,
    comm_vocab: u16,
    drone_enabled: bool,
}

impl BdiEpisodeCollector {
    /// Creates a new collector.
    #[instrument(skip_all)]
    pub fn new(config: TransferConfig, comm_vocab: u16, drone_enabled: bool) -> Self {
        Self {
            mapper: BdiIntentionMapper::new(config),
            comm_vocab,
            drone_enabled,
        }
    }

    /// Converts a batch of episodes to BDI training data.
    ///
    /// Uses the provided observation adapter to flatten FORGE observations
    /// into state vectors compatible with the BDI GRU+MLP input.
    #[instrument(skip(self, episodes, adapt_obs))]
    pub fn collect_from_episodes(
        &self,
        episodes: &[Episode],
        adapt_obs: &dyn Fn(&forge_types::observation::Observation) -> Vec<f32>,
    ) -> MangoMasResult<BdiTrainingData> {
        let mut samples = Vec::new();

        for episode in episodes {
            for transition in &episode.transitions {
                let state = adapt_obs(&transition.observation);
                let intention = self.mapper.map_action_id(
                    transition.action_id,
                    self.comm_vocab,
                    self.drone_enabled,
                );
                samples.push(BdiSample {
                    state,
                    intention,
                    reward: transition.reward,
                });
            }
        }

        Ok(BdiTrainingData {
            source_episodes: episodes.len() as u32,
            num_intentions: self.mapper.num_intentions(),
            samples,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_types::grid::Direction;

    fn default_mapper() -> BdiIntentionMapper {
        BdiIntentionMapper::new(TransferConfig::default())
    }

    #[test]
    fn test_intention_mapping_coverage() {
        let mapper = default_mapper();
        // Test all action categories map to valid intentions
        let actions = vec![
            Action::Move(Direction::Up),
            Action::PickUp,
            Action::Craft(0),
            Action::Push(Direction::Left),
            Action::Communicate(0),
            Action::Scan(Direction::Up),
            Action::Noop,
            Action::Hover,
            Action::Ascend,
            Action::DropPayload(0),
        ];

        for action in &actions {
            let intention = mapper.map_action(action);
            assert!(
                intention < mapper.num_intentions(),
                "action {:?} mapped to intention {} >= {}",
                action,
                intention,
                mapper.num_intentions()
            );
        }
    }

    #[test]
    fn test_move_is_navigate() {
        let mapper = default_mapper();
        assert_eq!(mapper.map_action(&Action::Move(Direction::Up)), 0);
        assert_eq!(mapper.map_action(&Action::Ascend), 0);
    }

    #[test]
    fn test_pickup_is_gather() {
        let mapper = default_mapper();
        assert_eq!(mapper.map_action(&Action::PickUp), 1);
    }

    #[test]
    fn test_craft_is_plan() {
        let mapper = default_mapper();
        assert_eq!(mapper.map_action(&Action::Craft(0)), 2);
    }

    #[test]
    fn test_noop_is_idle() {
        let mapper = default_mapper();
        assert_eq!(mapper.map_action(&Action::Noop), 7);
    }

    #[test]
    fn test_custom_override() {
        let config = TransferConfig {
            bdi_mapping_overrides: vec![(0, 5)], // Noop → Evade
            ..TransferConfig::default()
        };
        let mapper = BdiIntentionMapper::new(config);
        // Action ID 0 is Noop, should now map to 5 (Evade) via override
        assert_eq!(mapper.map_action_id(0, 16, false), 5);
    }

    #[test]
    fn test_num_intentions() {
        let mapper = default_mapper();
        assert_eq!(mapper.num_intentions(), 8);
    }

    #[test]
    fn test_collect_from_empty_episodes() {
        let config = TransferConfig::default();
        let collector = BdiEpisodeCollector::new(config, 16, false);
        let data = collector
            .collect_from_episodes(&[], &|_obs| vec![0.0; 18])
            .unwrap();
        assert_eq!(data.samples.len(), 0);
    }

    #[test]
    fn test_bdi_training_data_serde() {
        let data = BdiTrainingData {
            samples: vec![BdiSample {
                state: vec![0.1, 0.2, 0.3],
                intention: 2,
                reward: 1.0,
            }],
            num_intentions: 8,
            source_episodes: 1,
        };
        let json = serde_json::to_string(&data).unwrap();
        let deser: BdiTrainingData = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.samples.len(), 1);
        assert_eq!(deser.samples[0].intention, 2);
    }
}
