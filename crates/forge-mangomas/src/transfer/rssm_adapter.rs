//! RSSM (Recurrent State-Space Model) sequence data extraction.
//!
//! Extracts (state, action, next_state, reward) transition tuples from
//! FORGE episodes for pre-training the RSSM GRU transition model,
//! reward/value heads, and prior distribution.

use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::batch_runner::Episode;
use crate::config::TransferConfig;
use crate::error::MangoMasResult;

/// A single transition for RSSM training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RssmTransition {
    /// State vector at time t.
    pub state: Vec<f32>,
    /// Action taken at time t (discrete ID).
    pub action_id: u32,
    /// Reward received at time t.
    pub reward: f32,
    /// Whether the episode ended at time t.
    pub done: bool,
}

/// A sequence of transitions for RSSM sequence training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionSequence {
    /// Ordered transitions in this sequence.
    pub transitions: Vec<RssmTransition>,
    /// Sequence length.
    pub length: usize,
}

/// Dataset of transition sequences for RSSM pre-training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceDataset {
    /// All sequences in the dataset.
    pub sequences: Vec<TransitionSequence>,
    /// State vector dimensionality.
    pub state_dim: usize,
    /// Number of discrete actions.
    pub action_dim: u32,
    /// Total transitions across all sequences.
    pub total_transitions: u64,
}

/// Builds RSSM-compatible sequence data from FORGE episodes.
pub struct RssmSequenceBuilder {
    config: TransferConfig,
}

impl RssmSequenceBuilder {
    /// Creates a new sequence builder.
    #[instrument(skip_all)]
    pub fn new(config: TransferConfig) -> Self {
        Self { config }
    }

    /// Extracts transition sequences from episodes.
    ///
    /// Each episode is split into fixed-length sequences of
    /// `config.rssm_sequence_length` transitions. Shorter episodes
    /// produce a single truncated sequence.
    pub fn build_sequences(
        &self,
        episodes: &[Episode],
        adapt_obs: &dyn Fn(&forge_types::observation::Observation) -> Vec<f32>,
    ) -> MangoMasResult<SequenceDataset> {
        let seq_len = self.config.rssm_sequence_length as usize;
        let mut sequences = Vec::new();
        let mut total_transitions = 0u64;
        let mut state_dim = 0;

        for episode in episodes {
            let transitions: Vec<RssmTransition> = episode
                .transitions
                .iter()
                .map(|t| {
                    let state = adapt_obs(&t.observation);
                    if state_dim == 0 {
                        state_dim = state.len();
                    }
                    RssmTransition {
                        state,
                        action_id: t.action_id,
                        reward: t.reward,
                        done: t.done,
                    }
                })
                .collect();

            // Split into fixed-length sequences
            for chunk in transitions.chunks(seq_len) {
                total_transitions += chunk.len() as u64;
                sequences.push(TransitionSequence {
                    length: chunk.len(),
                    transitions: chunk.to_vec(),
                });
            }
        }

        Ok(SequenceDataset {
            sequences,
            state_dim,
            action_dim: forge_types::action::Action::space_size(16, true),
            total_transitions,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch_runner::{Episode, Transition};
    use forge_types::observation::{InventoryObservation, Observation, TileObservation};

    fn make_episode(length: usize) -> Episode {
        let transitions = (0..length)
            .map(|i| Transition {
                observation: Observation {
                    grid_view: vec![TileObservation::default()],
                    view_width: 1,
                    view_height: 1,
                    inventory: InventoryObservation { slots: vec![] },
                    health: 1.0,
                    stamina: 1.0,
                    position: (i as u16, 0),
                    messages: vec![],
                    day_phase: 0,
                    task_progress: vec![],
                    altitude: 0,
                    battery: 1.0,
                    morphology: 0,
                    heading: 0,
                    crop_scan_results: vec![],
                    soil_readings: vec![],
                    disease_detections: 0,
                    report_ready: false,
                },
                action_id: 0,
                reward: 0.1,
                done: i == length - 1,
            })
            .collect();

        Episode {
            transitions,
            total_reward: length as f32 * 0.1,
            length: length as u64,
            seed: 42,
        }
    }

    fn identity_adapter(obs: &Observation) -> Vec<f32> {
        vec![obs.position.0 as f32, obs.position.1 as f32]
    }

    #[test]
    fn test_build_sequences() {
        let config = TransferConfig {
            rssm_sequence_length: 5,
            ..TransferConfig::default()
        };
        let builder = RssmSequenceBuilder::new(config);
        let episodes = vec![make_episode(12)];
        let dataset = builder
            .build_sequences(&episodes, &identity_adapter)
            .unwrap();

        // 12 transitions / 5 per seq = 3 sequences (5, 5, 2)
        assert_eq!(dataset.sequences.len(), 3);
        assert_eq!(dataset.sequences[0].length, 5);
        assert_eq!(dataset.sequences[2].length, 2);
        assert_eq!(dataset.total_transitions, 12);
        assert_eq!(dataset.state_dim, 2);
    }

    #[test]
    fn test_short_episode() {
        let config = TransferConfig {
            rssm_sequence_length: 100,
            ..TransferConfig::default()
        };
        let builder = RssmSequenceBuilder::new(config);
        let episodes = vec![make_episode(3)];
        let dataset = builder
            .build_sequences(&episodes, &identity_adapter)
            .unwrap();

        assert_eq!(dataset.sequences.len(), 1);
        assert_eq!(dataset.sequences[0].length, 3);
    }

    #[test]
    fn test_empty_episodes() {
        let builder = RssmSequenceBuilder::new(TransferConfig::default());
        let dataset = builder.build_sequences(&[], &identity_adapter).unwrap();
        assert!(dataset.sequences.is_empty());
        assert_eq!(dataset.total_transitions, 0);
    }

    #[test]
    fn test_sequence_dataset_serde() {
        let dataset = SequenceDataset {
            sequences: vec![],
            state_dim: 18,
            action_dim: 56,
            total_transitions: 0,
        };
        let json = serde_json::to_string(&dataset).unwrap();
        let deser: SequenceDataset = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.state_dim, 18);
    }
}
