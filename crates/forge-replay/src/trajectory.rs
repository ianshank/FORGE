//! Full trajectory storage for ML training pipelines.
//!
//! Unlike compact replays (which store only seed + actions), trajectories
//! store the full observation-action-reward tuple at each step. This is
//! larger but directly usable for training RL policies, RLHF, or
//! exporting to HuggingFace Datasets.

use forge_types::agent_interface::{AgentMetadata, AgentResponse};
use forge_types::observation::Observation;
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument};

/// A single step in a trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryStep {
    /// Simulation tick.
    pub tick: u64,
    /// Per-agent observations before the action.
    pub observations: Vec<Observation>,
    /// Per-agent action IDs taken.
    pub actions: Vec<u32>,
    /// Per-agent rewards received.
    pub rewards: Vec<f32>,
    /// Whether the episode terminated after this step.
    pub terminated: bool,
    /// Whether the episode was truncated after this step.
    pub truncated: bool,
    /// Per-agent reasoning traces (from LLM agents), if available.
    pub reasoning: Vec<Option<String>>,
    /// Per-agent confidence scores.
    pub confidences: Vec<f32>,
    /// Per-agent decision time in milliseconds.
    pub decision_times_ms: Vec<u64>,
}

/// A complete trajectory of an episode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trajectory {
    /// Sequence of steps in this trajectory.
    pub steps: Vec<TrajectoryStep>,
    /// Metadata about the trajectory.
    pub metadata: TrajectoryMetadata,
}

/// Metadata for a trajectory.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrajectoryMetadata {
    /// Names of agents involved.
    pub agent_names: Vec<String>,
    /// Metadata for each agent.
    pub agent_metadata: Vec<AgentMetadata>,
    /// Total steps in the trajectory.
    pub total_steps: u64,
    /// Final per-agent rewards.
    pub final_rewards: Vec<f32>,
    /// Seed used for the episode.
    pub seed: u64,
    /// ISO 8601 timestamp of recording.
    pub timestamp: String,
    /// Optional scenario identifier.
    pub scenario_id: Option<String>,
}

impl Trajectory {
    /// Creates a new empty trajectory.
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            metadata: TrajectoryMetadata::default(),
        }
    }

    /// Returns the number of steps.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Returns true if the trajectory is empty.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Returns total reward for the given agent index.
    pub fn total_reward(&self, agent_idx: usize) -> f32 {
        self.steps
            .iter()
            .map(|step| step.rewards.get(agent_idx).copied().unwrap_or(0.0))
            .sum()
    }
}

impl Default for Trajectory {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for constructing trajectories during an episode.
pub struct TrajectoryBuilder {
    steps: Vec<TrajectoryStep>,
    metadata: TrajectoryMetadata,
}

impl TrajectoryBuilder {
    /// Creates a new trajectory builder.
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            metadata: TrajectoryMetadata::default(),
        }
    }

    /// Records a step from observations, agent responses, and step result.
    #[instrument(skip_all)]
    pub fn record_step(
        &mut self,
        tick: u64,
        observations: Vec<Observation>,
        responses: &[AgentResponse],
        rewards: Vec<f32>,
        terminated: bool,
        truncated: bool,
    ) {
        let actions: Vec<u32> = responses.iter().map(|r| r.action_id).collect();
        let reasoning: Vec<Option<String>> =
            responses.iter().map(|r| r.reasoning.clone()).collect();
        let confidences: Vec<f32> = responses.iter().map(|r| r.confidence).collect();
        let decision_times_ms: Vec<u64> = responses.iter().map(|r| r.decision_time_ms).collect();

        self.steps.push(TrajectoryStep {
            tick,
            observations,
            actions,
            rewards,
            terminated,
            truncated,
            reasoning,
            confidences,
            decision_times_ms,
        });
    }

    /// Sets the seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.metadata.seed = seed;
        self
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

    /// Sets scenario ID.
    pub fn scenario_id(mut self, id: String) -> Self {
        self.metadata.scenario_id = Some(id);
        self
    }

    /// Builds the trajectory.
    #[instrument(skip_all)]
    pub fn build(mut self, final_rewards: Vec<f32>) -> Trajectory {
        self.metadata.total_steps = self.steps.len() as u64;
        self.metadata.final_rewards = final_rewards;
        self.metadata.timestamp = chrono::Utc::now().to_rfc3339();

        debug!(
            steps = self.metadata.total_steps,
            agents = self.metadata.agent_names.len(),
            "Built trajectory"
        );

        Trajectory {
            steps: self.steps,
            metadata: self.metadata,
        }
    }
}

impl Default for TrajectoryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_types::constants;
    use forge_types::observation::{InventoryObservation, TileObservation};

    fn make_test_observation() -> Observation {
        Observation {
            grid_view: vec![TileObservation::default()],
            view_width: 1,
            view_height: 1,
            inventory: InventoryObservation {
                slots: vec![(constants::OBS_EMPTY_SLOT_ITEM, 0)],
            },
            health: 1.0,
            stamina: 1.0,
            position: (5, 5),
            messages: vec![],
            day_phase: 1,
            task_progress: vec![],
            altitude: 0,
            battery: 1.0,
            morphology: 0,
            heading: 0,
            crop_scan_results: vec![],
            soil_readings: vec![],
            disease_detections: 0,
            report_ready: false,
        }
    }

    #[test]
    fn test_trajectory_builder() {
        let mut builder = TrajectoryBuilder::new();

        let obs = make_test_observation();
        let responses = vec![AgentResponse::from_action(1)];

        builder.record_step(0, vec![obs.clone()], &responses, vec![0.5], false, false);
        builder.record_step(1, vec![obs], &responses, vec![1.0], true, false);

        let traj = builder
            .seed(42)
            .agent_names(vec!["TestAgent".to_string()])
            .build(vec![1.5]);

        assert_eq!(traj.len(), 2);
        assert_eq!(traj.metadata.total_steps, 2);
        assert_eq!(traj.metadata.seed, 42);
        assert_eq!(traj.metadata.final_rewards, vec![1.5]);
        assert!(!traj.metadata.timestamp.is_empty());
    }

    #[test]
    fn test_trajectory_total_reward() {
        let mut builder = TrajectoryBuilder::new();
        let obs = make_test_observation();
        let responses = vec![AgentResponse::from_action(0)];

        builder.record_step(0, vec![obs.clone()], &responses, vec![0.5], false, false);
        builder.record_step(1, vec![obs.clone()], &responses, vec![1.0], false, false);
        builder.record_step(2, vec![obs], &responses, vec![0.25], true, false);

        let traj = builder.build(vec![1.75]);
        assert!((traj.total_reward(0) - 1.75).abs() < f32::EPSILON);
        assert_eq!(traj.total_reward(1), 0.0); // No agent at index 1
    }

    #[test]
    fn test_empty_trajectory() {
        let traj = Trajectory::new();
        assert!(traj.is_empty());
        assert_eq!(traj.len(), 0);
        assert_eq!(traj.total_reward(0), 0.0);
    }

    #[test]
    fn test_trajectory_step_preserves_reasoning() {
        let mut builder = TrajectoryBuilder::new();
        let obs = make_test_observation();
        let mut response = AgentResponse::from_action(3);
        response.reasoning = Some("I should gather wood".to_string());
        response.confidence = 0.85;
        response.decision_time_ms = 150;

        builder.record_step(0, vec![obs], &[response], vec![1.0], false, false);

        let traj = builder.build(vec![1.0]);
        let step = &traj.steps[0];
        assert_eq!(step.reasoning[0].as_deref(), Some("I should gather wood"));
        assert_eq!(step.confidences[0], 0.85);
        assert_eq!(step.decision_times_ms[0], 150);
    }

    #[test]
    fn test_trajectory_serde_roundtrip() {
        let mut builder = TrajectoryBuilder::new();
        let obs = make_test_observation();
        let responses = vec![AgentResponse::from_action(2)];
        builder.record_step(0, vec![obs], &responses, vec![0.5], false, false);

        let traj = builder.seed(99).build(vec![0.5]);

        let json = serde_json::to_string(&traj).unwrap();
        let deser: Trajectory = serde_json::from_str(&json).unwrap();

        assert_eq!(deser.len(), 1);
        assert_eq!(deser.metadata.seed, 99);
        assert_eq!(deser.steps[0].actions, vec![2]);
    }

    #[test]
    fn test_trajectory_metadata_scenario_id() {
        let builder = TrajectoryBuilder::new();
        let traj = builder
            .scenario_id("search_and_rescue".to_string())
            .build(vec![]);
        assert_eq!(
            traj.metadata.scenario_id.as_deref(),
            Some("search_and_rescue")
        );
    }

    mod prop {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn step_count_matches_transitions(n in 0usize..50) {
                let mut builder = TrajectoryBuilder::new();
                let obs = make_test_observation();
                let responses = vec![AgentResponse::from_action(0)];

                for tick in 0..n {
                    builder.record_step(
                        tick as u64,
                        vec![obs.clone()],
                        &responses,
                        vec![0.1],
                        false,
                        false,
                    );
                }

                let traj = builder.build(vec![n as f32 * 0.1]);
                prop_assert_eq!(traj.len(), n);
                prop_assert_eq!(traj.metadata.total_steps, n as u64);
            }

            #[test]
            fn total_reward_sums_correctly(rewards in proptest::collection::vec(0.0f32..10.0, 1..20)) {
                let mut builder = TrajectoryBuilder::new();
                let obs = make_test_observation();
                let responses = vec![AgentResponse::from_action(0)];
                let expected_sum: f32 = rewards.iter().sum();

                for (tick, &reward) in rewards.iter().enumerate() {
                    builder.record_step(
                        tick as u64,
                        vec![obs.clone()],
                        &responses,
                        vec![reward],
                        false,
                        false,
                    );
                }

                let traj = builder.build(vec![expected_sum]);
                prop_assert!((traj.total_reward(0) - expected_sum).abs() < 0.01);
            }
        }
    }

    #[test]
    fn test_multi_agent_trajectory() {
        let mut builder = TrajectoryBuilder::new();
        let obs = make_test_observation();
        let responses = vec![AgentResponse::from_action(1), AgentResponse::from_action(3)];

        builder.record_step(
            0,
            vec![obs.clone(), obs],
            &responses,
            vec![0.5, 0.3],
            false,
            false,
        );

        let traj = builder.build(vec![0.5, 0.3]);
        assert_eq!(traj.steps[0].actions, vec![1, 3]);
        assert_eq!(traj.steps[0].rewards, vec![0.5, 0.3]);
        assert!((traj.total_reward(0) - 0.5).abs() < f32::EPSILON);
        assert!((traj.total_reward(1) - 0.3).abs() < f32::EPSILON);
    }
}
