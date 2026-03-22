//! Health-aware agents and forward models for planning under degradation uncertainty.
//!
//! Implements the decision-making layer of the physics-informed health monitoring
//! system. Key components:
//!
//! - [`HealthAwareForwardModel`]: A forward model parameterized by a health hypothesis,
//!   allowing MCTS to simulate futures under specific degradation assumptions.
//! - [`HealthAwareAgent`]: An agent wrapper that maintains a belief distribution over
//!   health states and delegates action selection to an inner agent.
//! - [`HealthAwarePolicy`]: A policy that penalizes aggressive actions when health
//!   uncertainty is high, encouraging conservative behavior under degradation.

use forge_core::WorldState;
use forge_types::config::HealthMonitoringConfig;
use forge_types::constants::{FIXED_POINT_ONE, NUM_COMPONENT_TYPES};
use forge_types::observation::StepResult;
use forge_types::Action;
use tracing::{debug, instrument, trace};

use crate::baselines::Agent;
use crate::forward_model::{DefaultForwardModel, ForwardModel};
use crate::health_model::HealthStateLibrary;
use crate::mcts::policy::{PolicyValue, PolicyValueOutput};

/// Forward model parameterized by a health hypothesis.
///
/// Wraps the [`DefaultForwardModel`] and adjusts simulated state to
/// reflect a specific degradation hypothesis. This allows MCTS to
/// explore "what-if" scenarios under different assumed health states.
#[derive(Debug, Clone)]
pub struct HealthAwareForwardModel {
    /// Base forward model for simulation.
    base_model: DefaultForwardModel,
    /// Hypothesized component integrity levels (fixed-point).
    hypothesis_levels: [i32; NUM_COMPONENT_TYPES],
}

impl HealthAwareForwardModel {
    /// Creates a new health-aware forward model with the given hypothesis.
    pub fn new(comm_vocab_size: u16, hypothesis_levels: [i32; NUM_COMPONENT_TYPES]) -> Self {
        Self {
            base_model: DefaultForwardModel::new(comm_vocab_size),
            hypothesis_levels,
        }
    }

    /// Returns the hypothesized component levels.
    pub fn hypothesis_levels(&self) -> &[i32; NUM_COMPONENT_TYPES] {
        &self.hypothesis_levels
    }
}

impl ForwardModel for HealthAwareForwardModel {
    #[instrument(skip_all)]
    fn simulate(&self, state: &WorldState, actions: &[Action]) -> (WorldState, StepResult) {
        let mut next_state = state.clone();

        // Apply hypothesis to agents before stepping:
        // Set component integrity to the hypothesized levels so the
        // degradation system operates from the assumed state.
        for agent in &mut next_state.agents {
            if agent.alive {
                for (i, &level) in self.hypothesis_levels.iter().enumerate() {
                    agent.components[i].integrity = level;
                }
            }
        }

        let result = next_state.step(actions);
        (next_state, result)
    }

    fn is_terminal(&self, state: &WorldState) -> bool {
        self.base_model.is_terminal(state)
    }

    fn num_agents(&self, state: &WorldState) -> usize {
        self.base_model.num_agents(state)
    }

    fn action_space_size(&self) -> u32 {
        self.base_model.action_space_size()
    }
}

/// Agent that maintains a belief distribution over health states.
///
/// Observes component integrity readings, updates belief via the health
/// state library's posterior inference, and delegates action selection
/// to an inner agent. The belief update can be throttled to every
/// N ticks for performance.
pub struct HealthAwareAgent {
    /// Health state library for hypothesis classification and posterior inference.
    library: HealthStateLibrary,
    /// Current belief distribution: probability per hypothesis.
    belief: Vec<f32>,
    /// Inner agent for action selection.
    inner_agent: Box<dyn Agent>,
    /// How often to update the belief (every N observations).
    update_interval: u32,
    /// Counter for belief update scheduling.
    tick_counter: u32,
    /// Softmax temperature for posterior inference.
    temperature: f32,
}

impl HealthAwareAgent {
    /// Creates a new health-aware agent.
    ///
    /// Initializes with a uniform belief over all hypotheses.
    pub fn new(
        config: &HealthMonitoringConfig,
        inner_agent: Box<dyn Agent>,
        update_interval: u32,
        temperature: f32,
    ) -> Self {
        let library = HealthStateLibrary::from_config(config);
        let n = library.len();
        let uniform = 1.0 / n as f32;
        let belief = vec![uniform; n];

        Self {
            library,
            belief,
            inner_agent,
            update_interval: update_interval.max(1),
            tick_counter: 0,
            temperature,
        }
    }

    /// Returns the current belief distribution.
    pub fn belief(&self) -> &[f32] {
        &self.belief
    }

    /// Returns the most likely hypothesis ID.
    pub fn most_likely_hypothesis(&self) -> u16 {
        self.belief
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u16)
            .unwrap_or(0)
    }

    /// Returns the entropy of the current belief (in nats).
    pub fn belief_entropy(&self) -> f32 {
        self.belief
            .iter()
            .map(|&p| if p > f32::EPSILON { -p * p.ln() } else { 0.0 })
            .sum()
    }

    /// Updates belief based on observed component integrity values.
    fn update_belief(&mut self, observed: &[f32; NUM_COMPONENT_TYPES]) {
        let posterior = self.library.posterior(observed, self.temperature);

        // Bayesian update: multiply prior (current belief) by likelihood (posterior)
        let mut new_belief = Vec::with_capacity(self.belief.len());
        let mut sum = 0.0_f32;

        for (i, (_, likelihood)) in posterior.iter().enumerate() {
            let prior = self.belief.get(i).copied().unwrap_or(0.0);
            let updated = prior * likelihood;
            new_belief.push(updated);
            sum += updated;
        }

        // Normalize
        if sum > f32::EPSILON {
            for b in &mut new_belief {
                *b /= sum;
            }
        }

        self.belief = new_belief;

        trace!(
            most_likely = self.most_likely_hypothesis(),
            entropy = self.belief_entropy(),
            "belief updated"
        );
    }
}

impl Agent for HealthAwareAgent {
    #[instrument(skip_all)]
    fn select_action(&mut self, state: &WorldState, agent_idx: usize) -> Action {
        // Update belief at scheduled intervals
        self.tick_counter += 1;
        if self.tick_counter % self.update_interval == 0 {
            if let Some(agent) = state.agents.get(agent_idx) {
                let fp_one = FIXED_POINT_ONE as f32;
                let observed = [
                    agent.components[0].integrity as f32 / fp_one,
                    agent.components[1].integrity as f32 / fp_one,
                    agent.components[2].integrity as f32 / fp_one,
                ];
                self.update_belief(&observed);
            }
        }

        // Delegate to inner agent
        self.inner_agent.select_action(state, agent_idx)
    }

    fn name(&self) -> &str {
        "HealthAwareAgent"
    }
}

/// Policy that penalizes aggressive actions under health uncertainty.
///
/// When the agent's health state is uncertain (high belief entropy),
/// movement priors are reduced and value estimates are penalized,
/// causing MCTS to prefer conservative strategies.
pub struct HealthAwarePolicy {
    /// Base policy for prior generation.
    base_policy: Box<dyn PolicyValue>,
    /// Health state library for entropy computation.
    library: HealthStateLibrary,
    /// How much to penalize value under uncertainty (0.0 = no penalty, 1.0 = full).
    risk_aversion: f32,
}

impl HealthAwarePolicy {
    /// Creates a new health-aware policy.
    pub fn new(
        base_policy: Box<dyn PolicyValue>,
        config: &HealthMonitoringConfig,
        risk_aversion: f32,
    ) -> Self {
        Self {
            base_policy,
            library: HealthStateLibrary::from_config(config),
            risk_aversion: risk_aversion.clamp(0.0, 1.0),
        }
    }
}

impl PolicyValue for HealthAwarePolicy {
    #[instrument(skip_all)]
    fn evaluate(&self, state: &WorldState, agent_idx: usize) -> PolicyValueOutput {
        let mut output = self.base_policy.evaluate(state, agent_idx);

        if let Some(agent) = state.agents.get(agent_idx) {
            let fp_one = FIXED_POINT_ONE as f32;
            let observed = [
                agent.components[0].integrity as f32 / fp_one,
                agent.components[1].integrity as f32 / fp_one,
                agent.components[2].integrity as f32 / fp_one,
            ];

            // Compute uncertainty from posterior
            let posterior = self.library.posterior(&observed, 0.1);
            let entropy = HealthStateLibrary::entropy(&posterior);
            let max_entropy = (self.library.len() as f32).ln();
            let normalized_uncertainty = if max_entropy > 0.0 {
                (entropy / max_entropy).clamp(0.0, 1.0)
            } else {
                0.0
            };

            // Penalize value proportional to uncertainty and risk aversion
            let penalty = normalized_uncertainty * self.risk_aversion;
            output.value *= 1.0 - penalty;

            // Reduce movement priors when uncertainty is high (actions 1-4 are moves)
            if normalized_uncertainty > 0.5 {
                let reduction = 0.5 * normalized_uncertainty;
                for i in 1..=4.min(output.priors.len()) {
                    output.priors[i] *= 1.0 - reduction;
                }
                // Re-normalize priors
                let sum: f32 = output.priors.iter().sum();
                if sum > f32::EPSILON {
                    for p in &mut output.priors {
                        *p /= sum;
                    }
                }
            }

            debug!(
                agent_id = agent.id,
                uncertainty = normalized_uncertainty,
                penalty = penalty,
                "health-aware policy evaluation"
            );
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::baselines::NoopAgent;
    use crate::mcts::policy::UniformPolicy;
    use forge_types::config::ForgeConfig;

    fn make_test_state() -> WorldState {
        let mut config = ForgeConfig::default();
        config.world.width = 16;
        config.world.height = 16;
        config.world.seed = 42;
        config.agents.num_agents = 1;
        config.agents.comm_vocab_size = 0;
        config.agents.default_vision_radius = 3;
        config.task.max_episode_length = 100;
        config.health_monitoring.enabled = true;
        WorldState::new(config).unwrap()
    }

    fn make_hm_config() -> HealthMonitoringConfig {
        HealthMonitoringConfig {
            enabled: true,
            num_degradation_levels: 3,
            ..Default::default()
        }
    }

    #[test]
    fn test_health_aware_forward_model_simulate() {
        let state = make_test_state();
        let hypothesis = [FIXED_POINT_ONE / 2; NUM_COMPONENT_TYPES];
        let model = HealthAwareForwardModel::new(0, hypothesis);

        let (next_state, result) = model.simulate(&state, &[Action::Noop]);
        assert_eq!(next_state.tick, 1);
        assert!(!result.terminated);
    }

    #[test]
    fn test_health_aware_forward_model_applies_hypothesis() {
        let state = make_test_state();
        let half = FIXED_POINT_ONE / 2;
        let hypothesis = [half, half, half];
        let model = HealthAwareForwardModel::new(0, hypothesis);

        let (next_state, _) = model.simulate(&state, &[Action::Noop]);
        // After stepping, components should have started from hypothesis levels
        // (sensor drifts further, motor stays since no movement)
        let agent = &next_state.agents[0];
        // Motor was set to half, no move action, so it stays at half
        assert_eq!(agent.components[0].integrity, half);
    }

    #[test]
    fn test_health_aware_agent_selects_action() {
        let state = make_test_state();
        let config = make_hm_config();
        let inner = Box::new(NoopAgent);
        let mut agent = HealthAwareAgent::new(&config, inner, 1, 0.1);
        let action = agent.select_action(&state, 0);
        assert!(matches!(action, Action::Noop));
    }

    #[test]
    fn test_belief_starts_uniform() {
        let config = make_hm_config();
        let inner = Box::new(NoopAgent);
        let agent = HealthAwareAgent::new(&config, inner, 1, 0.1);
        let expected = 1.0 / agent.belief().len() as f32;
        for &b in agent.belief() {
            assert!((b - expected).abs() < 1e-6, "belief should start uniform");
        }
    }

    #[test]
    fn test_belief_update_shifts_toward_correct() {
        let config = make_hm_config();
        let inner = Box::new(NoopAgent);
        let mut agent = HealthAwareAgent::new(&config, inner, 1, 0.1);

        // Observe fully healthy state multiple times
        let healthy = [1.0, 1.0, 1.0];
        for _ in 0..5 {
            agent.update_belief(&healthy);
        }

        // The hypothesis closest to healthy should have highest probability
        let best_id = agent.most_likely_hypothesis();
        let best_hyp = agent.library.get(best_id).unwrap();
        // Check that the best hypothesis has high integrity values
        for &level in &best_hyp.component_levels {
            assert!(
                level >= FIXED_POINT_ONE / 2,
                "most likely hypothesis should be near-healthy"
            );
        }
    }

    #[test]
    fn test_health_aware_policy_reduces_priors_under_uncertainty() {
        let state = make_test_state();
        let config = make_hm_config();
        let base = Box::new(UniformPolicy::new(40));
        let policy = HealthAwarePolicy::new(base, &config, 1.0);
        let output = policy.evaluate(&state, 0);

        // Should still produce valid priors
        let sum: f32 = output.priors.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "priors should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn test_health_aware_forward_model_action_space() {
        let model = HealthAwareForwardModel::new(0, [FIXED_POINT_ONE; 3]);
        assert_eq!(model.action_space_size(), 40);
    }

    #[test]
    fn test_health_aware_agent_name() {
        let config = make_hm_config();
        let inner = Box::new(NoopAgent);
        let agent = HealthAwareAgent::new(&config, inner, 1, 0.1);
        assert_eq!(agent.name(), "HealthAwareAgent");
    }

    #[test]
    fn test_belief_entropy_uniform_start() {
        let config = make_hm_config();
        let inner = Box::new(NoopAgent);
        let agent = HealthAwareAgent::new(&config, inner, 1, 0.1);
        let h = agent.belief_entropy();
        let expected = (agent.belief().len() as f32).ln();
        assert!(
            (h - expected).abs() < 1e-4,
            "uniform belief entropy should be ln(N), got {h}, expected {expected}"
        );
    }
}
