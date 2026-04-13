//! Adaptive latency-budgeted MCTS search.
//!
//! Wraps [`LatentMctsSearch`] from `forge-agent` with a [`LatencyEstimator`]
//! that dynamically adjusts the number of simulations per search to fit
//! within the configured latency budget.

use std::time::Instant;

use forge_agent::latent_mcts::model::LatentForwardModel;
use forge_agent::latent_mcts::search::{LatentMctsConfig, LatentMctsSearch};
use forge_types::config::EdgeConfig;
use forge_types::error::{EdgeError, ForgeError, ForgeResult};
use tracing::{debug, instrument, warn};

use crate::latency::LatencyEstimator;
use crate::metrics::AdaptiveSearchMetrics;

/// MCTS search with adaptive simulation budget based on latency constraint.
///
/// Wraps `LatentMctsSearch` from forge-agent. Before each search, estimates
/// how many simulations fit within the budget using [`LatencyEstimator`],
/// clamps to `[min_simulations, max_simulations]`, then delegates to the
/// inner search engine.
pub struct AdaptiveMctsSearch<M: LatentForwardModel + Clone> {
    /// The latent forward model used for inference.
    model: M,
    /// Latency estimator for adaptive budgeting.
    estimator: LatencyEstimator,
    /// Latency budget in milliseconds.
    budget_ms: u32,
    /// Minimum simulations to run regardless of budget.
    min_simulations: u32,
    /// Maximum simulations to run regardless of budget.
    max_simulations: u32,
    /// Base MCTS config template (num_simulations will be overridden per search).
    base_config: LatentMctsConfig,
}

impl<M: LatentForwardModel + Clone> AdaptiveMctsSearch<M> {
    /// Creates a new adaptive search from an [`EdgeConfig`], MCTS config, and model.
    ///
    /// The `edge_config` provides latency budget, simulation bounds, and EMA alpha.
    /// The `mcts_config` provides base MCTS parameters (c_puct, discount, etc.).
    pub fn new(model: M, edge_config: &EdgeConfig, mcts_config: LatentMctsConfig) -> Self {
        let estimator = LatencyEstimator::new(edge_config.latency_ema_alpha);
        Self {
            model,
            estimator,
            budget_ms: edge_config.mcts_latency_budget_ms,
            min_simulations: edge_config.mcts_min_simulations,
            max_simulations: edge_config.mcts_max_simulations,
            base_config: mcts_config,
        }
    }

    /// Runs MCTS with adaptive simulation budget.
    ///
    /// 1. Estimates how many simulations fit within the budget
    /// 2. Clamps to `[min_simulations, max_simulations]`
    /// 3. Runs `LatentMctsSearch` with the adjusted config
    /// 4. Updates the latency estimator with actual timing
    ///
    /// Returns the selected action ID and search metrics.
    #[instrument(skip_all, fields(budget_ms = %self.budget_ms))]
    pub fn search(&mut self, observation: &[f32]) -> ForgeResult<(u32, AdaptiveSearchMetrics)> {
        // Step 1: Estimate simulation count from latency data
        let estimated_sims = self.estimator.estimate_simulations(self.budget_ms);

        // Step 2: Clamp to configured bounds
        let clamped_sims = estimated_sims.clamp(self.min_simulations, self.max_simulations);

        debug!(
            estimated_sims,
            clamped_sims,
            budget_ms = self.budget_ms,
            "Adaptive simulation budget"
        );

        // Step 3: Build config with adjusted num_simulations
        let mut search_config = self.base_config.clone();
        search_config.base.num_simulations = clamped_sims;

        // Step 4: Create a LatentMctsSearch and run it
        // LatentMctsSearch::new is cheap (struct init, no allocation), so
        // rebuilding each call is acceptable to adjust num_simulations.
        let inner = LatentMctsSearch::new(self.model.clone(), search_config);

        let start = Instant::now();
        let result = inner.search(observation).map_err(|e| {
            ForgeError::Edge(EdgeError::Inference(format!(
                "latent MCTS search failed: {e}"
            )))
        })?;
        let elapsed_ms = start.elapsed().as_secs_f32() * 1000.0;

        // Step 5: Update estimator with actual timing
        self.estimator.update(clamped_sims, elapsed_ms);

        // Step 6: Build metrics
        let metrics = AdaptiveSearchMetrics {
            simulations_used: clamped_sims,
            actual_latency_ms: elapsed_ms,
            budget_ms: self.budget_ms,
            budget_utilization: if self.budget_ms > 0 {
                elapsed_ms / self.budget_ms as f32
            } else {
                0.0
            },
            estimated_per_sim_ms: self.estimator.estimated_per_sim_ms(),
            action_selected: result.action,
            root_value: result.root_value,
        };

        if elapsed_ms > self.budget_ms as f32 {
            warn!(
                budget_ms = self.budget_ms,
                actual_ms = elapsed_ms,
                "Search exceeded latency budget"
            );
        }

        Ok((result.action, metrics))
    }

    /// Returns a reference to the latency estimator for inspection.
    pub fn estimator(&self) -> &LatencyEstimator {
        &self.estimator
    }

    /// Resets the latency estimator state, clearing all timing history.
    pub fn reset_estimator(&mut self) {
        self.estimator.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_agent::latent_mcts::model::StubLatentModel;
    use forge_agent::mcts::tree::MctsConfig;

    fn make_edge_config() -> EdgeConfig {
        EdgeConfig {
            mcts_latency_budget_ms: 100,
            mcts_min_simulations: 4,
            mcts_max_simulations: 50,
            latency_ema_alpha: 0.3,
            ..EdgeConfig::default()
        }
    }

    fn make_mcts_config(action_space: u32) -> LatentMctsConfig {
        LatentMctsConfig {
            base: MctsConfig {
                num_simulations: 10,
                action_space,
                max_depth: 10,
                ..MctsConfig::default()
            },
            ..LatentMctsConfig::default()
        }
    }

    #[test]
    fn test_search_returns_valid_action() {
        let model = StubLatentModel::new(8, 64);
        let edge_cfg = make_edge_config();
        let mcts_cfg = make_mcts_config(8);
        let mut search = AdaptiveMctsSearch::new(model, &edge_cfg, mcts_cfg);

        let obs = vec![0.0; 100];
        let (action, metrics) = search.search(&obs).unwrap();
        assert!(action < 8);
        assert_eq!(metrics.action_selected, action);
        assert_eq!(metrics.budget_ms, 100);
    }

    #[test]
    fn test_clamps_to_min_when_estimator_says_zero() {
        let model = StubLatentModel::new(4, 32);
        let edge_cfg = make_edge_config();
        let mcts_cfg = make_mcts_config(4);
        let mut search = AdaptiveMctsSearch::new(model, &edge_cfg, mcts_cfg);

        // No prior timing data, so estimator returns 0 sims -> clamped to min (4)
        let obs = vec![0.0; 50];
        let (_, metrics) = search.search(&obs).unwrap();
        assert_eq!(metrics.simulations_used, 4);
    }

    #[test]
    fn test_clamps_to_max_when_budget_is_huge() {
        let model = StubLatentModel::new(4, 32);
        let mut edge_cfg = make_edge_config();
        edge_cfg.mcts_latency_budget_ms = 1_000_000; // huge budget
        let mcts_cfg = make_mcts_config(4);
        let mut search = AdaptiveMctsSearch::new(model, &edge_cfg, mcts_cfg);

        // Run once to seed the estimator with a low per-sim cost
        let obs = vec![0.0; 50];
        let _ = search.search(&obs).unwrap();

        // Second search: estimator now has data, huge budget -> clamped to max (50)
        let (_, metrics) = search.search(&obs).unwrap();
        assert_eq!(metrics.simulations_used, 50);
    }

    #[test]
    fn test_estimator_updates_after_search() {
        let model = StubLatentModel::new(4, 32);
        let edge_cfg = make_edge_config();
        let mcts_cfg = make_mcts_config(4);
        let mut search = AdaptiveMctsSearch::new(model, &edge_cfg, mcts_cfg);

        assert_eq!(search.estimator().total_samples(), 0);
        let obs = vec![0.0; 50];
        let _ = search.search(&obs).unwrap();
        assert_eq!(search.estimator().total_samples(), 1);
    }

    #[test]
    fn test_metrics_contain_correct_budget_utilization() {
        let model = StubLatentModel::new(4, 32);
        let edge_cfg = make_edge_config();
        let mcts_cfg = make_mcts_config(4);
        let mut search = AdaptiveMctsSearch::new(model, &edge_cfg, mcts_cfg);

        let obs = vec![0.0; 50];
        let (_, metrics) = search.search(&obs).unwrap();
        // budget_utilization = actual_latency / budget
        let expected = metrics.actual_latency_ms / metrics.budget_ms as f32;
        assert!((metrics.budget_utilization - expected).abs() < 1e-5);
    }

    #[test]
    fn test_reset_estimator() {
        let model = StubLatentModel::new(4, 32);
        let edge_cfg = make_edge_config();
        let mcts_cfg = make_mcts_config(4);
        let mut search = AdaptiveMctsSearch::new(model, &edge_cfg, mcts_cfg);

        let obs = vec![0.0; 50];
        let _ = search.search(&obs).unwrap();
        assert!(search.estimator().total_samples() > 0);

        search.reset_estimator();
        assert_eq!(search.estimator().total_samples(), 0);
        assert_eq!(search.estimator().estimated_per_sim_ms(), 0.0);
    }
}
