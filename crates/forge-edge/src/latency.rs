//! Exponential moving average latency estimator.
//!
//! Tracks per-simulation cost to predict how many MCTS simulations
//! fit within a latency budget. Uses the `latency_ema_alpha` from
//! [`EdgeConfig`](forge_types::config::EdgeConfig).

use serde::{Deserialize, Serialize};
use tracing::{debug, instrument};

/// Exponential moving average latency estimator.
///
/// Tracks per-simulation cost to predict how many MCTS simulations
/// fit within a latency budget. The smoothing factor `alpha` controls
/// how quickly the estimate adapts: higher alpha means more weight
/// on recent samples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyEstimator {
    /// EMA smoothing factor, clamped to (0.0, 1.0].
    alpha: f32,
    /// Current estimate of milliseconds per simulation.
    estimated_per_sim_ms: f32,
    /// Total number of update samples received.
    total_samples: u64,
}

impl LatencyEstimator {
    /// Creates a new estimator with the given EMA smoothing factor.
    ///
    /// Alpha is clamped to the range (0.0, 1.0]. An alpha of 0.0 is
    /// bumped to a small epsilon to avoid a frozen estimate.
    pub fn new(alpha: f32) -> Self {
        let clamped = alpha.clamp(f32::EPSILON, 1.0);
        Self {
            alpha: clamped,
            estimated_per_sim_ms: 0.0,
            total_samples: 0,
        }
    }

    /// Records a timing sample: `simulations` sims took `total_ms` milliseconds.
    ///
    /// Updates the EMA estimate of per-simulation cost. The first sample
    /// sets the estimate directly; subsequent samples blend via EMA.
    #[instrument(skip_all, fields(simulations, total_ms))]
    pub fn update(&mut self, simulations: u32, total_ms: f32) {
        if simulations == 0 {
            return;
        }
        let per_sim = total_ms / simulations as f32;
        if self.total_samples == 0 {
            self.estimated_per_sim_ms = per_sim;
        } else {
            self.estimated_per_sim_ms =
                self.alpha * per_sim + (1.0 - self.alpha) * self.estimated_per_sim_ms;
        }
        self.total_samples += 1;
        debug!(
            per_sim_ms = self.estimated_per_sim_ms,
            samples = self.total_samples,
            "Latency estimate updated"
        );
    }

    /// Returns the current estimate of milliseconds per simulation.
    pub fn estimated_per_sim_ms(&self) -> f32 {
        self.estimated_per_sim_ms
    }

    /// Estimates how many simulations fit in the given budget.
    ///
    /// Returns 0 if the estimate is zero (no data yet) or negative.
    pub fn estimate_simulations(&self, budget_ms: u32) -> u32 {
        if self.estimated_per_sim_ms <= 0.0 {
            return 0;
        }
        (budget_ms as f32 / self.estimated_per_sim_ms) as u32
    }

    /// Returns the total number of update samples received.
    pub fn total_samples(&self) -> u64 {
        self.total_samples
    }

    /// Resets the estimator state, clearing all history.
    pub fn reset(&mut self) {
        self.estimated_per_sim_ms = 0.0;
        self.total_samples = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_estimate_is_zero() {
        let est = LatencyEstimator::new(0.3);
        assert_eq!(est.estimated_per_sim_ms(), 0.0);
        assert_eq!(est.total_samples(), 0);
    }

    #[test]
    fn test_first_update_sets_estimate() {
        let mut est = LatencyEstimator::new(0.3);
        est.update(10, 50.0); // 5.0 ms per sim
        assert!((est.estimated_per_sim_ms() - 5.0).abs() < f32::EPSILON);
        assert_eq!(est.total_samples(), 1);
    }

    #[test]
    fn test_ema_blends_subsequent_updates() {
        let mut est = LatencyEstimator::new(0.5);
        est.update(10, 100.0); // 10.0 ms/sim
        est.update(10, 60.0); // 6.0 ms/sim; EMA = 0.5*6 + 0.5*10 = 8.0
        assert!((est.estimated_per_sim_ms() - 8.0).abs() < 1e-5);
        assert_eq!(est.total_samples(), 2);
    }

    #[test]
    fn test_budget_calculation() {
        let mut est = LatencyEstimator::new(0.3);
        est.update(10, 50.0); // 5.0 ms/sim
                              // budget=50ms / 5.0ms = 10 sims
        assert_eq!(est.estimate_simulations(50), 10);
    }

    #[test]
    fn test_budget_returns_zero_when_no_data() {
        let est = LatencyEstimator::new(0.3);
        assert_eq!(est.estimate_simulations(100), 0);
    }

    #[test]
    fn test_alpha_clamped_low() {
        let est = LatencyEstimator::new(0.0);
        // Alpha should be clamped to epsilon, not zero
        assert!(est.alpha > 0.0);
    }

    #[test]
    fn test_alpha_clamped_high() {
        let est = LatencyEstimator::new(2.0);
        assert!((est.alpha - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_reset_clears_state() {
        let mut est = LatencyEstimator::new(0.3);
        est.update(10, 50.0);
        est.update(10, 60.0);
        assert!(est.total_samples() > 0);
        est.reset();
        assert_eq!(est.estimated_per_sim_ms(), 0.0);
        assert_eq!(est.total_samples(), 0);
    }

    #[test]
    fn test_zero_simulations_ignored() {
        let mut est = LatencyEstimator::new(0.3);
        est.update(0, 100.0);
        assert_eq!(est.estimated_per_sim_ms(), 0.0);
        assert_eq!(est.total_samples(), 0);
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn updates_produce_non_negative_estimates(
                alpha in 0.01_f32..=1.0,
                samples in proptest::collection::vec((1u32..100, 0.1_f32..1000.0), 1..50),
            ) {
                let mut est = LatencyEstimator::new(alpha);
                for (sims, total_ms) in samples {
                    est.update(sims, total_ms);
                    prop_assert!(est.estimated_per_sim_ms() >= 0.0);
                }
            }

            #[test]
            fn estimate_simulations_non_negative(
                alpha in 0.01_f32..=1.0,
                sims in 1u32..100,
                total_ms in 0.1_f32..1000.0,
                budget in 1u32..10000,
            ) {
                let mut est = LatencyEstimator::new(alpha);
                est.update(sims, total_ms);
                let _ = est.estimate_simulations(budget);
            }
        }
    }
}
