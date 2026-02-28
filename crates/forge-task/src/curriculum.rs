//! Curriculum controller for adaptive difficulty adjustment.
//!
//! Tracks agent success rates over a rolling window and adjusts
//! the distribution over task tiers to maintain a target success rate.

use forge_types::config::CurriculumConfig;
use rand::Rng;
use tracing::instrument;

/// Number of difficulty tiers.
const NUM_TIERS: usize = 6;

/// Curriculum controller that adapts task difficulty based on agent performance.
pub struct CurriculumController {
    config: CurriculumConfig,
    /// Rolling window of episode outcomes (true = success).
    outcomes: Vec<bool>,
    /// Probability distribution over tiers 1-6.
    tier_weights: [f32; NUM_TIERS],
    /// Total number of episodes recorded.
    total_episodes: u64,
}

impl CurriculumController {
    /// Creates a new curriculum controller with uniform initial distribution.
    #[instrument(skip_all)]
    pub fn new(config: CurriculumConfig) -> Self {
        let uniform = 1.0 / NUM_TIERS as f32;
        Self {
            config,
            outcomes: Vec::new(),
            tier_weights: [uniform; NUM_TIERS],
            total_episodes: 0,
        }
    }

    /// Records the outcome of an episode and adjusts difficulty if appropriate.
    #[instrument(skip_all)]
    pub fn record_outcome(&mut self, success: bool) {
        self.outcomes.push(success);
        self.total_episodes += 1;

        // Trim window to configured size
        let window = self.config.window_size as usize;
        if self.outcomes.len() > window {
            let excess = self.outcomes.len() - window;
            self.outcomes.drain(..excess);
        }

        // Only adjust after warmup
        if self.total_episodes >= self.config.warmup_episodes as u64 && self.config.enabled {
            self.adjust_difficulty();
        }
    }

    /// Returns the success rate over the rolling window.
    pub fn success_rate(&self) -> f32 {
        if self.outcomes.is_empty() {
            return 0.0;
        }
        let successes = self.outcomes.iter().filter(|&&o| o).count();
        successes as f32 / self.outcomes.len() as f32
    }

    /// Samples a tier (1-6) from the current weight distribution.
    ///
    /// The returned tier is clamped to `[1, max_tier]`.
    #[instrument(skip_all)]
    pub fn sample_tier<R: Rng>(&self, rng: &mut R, max_tier: u8) -> u8 {
        let max_tier = max_tier.clamp(1, NUM_TIERS as u8);
        let effective_weights: Vec<f32> = self.tier_weights[..max_tier as usize].to_vec();
        let total: f32 = effective_weights.iter().sum();

        if total <= 0.0 {
            return 1;
        }

        let mut r = rng.gen_range(0.0..total);
        for (i, &w) in effective_weights.iter().enumerate() {
            r -= w;
            if r <= 0.0 {
                return (i + 1) as u8;
            }
        }

        // Fallback: return the highest available tier
        max_tier
    }

    /// Returns the current tier weight distribution.
    pub fn tier_weights(&self) -> &[f32; NUM_TIERS] {
        &self.tier_weights
    }

    /// Returns the total number of episodes recorded.
    pub fn total_episodes(&self) -> u64 {
        self.total_episodes
    }

    /// Adjusts the tier weight distribution based on current success rate.
    ///
    /// If the success rate is above target: shift weight toward harder tiers.
    /// If the success rate is below target: shift weight toward easier tiers.
    #[instrument(skip_all)]
    fn adjust_difficulty(&mut self) {
        let rate = self.success_rate();
        let target = self.config.target_success_rate;
        let lr = self.config.adjustment_rate;

        if rate > target + 0.05 {
            // Too easy: shift toward harder tiers
            self.shift_weights_harder(lr);
        } else if rate < target - 0.05 {
            // Too hard: shift toward easier tiers
            self.shift_weights_easier(lr);
        }
        // Within deadband: no adjustment
    }

    /// Shifts weight from easier tiers to harder tiers.
    fn shift_weights_harder(&mut self, rate: f32) {
        // Transfer weight from lower tiers to upper tiers
        for i in 0..NUM_TIERS {
            let shift = self.tier_weights[i] * rate;
            if i + 1 < NUM_TIERS {
                self.tier_weights[i] -= shift;
                self.tier_weights[i + 1] += shift;
            }
        }
        self.normalize_weights();
    }

    /// Shifts weight from harder tiers to easier tiers.
    fn shift_weights_easier(&mut self, rate: f32) {
        // Transfer weight from upper tiers to lower tiers
        for i in (0..NUM_TIERS).rev() {
            let shift = self.tier_weights[i] * rate;
            if i > 0 {
                self.tier_weights[i] -= shift;
                self.tier_weights[i - 1] += shift;
            }
        }
        self.normalize_weights();
    }

    /// Normalizes the tier weights so they sum to 1.0.
    fn normalize_weights(&mut self) {
        let sum: f32 = self.tier_weights.iter().sum();
        if sum > 0.0 {
            for w in &mut self.tier_weights {
                *w /= sum;
            }
        } else {
            // Reset to uniform if all weights collapsed to zero
            let uniform = 1.0 / NUM_TIERS as f32;
            self.tier_weights = [uniform; NUM_TIERS];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_pcg::Pcg64;

    fn default_config() -> CurriculumConfig {
        CurriculumConfig {
            enabled: true,
            target_success_rate: 0.5,
            window_size: 100,
            warmup_episodes: 10,
            adjustment_rate: 0.1,
        }
    }

    fn make_rng(seed: u64) -> Pcg64 {
        Pcg64::seed_from_u64(seed)
    }

    #[test]
    fn test_initial_uniform_distribution() {
        let ctrl = CurriculumController::new(default_config());
        let weights = ctrl.tier_weights();
        let expected = 1.0 / 6.0;
        for &w in weights.iter() {
            assert!(
                (w - expected).abs() < 1e-6,
                "expected uniform weight {}, got {}",
                expected,
                w
            );
        }
    }

    #[test]
    fn test_success_rate_computation() {
        let mut ctrl = CurriculumController::new(CurriculumConfig {
            enabled: false,
            ..default_config()
        });
        // No outcomes yet
        assert_eq!(ctrl.success_rate(), 0.0);

        // Record 7 successes and 3 failures
        for _ in 0..7 {
            ctrl.record_outcome(true);
        }
        for _ in 0..3 {
            ctrl.record_outcome(false);
        }
        let rate = ctrl.success_rate();
        assert!((rate - 0.7).abs() < 1e-6, "expected 0.7, got {}", rate);
    }

    #[test]
    fn test_warmup_period() {
        let config = CurriculumConfig {
            warmup_episodes: 10,
            ..default_config()
        };
        let mut ctrl = CurriculumController::new(config);
        let initial_weights = *ctrl.tier_weights();

        // Record 9 successes (below warmup threshold of 10)
        for _ in 0..9 {
            ctrl.record_outcome(true);
        }

        // Weights should not have changed during warmup
        assert_eq!(
            ctrl.tier_weights(),
            &initial_weights,
            "weights should not change during warmup"
        );
    }

    #[test]
    fn test_difficulty_increases_on_high_success() {
        let config = CurriculumConfig {
            warmup_episodes: 5,
            window_size: 20,
            target_success_rate: 0.5,
            adjustment_rate: 0.1,
            enabled: true,
        };
        let mut ctrl = CurriculumController::new(config);
        let initial_lower_weight: f32 = ctrl.tier_weights()[0..3].iter().sum();

        // Record many successes to push success rate well above target
        for _ in 0..20 {
            ctrl.record_outcome(true);
        }

        let final_lower_weight: f32 = ctrl.tier_weights()[0..3].iter().sum();
        assert!(
            final_lower_weight < initial_lower_weight,
            "lower tier weights should decrease when success rate is high: initial={}, final={}",
            initial_lower_weight,
            final_lower_weight
        );
    }

    #[test]
    fn test_difficulty_decreases_on_low_success() {
        let config = CurriculumConfig {
            warmup_episodes: 5,
            window_size: 20,
            target_success_rate: 0.5,
            adjustment_rate: 0.1,
            enabled: true,
        };
        let mut ctrl = CurriculumController::new(config);
        let initial_lower_weight: f32 = ctrl.tier_weights()[0..3].iter().sum();

        // Record many failures to push success rate below target
        for _ in 0..20 {
            ctrl.record_outcome(false);
        }

        let final_lower_weight: f32 = ctrl.tier_weights()[0..3].iter().sum();
        assert!(
            final_lower_weight > initial_lower_weight,
            "lower tier weights should increase when success rate is low: initial={}, final={}",
            initial_lower_weight,
            final_lower_weight
        );
    }

    #[test]
    fn test_sample_tier_distribution() {
        let ctrl = CurriculumController::new(default_config());
        let mut rng = make_rng(42);
        let mut counts = [0u32; NUM_TIERS];
        let n = 6000;

        for _ in 0..n {
            let tier = ctrl.sample_tier(&mut rng, 6);
            assert!((1..=6).contains(&tier), "tier {} out of range", tier);
            counts[(tier - 1) as usize] += 1;
        }

        // With uniform weights, each tier should get roughly 1/6 of samples
        let expected = n as f32 / NUM_TIERS as f32;
        for (i, &count) in counts.iter().enumerate() {
            let ratio = count as f32 / expected;
            assert!(
                (0.7..=1.3).contains(&ratio),
                "tier {} has {} samples (expected ~{}), ratio={}",
                i + 1,
                count,
                expected,
                ratio
            );
        }
    }

    #[test]
    fn test_window_size_limits() {
        let config = CurriculumConfig {
            window_size: 5,
            warmup_episodes: 0,
            enabled: false,
            ..default_config()
        };
        let mut ctrl = CurriculumController::new(config);

        // Record 10 outcomes, only last 5 should be retained
        for _ in 0..5 {
            ctrl.record_outcome(true);
        }
        for _ in 0..5 {
            ctrl.record_outcome(false);
        }

        // The window should only contain the last 5 (all false)
        assert_eq!(ctrl.outcomes.len(), 5);
        assert!(
            (ctrl.success_rate() - 0.0).abs() < 1e-6,
            "only failures should be in window, got rate {}",
            ctrl.success_rate()
        );
    }

    #[test]
    fn test_total_episodes_counter() {
        let config = CurriculumConfig {
            enabled: false,
            ..default_config()
        };
        let mut ctrl = CurriculumController::new(config);
        assert_eq!(ctrl.total_episodes(), 0);

        ctrl.record_outcome(true);
        ctrl.record_outcome(false);
        ctrl.record_outcome(true);
        assert_eq!(ctrl.total_episodes(), 3);
    }

    #[test]
    fn test_sample_tier_respects_max_tier() {
        let ctrl = CurriculumController::new(default_config());
        let mut rng = make_rng(100);
        for _ in 0..100 {
            let tier = ctrl.sample_tier(&mut rng, 3);
            assert!((1..=3).contains(&tier), "tier {} exceeds max_tier=3", tier);
        }
    }

    #[test]
    fn test_curriculum_extreme_adjustment_rate() {
        // Test with adjustment_rate = 0.0 (no adjustment should happen)
        let config_zero = CurriculumConfig {
            warmup_episodes: 0,
            window_size: 10,
            adjustment_rate: 0.0,
            target_success_rate: 0.5,
            enabled: true,
        };
        let mut ctrl_zero = CurriculumController::new(config_zero);
        let initial_weights = *ctrl_zero.tier_weights();

        for _ in 0..20 {
            ctrl_zero.record_outcome(true);
        }
        // With rate 0.0, weights should not change
        for (i, (&initial, &final_w)) in initial_weights
            .iter()
            .zip(ctrl_zero.tier_weights().iter())
            .enumerate()
        {
            assert!(
                (initial - final_w).abs() < 1e-6,
                "tier {} weight changed with rate 0.0: {} -> {}",
                i + 1,
                initial,
                final_w
            );
        }

        // Test with adjustment_rate = 1.0 (aggressive adjustment)
        let config_one = CurriculumConfig {
            warmup_episodes: 0,
            window_size: 10,
            adjustment_rate: 1.0,
            target_success_rate: 0.5,
            enabled: true,
        };
        let mut ctrl_one = CurriculumController::new(config_one);

        for _ in 0..20 {
            ctrl_one.record_outcome(true);
        }
        // With rate 1.0, weights should still sum to 1.0 and be valid
        let sum: f32 = ctrl_one.tier_weights().iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "weights should sum to 1.0, got {}",
            sum
        );
        // Higher tiers should have gained weight
        let higher_weight: f32 = ctrl_one.tier_weights()[3..6].iter().sum();
        assert!(
            higher_weight > 0.5,
            "higher tiers should dominate with rate 1.0 and all successes, got {}",
            higher_weight
        );
    }

    #[test]
    fn test_weights_remain_normalized() {
        let config = CurriculumConfig {
            warmup_episodes: 0,
            window_size: 10,
            ..default_config()
        };
        let mut ctrl = CurriculumController::new(config);

        for i in 0..50 {
            ctrl.record_outcome(i % 3 == 0);
            let sum: f32 = ctrl.tier_weights().iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "weights should sum to 1.0, got {} at episode {}",
                sum,
                i
            );
        }
    }
}
