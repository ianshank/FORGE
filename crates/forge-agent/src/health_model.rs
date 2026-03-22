//! Health state library for physics-informed health monitoring.
//!
//! Implements the "digital twin" concept: a library of discrete health
//! hypotheses generated from the physics model. Each hypothesis represents
//! a specific combination of component degradation levels and predicts
//! what sensor readings should look like under that state.
//!
//! The library enables:
//! - **Classification**: mapping noisy sensor readings to the best-matching hypothesis.
//! - **Posterior inference**: computing a probability distribution over all hypotheses.
//!
//! This follows the pattern from physics-informed ML where physics models
//! generate training data that real-world experience cannot provide.

use forge_types::config::HealthMonitoringConfig;
use forge_types::constants::{FIXED_POINT_ONE, NUM_COMPONENT_TYPES};
use serde::{Deserialize, Serialize};
use tracing::{instrument, trace};

/// A discrete health hypothesis — one possible internal degradation state.
///
/// Each hypothesis corresponds to a specific combination of component
/// integrity levels and includes the expected observation signature
/// that a physics model would predict for this state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthHypothesis {
    /// Unique identifier for this hypothesis.
    pub id: u16,
    /// Per-component integrity levels (fixed-point).
    /// Indexed by [`ComponentType`](forge_types::entity::ComponentType) discriminant.
    pub component_levels: [i32; NUM_COMPONENT_TYPES],
    /// Expected observation signature under this hypothesis.
    /// `[motor_efficiency_factor, sensor_noise_factor, damage_amplification_factor]`
    /// Each normalized to 0.0-1.0.
    pub expected_signature: [f32; NUM_COMPONENT_TYPES],
}

/// Library of health state hypotheses generated from the physics model.
///
/// Creates `num_levels^NUM_COMPONENT_TYPES` hypotheses covering all
/// combinations of degradation levels across components.
#[derive(Debug, Clone)]
pub struct HealthStateLibrary {
    /// All hypotheses in the library.
    hypotheses: Vec<HealthHypothesis>,
    /// Number of degradation levels per component.
    num_levels: u8,
}

impl HealthStateLibrary {
    /// Generates the full hypothesis library from configuration.
    ///
    /// Creates `num_degradation_levels^3` hypotheses, one per combination
    /// of motor, sensor, and structure degradation levels.
    #[instrument(skip_all)]
    pub fn from_config(config: &HealthMonitoringConfig) -> Self {
        let n = config.num_degradation_levels.max(2) as usize;
        let total = n * n * n;
        let mut hypotheses = Vec::with_capacity(total);

        let fp_one_f32 = FIXED_POINT_ONE as f32;

        for motor_idx in 0..n {
            for sensor_idx in 0..n {
                for structure_idx in 0..n {
                    let id = hypotheses.len() as u16;

                    // Map level index to integrity: level 0 = fully healthy, level (n-1) = floor/zero
                    let motor_integrity =
                        Self::level_to_integrity(motor_idx, n, config.motor_efficiency_floor);
                    let sensor_integrity = Self::level_to_integrity(sensor_idx, n, 0);
                    let structure_integrity = Self::level_to_integrity(structure_idx, n, 0);

                    // Expected signature: normalized integrity values
                    let expected_signature = [
                        motor_integrity as f32 / fp_one_f32,
                        sensor_integrity as f32 / fp_one_f32,
                        structure_integrity as f32 / fp_one_f32,
                    ];

                    hypotheses.push(HealthHypothesis {
                        id,
                        component_levels: [motor_integrity, sensor_integrity, structure_integrity],
                        expected_signature,
                    });
                }
            }
        }

        trace!(
            num_hypotheses = hypotheses.len(),
            num_levels = n,
            "health state library generated"
        );

        Self {
            hypotheses,
            num_levels: config.num_degradation_levels.max(2),
        }
    }

    /// Maps a level index to a fixed-point integrity value.
    ///
    /// Level 0 = `FIXED_POINT_ONE` (fully healthy).
    /// Level `n-1` = `floor` (maximally degraded, subject to floor).
    fn level_to_integrity(level: usize, num_levels: usize, floor: i32) -> i32 {
        if num_levels <= 1 {
            return FIXED_POINT_ONE;
        }
        let range = FIXED_POINT_ONE - floor;
        let step = range / (num_levels as i32 - 1);
        (FIXED_POINT_ONE - step * level as i32).max(floor)
    }

    /// Returns the hypothesis that best matches observed sensor readings.
    ///
    /// Uses sum-of-squared-differences as the distance metric.
    /// O(N) where N = number of hypotheses. No heap allocation.
    #[instrument(skip_all)]
    pub fn classify(&self, observed: &[f32; NUM_COMPONENT_TYPES]) -> &HealthHypothesis {
        let mut best_idx = 0;
        let mut best_dist = f32::MAX;

        for (i, hyp) in self.hypotheses.iter().enumerate() {
            let dist = squared_distance(observed, &hyp.expected_signature);
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }

        &self.hypotheses[best_idx]
    }

    /// Returns a probability distribution over all hypotheses given observations.
    ///
    /// Uses softmax over negative squared distances with configurable temperature.
    /// Lower temperature = sharper distribution (more confident).
    /// Higher temperature = flatter distribution (more uncertain).
    ///
    /// Returns pairs of (hypothesis_id, probability), sorted by hypothesis ID.
    #[instrument(skip_all)]
    pub fn posterior(
        &self,
        observed: &[f32; NUM_COMPONENT_TYPES],
        temperature: f32,
    ) -> Vec<(u16, f32)> {
        let temp = temperature.max(f32::EPSILON);

        // Compute negative squared distances
        let log_weights: Vec<f32> = self
            .hypotheses
            .iter()
            .map(|hyp| -squared_distance(observed, &hyp.expected_signature) / temp)
            .collect();

        // Softmax: shift by max for numerical stability
        let max_log = log_weights
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let exp_weights: Vec<f32> = log_weights.iter().map(|w| (w - max_log).exp()).collect();
        let sum: f32 = exp_weights.iter().sum();

        self.hypotheses
            .iter()
            .zip(exp_weights.iter())
            .map(|(hyp, &w)| (hyp.id, w / sum))
            .collect()
    }

    /// Computes the entropy of a posterior distribution (in nats).
    ///
    /// Higher entropy = more uncertainty about the true health state.
    pub fn entropy(posterior: &[(u16, f32)]) -> f32 {
        posterior
            .iter()
            .map(|(_, p)| if *p > f32::EPSILON { -p * p.ln() } else { 0.0 })
            .sum()
    }

    /// Returns the number of hypotheses in the library.
    pub fn len(&self) -> usize {
        self.hypotheses.len()
    }

    /// Returns whether the library is empty.
    pub fn is_empty(&self) -> bool {
        self.hypotheses.is_empty()
    }

    /// Returns the number of degradation levels per component.
    pub fn num_levels(&self) -> u8 {
        self.num_levels
    }

    /// Returns a reference to all hypotheses.
    pub fn hypotheses(&self) -> &[HealthHypothesis] {
        &self.hypotheses
    }

    /// Returns a specific hypothesis by ID.
    pub fn get(&self, id: u16) -> Option<&HealthHypothesis> {
        self.hypotheses.get(id as usize)
    }
}

/// Computes the sum of squared differences between two observation vectors.
#[inline]
fn squared_distance(a: &[f32; NUM_COMPONENT_TYPES], b: &[f32; NUM_COMPONENT_TYPES]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_config() -> HealthMonitoringConfig {
        HealthMonitoringConfig {
            enabled: true,
            num_degradation_levels: 5,
            ..Default::default()
        }
    }

    #[test]
    fn test_library_size() {
        let config = make_test_config();
        let lib = HealthStateLibrary::from_config(&config);
        assert_eq!(lib.len(), 5 * 5 * 5);
        assert_eq!(lib.len(), 125);
    }

    #[test]
    fn test_library_size_custom_levels() {
        let mut config = make_test_config();
        config.num_degradation_levels = 3;
        let lib = HealthStateLibrary::from_config(&config);
        assert_eq!(lib.len(), 27); // 3^3
    }

    #[test]
    fn test_library_minimum_levels() {
        let mut config = make_test_config();
        config.num_degradation_levels = 1; // Should clamp to 2
        let lib = HealthStateLibrary::from_config(&config);
        assert_eq!(lib.len(), 8); // 2^3
    }

    #[test]
    fn test_classify_exact_match_healthy() {
        let config = make_test_config();
        let lib = HealthStateLibrary::from_config(&config);
        // Fully healthy observation should match hypothesis 0
        let observed = [1.0, 1.0, 1.0];
        let best = lib.classify(&observed);
        assert_eq!(best.id, 0);
        for &sig in &best.expected_signature {
            assert!((sig - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_classify_fully_degraded() {
        let config = make_test_config();
        let lib = HealthStateLibrary::from_config(&config);
        // Fully degraded: motor at floor (0.25), sensor at 0, structure at 0
        let floor = config.motor_efficiency_floor as f32 / FIXED_POINT_ONE as f32;
        let observed = [floor, 0.0, 0.0];
        let best = lib.classify(&observed);
        // Should be the last hypothesis (max degradation on all components)
        assert_eq!(best.id, (lib.len() - 1) as u16);
    }

    #[test]
    fn test_classify_noisy_input() {
        let config = make_test_config();
        let lib = HealthStateLibrary::from_config(&config);
        // Slightly noisy version of fully healthy
        let observed = [0.98, 0.99, 1.01];
        let best = lib.classify(&observed);
        // Should still match the healthy hypothesis
        assert_eq!(best.id, 0);
    }

    #[test]
    fn test_posterior_sums_to_one() {
        let config = make_test_config();
        let lib = HealthStateLibrary::from_config(&config);
        let observed = [0.8, 0.6, 0.9];
        let post = lib.posterior(&observed, 0.1);
        let sum: f32 = post.iter().map(|(_, p)| p).sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "posterior should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn test_posterior_high_confidence() {
        let config = make_test_config();
        let lib = HealthStateLibrary::from_config(&config);
        let observed = [1.0, 1.0, 1.0];
        let post = lib.posterior(&observed, 0.01); // Low temp = high confidence
                                                   // The best match should have high probability
        let max_p = post.iter().map(|(_, p)| *p).fold(0.0_f32, f32::max);
        assert!(
            max_p > 0.5,
            "best hypothesis should have >50% probability, got {max_p}"
        );
    }

    #[test]
    fn test_posterior_uniform_under_ambiguity() {
        let config = make_test_config();
        let lib = HealthStateLibrary::from_config(&config);
        let observed = [0.5, 0.5, 0.5];
        let post = lib.posterior(&observed, 100.0); // Very high temp = flat
        let max_p = post.iter().map(|(_, p)| *p).fold(0.0_f32, f32::max);
        let min_p = post.iter().map(|(_, p)| *p).fold(f32::MAX, f32::min);
        let ratio = max_p / min_p.max(f32::EPSILON);
        assert!(
            ratio < 5.0,
            "high temperature should produce near-uniform, ratio={ratio}"
        );
    }

    #[test]
    fn test_library_from_config_determinism() {
        let config = make_test_config();
        let lib1 = HealthStateLibrary::from_config(&config);
        let lib2 = HealthStateLibrary::from_config(&config);
        assert_eq!(lib1.len(), lib2.len());
        for (h1, h2) in lib1.hypotheses().iter().zip(lib2.hypotheses().iter()) {
            assert_eq!(h1.id, h2.id);
            assert_eq!(h1.component_levels, h2.component_levels);
            assert_eq!(h1.expected_signature, h2.expected_signature);
        }
    }

    #[test]
    fn test_entropy_certain() {
        // All probability on one hypothesis
        let post = vec![(0, 1.0), (1, 0.0), (2, 0.0)];
        let h = HealthStateLibrary::entropy(&post);
        assert!(
            h.abs() < 1e-6,
            "certain distribution should have ~0 entropy, got {h}"
        );
    }

    #[test]
    fn test_entropy_uniform() {
        // Uniform distribution over 4 hypotheses
        let post = vec![(0, 0.25), (1, 0.25), (2, 0.25), (3, 0.25)];
        let h = HealthStateLibrary::entropy(&post);
        let expected = -(4.0 * 0.25 * 0.25_f32.ln());
        assert!(
            (h - expected).abs() < 1e-4,
            "uniform entropy should be {expected}, got {h}"
        );
    }

    #[test]
    fn test_get_hypothesis_by_id() {
        let config = make_test_config();
        let lib = HealthStateLibrary::from_config(&config);
        let hyp = lib.get(0).unwrap();
        assert_eq!(hyp.id, 0);
        assert!(lib.get(lib.len() as u16).is_none());
    }

    #[test]
    fn test_level_to_integrity() {
        let n = 5;
        let floor = FIXED_POINT_ONE / 4; // 0.25
                                         // Level 0 = fully healthy
        assert_eq!(
            HealthStateLibrary::level_to_integrity(0, n, floor),
            FIXED_POINT_ONE
        );
        // Level n-1 = floor
        assert_eq!(
            HealthStateLibrary::level_to_integrity(n - 1, n, floor),
            floor
        );
        // Intermediate levels should be monotonically decreasing
        let mut prev = FIXED_POINT_ONE + 1;
        for i in 0..n {
            let v = HealthStateLibrary::level_to_integrity(i, n, floor);
            assert!(
                v < prev,
                "level {i} integrity {v} should be < previous {prev}"
            );
            assert!(
                v >= floor,
                "level {i} integrity {v} should be >= floor {floor}"
            );
            prev = v;
        }
    }

    #[test]
    fn test_squared_distance() {
        let a = [1.0, 0.0, 0.5];
        let b = [1.0, 0.0, 0.5];
        assert!((squared_distance(&a, &b)).abs() < 1e-10);

        let c = [0.0, 0.0, 0.0];
        let d = [1.0, 1.0, 1.0];
        assert!((squared_distance(&c, &d) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_is_empty() {
        let config = make_test_config();
        let lib = HealthStateLibrary::from_config(&config);
        assert!(!lib.is_empty());
    }
}
