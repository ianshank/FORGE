//! Platform-specific curriculum with adaptive difficulty progression.
//!
//! Defines 5-tier curricula for car and drone platforms, mapping
//! FORGE scenarios to progressive training stages.

use forge_task::curriculum::CurriculumController;
use forge_types::config::CurriculumConfig;
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::config::{Platform, PlatformCurriculumConfig};
use crate::error::MangoMasResult;

/// A single tier in the curriculum progression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierDefinition {
    /// Tier number (1-5).
    pub tier: u8,
    /// Human-readable tier name.
    pub name: String,
    /// FORGE scenario config file to use (e.g., "patrol", "escort").
    pub forge_scenario: String,
    /// Description of the training objective.
    pub description: String,
    /// Success threshold for this tier (0.0-1.0).
    pub success_threshold: f32,
}

/// Platform-specific curriculum manager.
///
/// Wraps FORGE's `CurriculumController` with platform-specific tier
/// definitions and scenario mappings.
pub struct PlatformCurriculum {
    controller: CurriculumController,
    platform: Platform,
    tier_definitions: Vec<TierDefinition>,
}

impl PlatformCurriculum {
    /// Creates a new curriculum for the given platform.
    #[instrument(skip_all)]
    pub fn new(config: PlatformCurriculumConfig) -> Self {
        let curriculum_config = CurriculumConfig {
            enabled: true,
            target_success_rate: config.target_success_rate,
            window_size: config.window_size,
            warmup_episodes: config.warmup_episodes,
            adjustment_rate: config.adjustment_rate,
        };

        let tier_definitions = match config.platform {
            Platform::Car => Self::car_tiers(),
            Platform::Drone => Self::drone_tiers(),
        };

        Self {
            controller: CurriculumController::new(curriculum_config),
            platform: config.platform,
            tier_definitions,
        }
    }

    /// Returns the current tier definitions.
    pub fn tier_definitions(&self) -> &[TierDefinition] {
        &self.tier_definitions
    }

    /// Returns the platform.
    pub fn platform(&self) -> Platform {
        self.platform
    }

    /// Records the outcome of a training episode.
    #[instrument(skip(self))]
    pub fn record_outcome(&mut self, success: bool) {
        self.controller.record_outcome(success);
    }

    /// Samples a tier from the current adaptive distribution.
    #[instrument(skip(self, rng))]
    pub fn sample_tier<R: rand::Rng>(&self, rng: &mut R) -> MangoMasResult<&TierDefinition> {
        let tier = self
            .controller
            .sample_tier(rng, self.tier_definitions.len() as u8);
        let idx = (tier as usize)
            .saturating_sub(1)
            .min(self.tier_definitions.len() - 1);
        Ok(&self.tier_definitions[idx])
    }

    /// Returns the current success rate.
    pub fn success_rate(&self) -> f32 {
        self.controller.success_rate()
    }

    /// Returns the total number of episodes recorded.
    pub fn total_episodes(&self) -> u64 {
        self.controller.total_episodes()
    }

    /// Returns the current tier weight distribution.
    pub fn tier_weights(&self) -> &[f32] {
        self.controller.tier_weights()
    }

    /// Defines the 5-tier car curriculum.
    fn car_tiers() -> Vec<TierDefinition> {
        vec![
            TierDefinition {
                tier: 1,
                name: "Straight Navigation".to_string(),
                forge_scenario: "patrol".to_string(),
                description: "Navigate to a single waypoint on open terrain".to_string(),
                success_threshold: 0.7,
            },
            TierDefinition {
                tier: 2,
                name: "Obstacle Avoidance".to_string(),
                forge_scenario: "patrol".to_string(),
                description: "Navigate while avoiding static obstacles".to_string(),
                success_threshold: 0.6,
            },
            TierDefinition {
                tier: 3,
                name: "Multi-Waypoint Route".to_string(),
                forge_scenario: "patrol".to_string(),
                description: "Complete a sequence of waypoints in order".to_string(),
                success_threshold: 0.5,
            },
            TierDefinition {
                tier: 4,
                name: "Dynamic Obstacles".to_string(),
                forge_scenario: "area_denial".to_string(),
                description: "Navigate with moving obstacles and sensor dropout".to_string(),
                success_threshold: 0.4,
            },
            TierDefinition {
                tier: 5,
                name: "Full Mission".to_string(),
                forge_scenario: "escort".to_string(),
                description: "Complete multi-step mission with resource constraints".to_string(),
                success_threshold: 0.3,
            },
        ]
    }

    /// Defines the 5-tier drone curriculum.
    fn drone_tiers() -> Vec<TierDefinition> {
        vec![
            TierDefinition {
                tier: 1,
                name: "Hover and Altitude".to_string(),
                forge_scenario: "patrol".to_string(),
                description: "Maintain stable hover and altitude control".to_string(),
                success_threshold: 0.7,
            },
            TierDefinition {
                tier: 2,
                name: "Waypoint Navigation".to_string(),
                forge_scenario: "patrol".to_string(),
                description: "Navigate to waypoints with altitude management".to_string(),
                success_threshold: 0.6,
            },
            TierDefinition {
                tier: 3,
                name: "Patrol Pattern".to_string(),
                forge_scenario: "patrol".to_string(),
                description: "Execute patrol patterns with battery management".to_string(),
                success_threshold: 0.5,
            },
            TierDefinition {
                tier: 4,
                name: "Search and Rescue".to_string(),
                forge_scenario: "search_and_rescue".to_string(),
                description: "Search grid areas and locate targets".to_string(),
                success_threshold: 0.4,
            },
            TierDefinition {
                tier: 5,
                name: "Multi-Drone Escort".to_string(),
                forge_scenario: "escort".to_string(),
                description: "Coordinate with other drones for escort mission".to_string(),
                success_threshold: 0.3,
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_pcg::Pcg64;

    fn make_rng() -> Pcg64 {
        Pcg64::seed_from_u64(42)
    }

    #[test]
    fn test_car_curriculum_has_5_tiers() {
        let curriculum = PlatformCurriculum::new(PlatformCurriculumConfig {
            platform: Platform::Car,
            ..PlatformCurriculumConfig::default()
        });
        assert_eq!(curriculum.tier_definitions().len(), 5);
        assert_eq!(curriculum.platform(), Platform::Car);
    }

    #[test]
    fn test_drone_curriculum_has_5_tiers() {
        let curriculum = PlatformCurriculum::new(PlatformCurriculumConfig {
            platform: Platform::Drone,
            ..PlatformCurriculumConfig::default()
        });
        assert_eq!(curriculum.tier_definitions().len(), 5);
    }

    #[test]
    fn test_tier_numbers_sequential() {
        let curriculum = PlatformCurriculum::new(PlatformCurriculumConfig::default());
        for (i, tier) in curriculum.tier_definitions().iter().enumerate() {
            assert_eq!(tier.tier, (i + 1) as u8);
        }
    }

    #[test]
    fn test_sample_tier() {
        let curriculum = PlatformCurriculum::new(PlatformCurriculumConfig::default());
        let mut rng = make_rng();
        let tier = curriculum.sample_tier(&mut rng).unwrap();
        assert!(tier.tier >= 1 && tier.tier <= 5);
    }

    #[test]
    fn test_record_outcome_tracks() {
        let mut curriculum = PlatformCurriculum::new(PlatformCurriculumConfig::default());
        assert_eq!(curriculum.total_episodes(), 0);
        curriculum.record_outcome(true);
        curriculum.record_outcome(false);
        assert_eq!(curriculum.total_episodes(), 2);
    }

    #[test]
    fn test_success_rate_computation() {
        let mut curriculum = PlatformCurriculum::new(PlatformCurriculumConfig::default());
        for _ in 0..7 {
            curriculum.record_outcome(true);
        }
        for _ in 0..3 {
            curriculum.record_outcome(false);
        }
        assert!((curriculum.success_rate() - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_success_thresholds_decrease_with_tier() {
        let curriculum = PlatformCurriculum::new(PlatformCurriculumConfig {
            platform: Platform::Car,
            ..PlatformCurriculumConfig::default()
        });
        let tiers = curriculum.tier_definitions();
        for i in 1..tiers.len() {
            assert!(
                tiers[i].success_threshold <= tiers[i - 1].success_threshold,
                "tier {} threshold ({}) should be <= tier {} threshold ({})",
                tiers[i].tier,
                tiers[i].success_threshold,
                tiers[i - 1].tier,
                tiers[i - 1].success_threshold,
            );
        }
    }
}
