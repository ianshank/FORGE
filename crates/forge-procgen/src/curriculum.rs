//! Adaptive difficulty curriculum controller.
//!
//! Tracks agent win rates over a rolling window and automatically adjusts
//! difficulty tiers to maintain an optimal learning challenge.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use tracing::instrument;

/// Constants governing how difficulty scales with tier progression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifficultyScaling {
    /// Minimum grid size at tier 1.
    pub grid_size_min: u16,
    /// Grid size range added across tiers.
    pub grid_size_range: f64,
    /// Minimum number of agents at tier 1.
    pub num_agents_min: u32,
    /// Agent count range added across tiers.
    pub num_agents_range: f64,
    /// Tier at which fog of war activates.
    pub fog_of_war_tier: u8,
    /// Minimum enemy skill at tier 1.
    pub enemy_skill_min: f64,
    /// Enemy skill range added across tiers.
    pub enemy_skill_range: f64,
    /// Minimum terrain complexity at tier 1.
    pub terrain_complexity_min: f64,
    /// Terrain complexity range added across tiers.
    pub terrain_complexity_range: f64,
    /// Minimum resource scarcity at tier 1.
    pub resource_scarcity_min: f64,
    /// Resource scarcity range added across tiers.
    pub resource_scarcity_range: f64,
}

impl Default for DifficultyScaling {
    fn default() -> Self {
        Self {
            grid_size_min: 32,
            grid_size_range: 96.0,
            num_agents_min: 2,
            num_agents_range: 6.0,
            fog_of_war_tier: 3,
            enemy_skill_min: 0.1,
            enemy_skill_range: 0.8,
            terrain_complexity_min: 0.2,
            terrain_complexity_range: 0.6,
            resource_scarcity_min: 0.1,
            resource_scarcity_range: 0.7,
        }
    }
}

/// Parameters describing the current difficulty level of the environment.
///
/// These parameters are adjusted by the [`CurriculumController`] as agents
/// demonstrate mastery or struggle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumParams {
    /// Grid dimension (square grids: grid_size x grid_size).
    pub grid_size: u16,
    /// Number of agents in the scenario.
    pub num_agents: u32,
    /// Whether fog of war is enabled.
    pub fog_of_war: bool,
    /// Enemy AI skill level (0.0 = passive, 1.0 = expert).
    pub enemy_skill: f64,
    /// Terrain complexity (0.0 = flat, 1.0 = highly varied).
    pub terrain_complexity: f64,
    /// Resource scarcity (0.0 = abundant, 1.0 = very scarce).
    pub resource_scarcity: f64,
}

impl Default for CurriculumParams {
    fn default() -> Self {
        Self {
            grid_size: 32,
            num_agents: 2,
            fog_of_war: false,
            enemy_skill: 0.1,
            terrain_complexity: 0.2,
            resource_scarcity: 0.1,
        }
    }
}

/// Adaptive difficulty controller that adjusts parameters based on win rate.
///
/// Maintains a rolling window of outcomes and advances or retreats
/// the difficulty tier when the win rate crosses configured thresholds.
#[derive(Debug, Clone)]
pub struct CurriculumController {
    /// Current curriculum parameters.
    params: CurriculumParams,
    /// Rolling window of win/loss outcomes.
    win_history: VecDeque<bool>,
    /// Size of the rolling window.
    window_size: usize,
    /// Win rate above which difficulty advances.
    advance_threshold: f64,
    /// Win rate below which difficulty retreats.
    retreat_threshold: f64,
    /// Current difficulty tier.
    current_tier: u8,
    /// Minimum allowed tier.
    min_tier: u8,
    /// Maximum allowed tier.
    max_tier: u8,
    /// Difficulty scaling configuration.
    scaling: DifficultyScaling,
}

impl CurriculumController {
    /// Creates a new curriculum controller with the given parameters.
    ///
    /// # Arguments
    /// * `window_size` — Number of recent games to consider for win rate.
    /// * `advance_threshold` — Win rate above which to increase difficulty.
    /// * `retreat_threshold` — Win rate below which to decrease difficulty.
    /// * `min_tier` — Lowest allowed difficulty tier.
    /// * `max_tier` — Highest allowed difficulty tier.
    #[instrument]
    pub fn new(
        window_size: usize,
        advance_threshold: f64,
        retreat_threshold: f64,
        min_tier: u8,
        max_tier: u8,
    ) -> Self {
        Self {
            params: CurriculumParams::default(),
            win_history: VecDeque::with_capacity(window_size),
            window_size,
            advance_threshold,
            retreat_threshold,
            current_tier: min_tier,
            min_tier,
            max_tier,
            scaling: DifficultyScaling::default(),
        }
    }

    /// Records a game outcome (win or loss).
    ///
    /// Maintains the rolling window by evicting the oldest entry when full.
    #[instrument(skip(self))]
    pub fn record_outcome(&mut self, won: bool) {
        if self.win_history.len() >= self.window_size {
            self.win_history.pop_front();
        }
        self.win_history.push_back(won);
    }

    /// Returns the current win rate over the rolling window.
    ///
    /// Returns 0.0 if no games have been recorded.
    #[instrument(skip(self))]
    pub fn current_win_rate(&self) -> f64 {
        if self.win_history.is_empty() {
            return 0.0;
        }
        let wins = self.win_history.iter().filter(|&&w| w).count();
        wins as f64 / self.win_history.len() as f64
    }

    /// Returns whether the current win rate exceeds the advance threshold.
    #[instrument(skip(self))]
    pub fn should_advance(&self) -> bool {
        self.current_win_rate() >= self.advance_threshold
    }

    /// Returns whether the current win rate is below the retreat threshold.
    #[instrument(skip(self))]
    pub fn should_retreat(&self) -> bool {
        self.current_win_rate() <= self.retreat_threshold
    }

    /// Automatically adjusts the difficulty tier based on current win rate.
    ///
    /// Advances the tier if the win rate is above the advance threshold,
    /// or retreats if below the retreat threshold. Updates curriculum
    /// parameters to match the new tier.
    #[instrument(skip(self))]
    pub fn adjust(&mut self) {
        if self.should_advance() && self.current_tier < self.max_tier {
            self.current_tier += 1;
            self.update_params();
        } else if self.should_retreat() && self.current_tier > self.min_tier {
            self.current_tier -= 1;
            self.update_params();
        }
    }

    /// Returns a reference to the current curriculum parameters.
    ///
    /// These parameters reflect the difficulty settings for the active tier
    /// and are updated automatically when the tier changes via [`Self::adjust`].
    #[instrument(skip(self))]
    pub fn current_params(&self) -> &CurriculumParams {
        &self.params
    }

    /// Returns the current difficulty tier.
    ///
    /// The tier is bounded by `min_tier` and `max_tier` as configured at
    /// construction time. Tier 1 is the easiest and higher tiers increase
    /// environment complexity.
    #[instrument(skip(self))]
    pub fn current_tier(&self) -> u8 {
        self.current_tier
    }

    /// Updates params to reflect the current tier.
    fn update_params(&mut self) {
        let t = self.current_tier as f64;
        let max = self.max_tier as f64;
        let fraction = (t - 1.0) / (max - 1.0).max(1.0);
        let s = &self.scaling;

        self.params.grid_size = s.grid_size_min + (fraction * s.grid_size_range) as u16;
        self.params.num_agents = s.num_agents_min + (fraction * s.num_agents_range) as u32;
        self.params.fog_of_war = self.current_tier >= s.fog_of_war_tier;
        self.params.enemy_skill = s.enemy_skill_min + fraction * s.enemy_skill_range;
        self.params.terrain_complexity =
            s.terrain_complexity_min + fraction * s.terrain_complexity_range;
        self.params.resource_scarcity =
            s.resource_scarcity_min + fraction * s.resource_scarcity_range;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_win_rate_empty() {
        let ctrl = CurriculumController::new(100, 0.7, 0.3, 1, 6);
        assert_eq!(ctrl.current_win_rate(), 0.0);
    }

    #[test]
    fn test_win_rate_calculation() {
        let mut ctrl = CurriculumController::new(10, 0.7, 0.3, 1, 6);
        for _ in 0..7 {
            ctrl.record_outcome(true);
        }
        for _ in 0..3 {
            ctrl.record_outcome(false);
        }
        let rate = ctrl.current_win_rate();
        assert!((rate - 0.7).abs() < 1e-10, "Expected 0.7, got {rate}");
    }

    #[test]
    fn test_rolling_window() {
        let mut ctrl = CurriculumController::new(5, 0.7, 0.3, 1, 6);
        // Fill with wins
        for _ in 0..5 {
            ctrl.record_outcome(true);
        }
        assert_eq!(ctrl.current_win_rate(), 1.0);
        // Add losses to push out wins
        for _ in 0..5 {
            ctrl.record_outcome(false);
        }
        assert_eq!(ctrl.current_win_rate(), 0.0);
    }

    #[test]
    fn test_should_advance() {
        let mut ctrl = CurriculumController::new(10, 0.7, 0.3, 1, 6);
        for _ in 0..8 {
            ctrl.record_outcome(true);
        }
        for _ in 0..2 {
            ctrl.record_outcome(false);
        }
        assert!(ctrl.should_advance());
        assert!(!ctrl.should_retreat());
    }

    #[test]
    fn test_should_retreat() {
        let mut ctrl = CurriculumController::new(10, 0.7, 0.3, 1, 6);
        for _ in 0..2 {
            ctrl.record_outcome(true);
        }
        for _ in 0..8 {
            ctrl.record_outcome(false);
        }
        assert!(!ctrl.should_advance());
        assert!(ctrl.should_retreat());
    }

    #[test]
    fn test_advance_tier() {
        let mut ctrl = CurriculumController::new(10, 0.7, 0.3, 1, 6);
        assert_eq!(ctrl.current_tier(), 1);
        for _ in 0..10 {
            ctrl.record_outcome(true);
        }
        ctrl.adjust();
        assert_eq!(ctrl.current_tier(), 2);
    }

    #[test]
    fn test_tier_upper_bound() {
        let mut ctrl = CurriculumController::new(5, 0.7, 0.3, 1, 3);
        // Advance to max
        for _ in 0..20 {
            for _ in 0..5 {
                ctrl.record_outcome(true);
            }
            ctrl.adjust();
        }
        assert_eq!(ctrl.current_tier(), 3);
    }

    #[test]
    fn test_tier_lower_bound() {
        let mut ctrl = CurriculumController::new(5, 0.7, 0.3, 2, 6);
        // All losses, try to retreat below min
        for _ in 0..20 {
            for _ in 0..5 {
                ctrl.record_outcome(false);
            }
            ctrl.adjust();
        }
        assert_eq!(ctrl.current_tier(), 2);
    }

    #[test]
    fn test_retreat_tier() {
        let mut ctrl = CurriculumController::new(5, 0.7, 0.3, 1, 6);
        // First advance
        for _ in 0..5 {
            ctrl.record_outcome(true);
        }
        ctrl.adjust();
        assert_eq!(ctrl.current_tier(), 2);
        // Clear window with losses
        for _ in 0..5 {
            ctrl.record_outcome(false);
        }
        ctrl.adjust();
        assert_eq!(ctrl.current_tier(), 1);
    }

    #[test]
    fn test_params_update_on_advance() {
        let mut ctrl = CurriculumController::new(5, 0.7, 0.3, 1, 6);
        let initial_skill = ctrl.current_params().enemy_skill;
        for _ in 0..5 {
            ctrl.record_outcome(true);
        }
        ctrl.adjust();
        assert!(
            ctrl.current_params().enemy_skill > initial_skill,
            "Enemy skill should increase on advance"
        );
    }

    #[test]
    fn test_tier_at_max_advance_stays_at_max() {
        let mut ctrl = CurriculumController::new(5, 0.7, 0.3, 1, 3);
        // Advance to max tier
        for _ in 0..50 {
            for _ in 0..5 {
                ctrl.record_outcome(true);
            }
            ctrl.adjust();
        }
        assert_eq!(ctrl.current_tier(), 3);
        // Try to advance further — should stay at max
        for _ in 0..5 {
            ctrl.record_outcome(true);
        }
        ctrl.adjust();
        assert_eq!(ctrl.current_tier(), 3);
    }

    #[test]
    fn test_retreat_at_min_stays_at_min() {
        let mut ctrl = CurriculumController::new(5, 0.7, 0.3, 1, 6);
        assert_eq!(ctrl.current_tier(), 1);
        // All losses, try to retreat below min_tier=1
        for _ in 0..5 {
            ctrl.record_outcome(false);
        }
        ctrl.adjust();
        assert_eq!(ctrl.current_tier(), 1);
    }

    #[test]
    fn test_empty_window_win_rate_zero() {
        let ctrl = CurriculumController::new(10, 0.7, 0.3, 1, 6);
        assert_eq!(ctrl.current_win_rate(), 0.0);
    }

    #[test]
    fn test_default_curriculum_params() {
        let params = CurriculumParams::default();
        assert_eq!(params.grid_size, 32);
        assert_eq!(params.num_agents, 2);
        assert!(!params.fog_of_war);
        assert!((params.enemy_skill - 0.1).abs() < 1e-10);
        assert!((params.terrain_complexity - 0.2).abs() < 1e-10);
        assert!((params.resource_scarcity - 0.1).abs() < 1e-10);
    }
}
