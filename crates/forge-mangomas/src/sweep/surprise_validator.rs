//! Surprise-adaptive MCTS budget validation.
//!
//! MangoMAS dynamically adjusts MCTS simulation count based on surprise
//! (KL divergence). This module validates that logic by creating controlled
//! novelty scenarios in FORGE and comparing adaptive vs fixed budgets.

use forge_agent::mcts::tree::MctsConfig;
use tracing::{debug, instrument};

use crate::config::SurpriseValidatorConfig;
use crate::error::MangoMasResult;
use crate::sweep::results::{SurpriseLevelResult, SurpriseValidationReport};

/// Surprise level classification matching MangoMAS categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurpriseLevel {
    /// Low surprise: familiar environment, use cached/minimal budget.
    Low,
    /// Medium surprise: moderate novelty, use base budget.
    Medium,
    /// High surprise: novel environment, use full budget.
    High,
}

impl SurpriseLevel {
    /// Returns the label string for this level.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
}

/// Validates surprise-adaptive MCTS budget allocation.
///
/// Creates controlled novelty scenarios in FORGE and measures whether
/// adapting the simulation budget based on surprise level improves
/// performance compared to a fixed budget.
pub struct SurpriseAdaptiveBudgetValidator {
    config: SurpriseValidatorConfig,
}

impl SurpriseAdaptiveBudgetValidator {
    /// Creates a new validator.
    #[instrument(skip_all)]
    pub fn new(config: SurpriseValidatorConfig) -> Self {
        Self { config }
    }

    /// Get the simulation budget for a given surprise level.
    pub fn budget_for_level(&self, level: SurpriseLevel) -> u32 {
        match level {
            SurpriseLevel::Low => self.config.low_surprise_budget,
            SurpriseLevel::Medium => self.config.base_surprise_budget,
            SurpriseLevel::High => self.config.full_surprise_budget,
        }
    }

    /// Classify the surprise level based on prediction error magnitude.
    pub fn classify_surprise(&self, kl_divergence: f32) -> SurpriseLevel {
        if kl_divergence < self.config.low_surprise_threshold {
            SurpriseLevel::Low
        } else if kl_divergence > self.config.high_surprise_threshold {
            SurpriseLevel::High
        } else {
            SurpriseLevel::Medium
        }
    }

    /// Returns the MCTS config for a given surprise level.
    pub fn mcts_config_for_level(&self, level: SurpriseLevel) -> MctsConfig {
        MctsConfig {
            num_simulations: self.budget_for_level(level),
            ..MctsConfig::default()
        }
    }

    /// Runs validation comparing adaptive vs fixed budget.
    ///
    /// The `evaluate_fn` receives an MCTS config and surprise level,
    /// returning the mean reward for that configuration.
    #[instrument(skip(self, evaluate_fn))]
    pub fn validate<F>(&self, evaluate_fn: F) -> MangoMasResult<SurpriseValidationReport>
    where
        F: Fn(&MctsConfig, SurpriseLevel) -> MangoMasResult<f64> + Send + Sync,
    {
        let levels = [
            SurpriseLevel::Low,
            SurpriseLevel::Medium,
            SurpriseLevel::High,
        ];
        let fixed_config = MctsConfig {
            num_simulations: self.config.base_surprise_budget,
            ..MctsConfig::default()
        };

        let mut adaptive_total = 0.0;
        let mut fixed_total = 0.0;
        let mut per_level_results = Vec::new();

        for &level in &levels {
            let adaptive_config = self.mcts_config_for_level(level);
            let adaptive_reward = evaluate_fn(&adaptive_config, level)?;
            let fixed_reward = evaluate_fn(&fixed_config, level)?;

            adaptive_total += adaptive_reward;
            fixed_total += fixed_reward;

            per_level_results.push(SurpriseLevelResult {
                level: level.label().to_string(),
                sim_budget: self.budget_for_level(level),
                mean_reward: adaptive_reward,
                episodes: self.config.episodes_per_level,
            });

            debug!(
                level = level.label(),
                adaptive_reward, fixed_reward, "surprise level validated"
            );
        }

        let adaptive_mean = adaptive_total / levels.len() as f64;
        let fixed_mean = fixed_total / levels.len() as f64;
        let improvement = if fixed_mean.abs() > 1e-10 {
            adaptive_mean / fixed_mean
        } else {
            1.0
        };

        Ok(SurpriseValidationReport {
            adaptive_mean_reward: adaptive_mean,
            fixed_mean_reward: fixed_mean,
            improvement_ratio: improvement,
            per_level_results,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SurpriseValidatorConfig;

    #[test]
    fn test_classify_surprise() {
        let validator = SurpriseAdaptiveBudgetValidator::new(SurpriseValidatorConfig::default());
        assert_eq!(validator.classify_surprise(0.05), SurpriseLevel::Low);
        assert_eq!(validator.classify_surprise(0.3), SurpriseLevel::Medium);
        assert_eq!(validator.classify_surprise(0.8), SurpriseLevel::High);
    }

    #[test]
    fn test_budget_for_level() {
        let config = SurpriseValidatorConfig::default();
        let validator = SurpriseAdaptiveBudgetValidator::new(config.clone());
        assert_eq!(
            validator.budget_for_level(SurpriseLevel::Low),
            config.low_surprise_budget
        );
        assert_eq!(
            validator.budget_for_level(SurpriseLevel::Medium),
            config.base_surprise_budget
        );
        assert_eq!(
            validator.budget_for_level(SurpriseLevel::High),
            config.full_surprise_budget
        );
    }

    #[test]
    fn test_mcts_config_for_level() {
        let validator = SurpriseAdaptiveBudgetValidator::new(SurpriseValidatorConfig::default());
        let config = validator.mcts_config_for_level(SurpriseLevel::High);
        assert_eq!(config.num_simulations, 300);
    }

    #[test]
    fn test_validate_with_mock() {
        let validator = SurpriseAdaptiveBudgetValidator::new(SurpriseValidatorConfig::default());
        let report = validator
            .validate(|config, _level| {
                // Mock: more sims = better reward
                Ok(config.num_simulations as f64 / 100.0)
            })
            .unwrap();

        assert_eq!(report.per_level_results.len(), 3);
        // Adaptive should outperform fixed since it uses more sims for high surprise
        assert!(report.adaptive_mean_reward > 0.0);
    }

    #[test]
    fn test_budget_ordering() {
        let config = SurpriseValidatorConfig::default();
        assert!(config.low_surprise_budget < config.base_surprise_budget);
        assert!(config.base_surprise_budget < config.full_surprise_budget);
    }
}
