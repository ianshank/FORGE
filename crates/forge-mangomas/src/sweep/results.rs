//! Sweep result types for MCTS hyperparameter optimization.
//!
//! Results are serializable for export to MangoMAS configuration files.

use forge_agent::mcts::tree::MctsConfig;
use serde::{Deserialize, Serialize};

/// Result of evaluating a single MCTS parameter configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepResult {
    /// The MCTS configuration evaluated.
    pub config: MctsConfig,
    /// Mean reward across evaluation episodes.
    pub mean_reward: f64,
    /// Standard deviation of rewards.
    pub std_reward: f64,
    /// Mean planning time per step in microseconds.
    pub mean_planning_time_us: f64,
    /// Number of episodes evaluated.
    pub episodes_run: u32,
}

/// Full sweep report with all results and the best configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepReport {
    /// All evaluated configurations and their results.
    pub results: Vec<SweepResult>,
    /// The best configuration (highest mean reward).
    pub best: Option<SweepResult>,
    /// Total wall-clock time for the sweep in seconds.
    pub total_time_secs: f64,
}

impl SweepReport {
    /// Creates a new report from a list of results.
    pub fn from_results(results: Vec<SweepResult>, total_time_secs: f64) -> Self {
        let best = results
            .iter()
            .max_by(|a, b| a.mean_reward.total_cmp(&b.mean_reward))
            .cloned();
        Self {
            results,
            best,
            total_time_secs,
        }
    }
}

/// Comparison between PUCT and UCB1 selection strategies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PuctVsUcb1Report {
    /// PUCT results per scenario.
    pub puct_results: Vec<SweepResult>,
    /// UCB1 results per scenario.
    pub ucb1_results: Vec<SweepResult>,
    /// Mean reward advantage of PUCT over UCB1 (positive = PUCT better).
    pub puct_advantage: f64,
}

/// Results of surprise-adaptive budget validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurpriseValidationReport {
    /// Mean reward with adaptive budget.
    pub adaptive_mean_reward: f64,
    /// Mean reward with fixed budget.
    pub fixed_mean_reward: f64,
    /// Improvement ratio (adaptive / fixed).
    pub improvement_ratio: f64,
    /// Per-surprise-level results.
    pub per_level_results: Vec<SurpriseLevelResult>,
}

/// Results for a specific surprise level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurpriseLevelResult {
    /// Surprise level label.
    pub level: String,
    /// Simulation budget used.
    pub sim_budget: u32,
    /// Mean reward at this surprise level.
    pub mean_reward: f64,
    /// Number of episodes evaluated.
    pub episodes: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_result(reward: f64) -> SweepResult {
        SweepResult {
            config: MctsConfig::default(),
            mean_reward: reward,
            std_reward: 0.1,
            mean_planning_time_us: 50.0,
            episodes_run: 10,
        }
    }

    #[test]
    fn test_sweep_report_finds_best() {
        let results = vec![sample_result(1.0), sample_result(3.0), sample_result(2.0)];
        let report = SweepReport::from_results(results, 10.0);
        assert!(report.best.is_some());
        assert!((report.best.unwrap().mean_reward - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_sweep_report_empty() {
        let report = SweepReport::from_results(vec![], 0.0);
        assert!(report.best.is_none());
    }

    #[test]
    fn test_sweep_result_serde_roundtrip() {
        let result = sample_result(2.5);
        let json = serde_json::to_string(&result).unwrap();
        let deser: SweepResult = serde_json::from_str(&json).unwrap();
        assert!((deser.mean_reward - 2.5).abs() < 1e-6);
    }
}
