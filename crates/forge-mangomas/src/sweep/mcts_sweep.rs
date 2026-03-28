//! MCTS hyperparameter sweep engine.
//!
//! Generates a grid of MCTS configurations, evaluates each in parallel
//! using FORGE environments, and reports the optimal configuration
//! for transfer to MangoMAS.

use std::time::Instant;

use forge_agent::mcts::tree::MctsConfig;
use rayon::prelude::*;
use tracing::{debug, info, instrument};

use crate::config::SweepConfig;
use crate::error::MangoMasResult;
use crate::sweep::results::{SweepReport, SweepResult};

/// Generates and evaluates MCTS hyperparameter configurations.
pub struct MctsParamSweep {
    config: SweepConfig,
}

impl MctsParamSweep {
    /// Creates a new sweep engine.
    #[instrument(skip_all)]
    pub fn new(config: SweepConfig) -> Self {
        Self { config }
    }

    /// Returns the sweep configuration.
    pub fn config(&self) -> &SweepConfig {
        &self.config
    }

    /// Generates the parameter grid as a list of MCTS configs to evaluate.
    #[instrument(skip_all)]
    pub fn generate_grid(&self) -> Vec<MctsConfig> {
        let c_puct_values = linspace(
            self.config.c_puct_range.0,
            self.config.c_puct_range.1,
            self.config.c_puct_steps,
        );
        let sim_budget_values = linspace_u32(
            self.config.sim_budget_range.0,
            self.config.sim_budget_range.1,
            self.config.sim_budget_steps,
        );
        let depth_values = linspace_u32(
            self.config.depth_range.0,
            self.config.depth_range.1,
            self.config.depth_steps,
        );
        let discount_values = linspace(
            self.config.discount_range.0,
            self.config.discount_range.1,
            self.config.discount_steps,
        );

        let mut grid = Vec::new();
        for &c_puct in &c_puct_values {
            for &sims in &sim_budget_values {
                for &depth in &depth_values {
                    for &discount in &discount_values {
                        grid.push(MctsConfig {
                            c_puct,
                            num_simulations: sims,
                            max_depth: depth,
                            discount,
                            ..MctsConfig::default()
                        });
                    }
                }
            }
        }

        debug!(grid_size = grid.len(), "parameter grid generated");
        grid
    }

    /// Runs the full sweep and returns a report.
    ///
    /// Each configuration is evaluated by running `episodes_per_config`
    /// episodes in FORGE and measuring mean reward.
    #[instrument(skip(self, evaluate_fn))]
    pub fn run_sweep<F>(&self, evaluate_fn: F) -> MangoMasResult<SweepReport>
    where
        F: Fn(&MctsConfig, u32) -> MangoMasResult<(f64, f64, f64)> + Send + Sync,
    {
        let grid = self.generate_grid();
        let start = Instant::now();

        let results: Vec<MangoMasResult<SweepResult>> = grid
            .par_iter()
            .map(|config| {
                let (mean_reward, std_reward, mean_time) =
                    evaluate_fn(config, self.config.episodes_per_config)?;
                Ok(SweepResult {
                    config: config.clone(),
                    mean_reward,
                    std_reward,
                    mean_planning_time_us: mean_time,
                    episodes_run: self.config.episodes_per_config,
                })
            })
            .collect();

        let mut collected = Vec::with_capacity(results.len());
        for result in results {
            collected.push(result?);
        }

        let elapsed = start.elapsed().as_secs_f64();
        let report = SweepReport::from_results(collected, elapsed);

        info!(
            grid_size = grid.len(),
            elapsed_secs = elapsed,
            best_reward = report.best.as_ref().map(|b| b.mean_reward),
            "sweep complete"
        );

        Ok(report)
    }
}

/// Generates `steps` evenly spaced f32 values in [min, max].
fn linspace(min: f32, max: f32, steps: u32) -> Vec<f32> {
    if steps == 0 {
        return vec![];
    }
    if steps == 1 {
        return vec![min];
    }
    let step_size = (max - min) / (steps - 1) as f32;
    (0..steps).map(|i| min + i as f32 * step_size).collect()
}

/// Generates `steps` evenly spaced u32 values in [min, max].
fn linspace_u32(min: u32, max: u32, steps: u32) -> Vec<u32> {
    if steps == 0 {
        return vec![];
    }
    if steps == 1 {
        return vec![min];
    }
    let step_size = (max - min) as f32 / (steps - 1) as f32;
    (0..steps)
        .map(|i| (min as f32 + i as f32 * step_size).round() as u32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SweepConfig;

    #[test]
    fn test_linspace() {
        let values = linspace(0.0, 1.0, 5);
        assert_eq!(values.len(), 5);
        assert!((values[0] - 0.0).abs() < 1e-6);
        assert!((values[4] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_linspace_single() {
        let values = linspace(0.5, 1.0, 1);
        assert_eq!(values.len(), 1);
        assert!((values[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_linspace_u32() {
        let values = linspace_u32(10, 100, 4);
        assert_eq!(values.len(), 4);
        assert_eq!(values[0], 10);
        assert_eq!(values[3], 100);
    }

    #[test]
    fn test_generate_grid_size() {
        let config = SweepConfig {
            c_puct_steps: 3,
            sim_budget_steps: 2,
            depth_steps: 2,
            discount_steps: 2,
            ..SweepConfig::default()
        };
        let sweep = MctsParamSweep::new(config);
        let grid = sweep.generate_grid();
        assert_eq!(grid.len(), 3 * 2 * 2 * 2);
    }

    #[test]
    fn test_run_sweep_with_mock() {
        let config = SweepConfig {
            c_puct_steps: 2,
            sim_budget_steps: 2,
            depth_steps: 1,
            discount_steps: 1,
            episodes_per_config: 1,
            ..SweepConfig::default()
        };
        let sweep = MctsParamSweep::new(config);
        let report = sweep
            .run_sweep(|config, _episodes| {
                // Mock: reward proportional to c_puct
                Ok((config.c_puct as f64, 0.1, 50.0))
            })
            .unwrap();

        assert_eq!(report.results.len(), 2 * 2);
        assert!(report.best.is_some());
    }

    #[test]
    fn test_sweep_empty_grid() {
        let config = SweepConfig {
            c_puct_steps: 0,
            ..Default::default()
        };
        let sweep = MctsParamSweep::new(config);
        let grid = sweep.generate_grid();
        assert!(grid.is_empty());
    }

    #[test]
    fn test_grid_configs_have_correct_ranges() {
        let config = SweepConfig {
            c_puct_range: (1.0, 2.0),
            c_puct_steps: 3,
            sim_budget_range: (50, 150),
            sim_budget_steps: 2,
            depth_steps: 1,
            discount_steps: 1,
            ..SweepConfig::default()
        };
        let sweep = MctsParamSweep::new(config);
        let grid = sweep.generate_grid();

        for config in &grid {
            assert!(config.c_puct >= 1.0 && config.c_puct <= 2.0);
            assert!(config.num_simulations >= 50 && config.num_simulations <= 150);
        }
    }
}
