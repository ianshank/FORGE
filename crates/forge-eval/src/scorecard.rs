//! Evaluation scorecard: aggregated results across scenarios and tiers.
//!
//! The [`Scorecard`] is the primary output of an evaluation run. It contains
//! per-tier success rates, mean rewards, and timing data — suitable for
//! leaderboard display, regression testing, and agent comparison.

use forge_types::agent_interface::AgentMetadata;
use serde::{Deserialize, Serialize};
use tracing::instrument;

/// Aggregated evaluation results for a single agent across all scenarios.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scorecard {
    /// Metadata about the evaluated agent.
    pub agent_metadata: AgentMetadata,
    /// ISO 8601 timestamp of evaluation.
    pub timestamp: String,
    /// Overall weighted score (0.0–1.0).
    pub overall_score: f64,
    /// Per-tier aggregated scores.
    pub tier_scores: Vec<TierScore>,
    /// Per-scenario detailed results.
    pub scenario_results: Vec<ScenarioResult>,
    /// Summary statistics.
    pub summary: SummaryStats,
}

/// Aggregated score for a single difficulty tier.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TierScore {
    /// Difficulty tier (1–6).
    pub tier: u8,
    /// Fraction of episodes where all tasks were completed.
    pub success_rate: f64,
    /// Mean total reward across episodes.
    pub mean_reward: f64,
    /// Mean steps to completion (for successful episodes).
    pub mean_steps_to_completion: f64,
    /// Total episodes evaluated at this tier.
    pub episodes_evaluated: u32,
    /// Number of scenarios at this tier.
    pub scenarios_count: u32,
}

/// Results for a single scenario.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioResult {
    /// Scenario identifier.
    pub scenario_id: String,
    /// Scenario difficulty tier.
    pub tier: u8,
    /// Per-episode results.
    pub episodes: Vec<EpisodeResult>,
    /// Success rate across episodes.
    pub success_rate: f64,
    /// Mean reward across episodes.
    pub mean_reward: f64,
    /// Mean decision time in milliseconds.
    pub mean_decision_time_ms: f64,
}

/// Results for a single episode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeResult {
    /// Seed used for this episode.
    pub seed: u64,
    /// Total reward accumulated.
    pub total_reward: f64,
    /// Whether the episode was successful (all tasks completed).
    pub success: bool,
    /// Number of steps the episode ran.
    pub steps: u64,
    /// Whether the episode terminated naturally.
    pub terminated: bool,
    /// Whether the episode was truncated.
    pub truncated: bool,
    /// Mean decision time per step in milliseconds.
    pub mean_decision_time_ms: f64,
}

/// Summary statistics for the full evaluation run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SummaryStats {
    /// Total episodes evaluated.
    pub total_episodes: u32,
    /// Total simulation steps across all episodes.
    pub total_steps: u64,
    /// Wall-clock time for the entire evaluation.
    pub wall_clock_seconds: f64,
    /// Mean agent decision latency.
    pub mean_decision_latency_ms: f64,
}

impl Scorecard {
    /// Computes the overall score as a weighted average of tier success rates.
    ///
    /// Higher tiers are weighted more heavily (tier weight = tier number).
    #[instrument(skip_all)]
    pub fn compute_overall_score(tier_scores: &[TierScore]) -> f64 {
        if tier_scores.is_empty() {
            return 0.0;
        }
        let total_weight: f64 = tier_scores.iter().map(|t| t.tier as f64).sum();
        if total_weight == 0.0 {
            return 0.0;
        }
        let weighted_sum: f64 = tier_scores
            .iter()
            .map(|t| t.tier as f64 * t.success_rate)
            .sum();
        weighted_sum / total_weight
    }

    /// Serializes the scorecard to pretty-printed JSON.
    ///
    /// Returns an error if serialization fails.
    #[instrument(skip_all)]
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| format!("Scorecard JSON serialization failed: {e}"))
    }

    /// Deserializes a scorecard from JSON.
    #[instrument(skip_all)]
    pub fn from_json(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| format!("JSON deserialization failed: {e}"))
    }

    /// Generates a Markdown summary of the scorecard.
    #[instrument(skip_all)]
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();
        md.push_str(&format!(
            "# FORGE Evaluation Scorecard\n\n**Agent**: {} ({})\n**Overall Score**: {:.1}%\n**Date**: {}\n\n",
            self.agent_metadata.model_name,
            self.agent_metadata.agent_type,
            self.overall_score * 100.0,
            self.timestamp,
        ));

        md.push_str("## Per-Tier Results\n\n");
        md.push_str("| Tier | Success Rate | Mean Reward | Mean Steps | Episodes |\n");
        md.push_str("|------|-------------|-------------|------------|----------|\n");
        for tier in &self.tier_scores {
            md.push_str(&format!(
                "| {} | {:.1}% | {:.2} | {:.0} | {} |\n",
                tier.tier,
                tier.success_rate * 100.0,
                tier.mean_reward,
                tier.mean_steps_to_completion,
                tier.episodes_evaluated,
            ));
        }

        md.push_str(&format!(
            "\n## Summary\n\n- **Total Episodes**: {}\n- **Total Steps**: {}\n- **Wall Clock**: {:.1}s\n- **Mean Decision Latency**: {:.1}ms\n",
            self.summary.total_episodes,
            self.summary.total_steps,
            self.summary.wall_clock_seconds,
            self.summary.mean_decision_latency_ms,
        ));

        md
    }
}

impl ScenarioResult {
    /// Computes aggregated metrics from episode results.
    #[instrument(skip(episodes))]
    pub fn from_episodes(scenario_id: String, tier: u8, episodes: Vec<EpisodeResult>) -> Self {
        let count = episodes.len() as f64;
        let success_rate = if count > 0.0 {
            episodes.iter().filter(|e| e.success).count() as f64 / count
        } else {
            0.0
        };
        let mean_reward = if count > 0.0 {
            episodes.iter().map(|e| e.total_reward).sum::<f64>() / count
        } else {
            0.0
        };
        let mean_decision_time_ms = if count > 0.0 {
            episodes
                .iter()
                .map(|e| e.mean_decision_time_ms)
                .sum::<f64>()
                / count
        } else {
            0.0
        };

        Self {
            scenario_id,
            tier,
            episodes,
            success_rate,
            mean_reward,
            mean_decision_time_ms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_episode(seed: u64, reward: f64, success: bool, steps: u64) -> EpisodeResult {
        EpisodeResult {
            seed,
            total_reward: reward,
            success,
            steps,
            terminated: success,
            truncated: !success,
            mean_decision_time_ms: 1.0,
        }
    }

    #[test]
    fn test_overall_score_empty() {
        assert_eq!(Scorecard::compute_overall_score(&[]), 0.0);
    }

    #[test]
    fn test_overall_score_single_tier() {
        let tiers = vec![TierScore {
            tier: 1,
            success_rate: 0.5,
            ..Default::default()
        }];
        assert!((Scorecard::compute_overall_score(&tiers) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_overall_score_weighted() {
        let tiers = vec![
            TierScore {
                tier: 1,
                success_rate: 1.0,
                ..Default::default()
            },
            TierScore {
                tier: 2,
                success_rate: 0.0,
                ..Default::default()
            },
        ];
        // Weighted: (1*1.0 + 2*0.0) / (1+2) = 1/3
        let score = Scorecard::compute_overall_score(&tiers);
        assert!((score - 1.0 / 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_scenario_result_from_episodes() {
        let episodes = vec![
            make_episode(0, 1.0, true, 50),
            make_episode(1, 0.5, false, 100),
            make_episode(2, 0.8, true, 60),
        ];

        let result = ScenarioResult::from_episodes("test_scenario".into(), 1, episodes);
        assert_eq!(result.scenario_id, "test_scenario");
        assert_eq!(result.tier, 1);
        assert!((result.success_rate - 2.0 / 3.0).abs() < f64::EPSILON);
        assert!((result.mean_reward - (1.0 + 0.5 + 0.8) / 3.0).abs() < 0.001);
    }

    #[test]
    fn test_scenario_result_empty_episodes() {
        let result = ScenarioResult::from_episodes("empty".into(), 1, vec![]);
        assert_eq!(result.success_rate, 0.0);
        assert_eq!(result.mean_reward, 0.0);
    }

    #[test]
    fn test_scorecard_json_roundtrip() {
        let scorecard = Scorecard {
            agent_metadata: AgentMetadata::heuristic("NoopAgent"),
            timestamp: "2026-03-29T00:00:00Z".to_string(),
            overall_score: 0.42,
            tier_scores: vec![TierScore {
                tier: 1,
                success_rate: 0.42,
                mean_reward: 1.5,
                mean_steps_to_completion: 50.0,
                episodes_evaluated: 10,
                scenarios_count: 2,
            }],
            scenario_results: vec![],
            summary: SummaryStats {
                total_episodes: 10,
                total_steps: 500,
                wall_clock_seconds: 1.5,
                mean_decision_latency_ms: 0.5,
            },
        };

        let json = scorecard.to_json().unwrap();
        let deser = Scorecard::from_json(&json).unwrap();
        assert_eq!(deser.overall_score, 0.42);
        assert_eq!(deser.tier_scores[0].tier, 1);
        assert_eq!(deser.summary.total_episodes, 10);
    }

    #[test]
    fn test_scorecard_from_json_invalid() {
        let result = Scorecard::from_json("not valid json");
        assert!(result.is_err());
    }

    #[test]
    fn test_scorecard_markdown() {
        let scorecard = Scorecard {
            agent_metadata: AgentMetadata::heuristic("TestAgent"),
            timestamp: "2026-03-29".to_string(),
            overall_score: 0.75,
            tier_scores: vec![TierScore {
                tier: 1,
                success_rate: 0.8,
                mean_reward: 2.0,
                mean_steps_to_completion: 30.0,
                episodes_evaluated: 10,
                scenarios_count: 1,
            }],
            scenario_results: vec![],
            summary: SummaryStats {
                total_episodes: 10,
                total_steps: 300,
                wall_clock_seconds: 0.5,
                mean_decision_latency_ms: 0.1,
            },
        };

        let md = scorecard.to_markdown();
        assert!(md.contains("FORGE Evaluation Scorecard"));
        assert!(md.contains("TestAgent"));
        assert!(md.contains("75.0%"));
        assert!(md.contains("80.0%"));
    }

    #[test]
    fn test_episode_result_fields() {
        let ep = make_episode(42, 3.5, true, 100);
        assert_eq!(ep.seed, 42);
        assert!((ep.total_reward - 3.5).abs() < f64::EPSILON);
        assert!(ep.success);
        assert_eq!(ep.steps, 100);
        assert!(ep.terminated);
        assert!(!ep.truncated);
    }

    #[test]
    fn test_tier_score_default() {
        let tier = TierScore::default();
        assert_eq!(tier.tier, 0);
        assert_eq!(tier.success_rate, 0.0);
        assert_eq!(tier.episodes_evaluated, 0);
    }

    #[test]
    fn test_overall_score_all_tiers_equal() {
        let tiers: Vec<TierScore> = (1..=6)
            .map(|t| TierScore {
                tier: t,
                success_rate: 0.5,
                ..Default::default()
            })
            .collect();
        let score = Scorecard::compute_overall_score(&tiers);
        assert!(
            (score - 0.5).abs() < f64::EPSILON,
            "equal success rates should yield that rate as overall"
        );
    }

    #[test]
    fn test_overall_score_higher_tiers_weighted_more() {
        let tiers = vec![
            TierScore {
                tier: 1,
                success_rate: 0.0,
                ..Default::default()
            },
            TierScore {
                tier: 6,
                success_rate: 1.0,
                ..Default::default()
            },
        ];
        let score = Scorecard::compute_overall_score(&tiers);
        // Weighted: (1*0.0 + 6*1.0) / (1+6) = 6/7 ≈ 0.857
        assert!(
            (score - 6.0 / 7.0).abs() < 0.001,
            "higher tier should dominate: got {score}"
        );
    }

    #[test]
    fn test_overall_score_zero_tier_returns_zero() {
        let tiers = vec![TierScore {
            tier: 0,
            success_rate: 1.0,
            ..Default::default()
        }];
        // tier 0 has weight 0, so total weight is 0
        assert_eq!(Scorecard::compute_overall_score(&tiers), 0.0);
    }

    #[test]
    fn test_scenario_result_decision_time_aggregation() {
        let episodes = vec![
            EpisodeResult {
                seed: 0,
                total_reward: 0.0,
                success: false,
                steps: 10,
                terminated: false,
                truncated: true,
                mean_decision_time_ms: 2.0,
            },
            EpisodeResult {
                seed: 1,
                total_reward: 0.0,
                success: false,
                steps: 10,
                terminated: false,
                truncated: true,
                mean_decision_time_ms: 4.0,
            },
            EpisodeResult {
                seed: 2,
                total_reward: 0.0,
                success: false,
                steps: 10,
                terminated: false,
                truncated: true,
                mean_decision_time_ms: 6.0,
            },
        ];
        let result = ScenarioResult::from_episodes("test".into(), 1, episodes);
        assert!(
            (result.mean_decision_time_ms - 4.0).abs() < f64::EPSILON,
            "mean of [2,4,6] should be 4.0, got {}",
            result.mean_decision_time_ms
        );
    }

    #[test]
    fn test_scenario_result_all_success() {
        let episodes = vec![
            make_episode(0, 1.0, true, 50),
            make_episode(1, 1.0, true, 60),
        ];
        let result = ScenarioResult::from_episodes("all_pass".into(), 2, episodes);
        assert!((result.success_rate - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_scenario_result_all_failure() {
        let episodes = vec![
            make_episode(0, 0.0, false, 100),
            make_episode(1, 0.0, false, 100),
        ];
        let result = ScenarioResult::from_episodes("all_fail".into(), 3, episodes);
        assert!((result.success_rate - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_scorecard_json_special_characters() {
        let mut meta = AgentMetadata::heuristic("Agent \"with quotes\"");
        meta.agent_type = "test <type>".to_string();
        let sc = Scorecard {
            agent_metadata: meta,
            timestamp: "2026-04-14".to_string(),
            overall_score: 0.5,
            tier_scores: vec![],
            scenario_results: vec![],
            summary: SummaryStats::default(),
        };
        let json = sc.to_json().unwrap();
        let deser = Scorecard::from_json(&json).unwrap();
        assert_eq!(deser.agent_metadata.model_name, "Agent \"with quotes\"");
    }

    #[test]
    fn test_scorecard_markdown_contains_all_sections() {
        let sc = Scorecard {
            agent_metadata: AgentMetadata::heuristic("MarkdownAgent"),
            timestamp: "2026-04-14".to_string(),
            overall_score: 0.75,
            tier_scores: vec![
                TierScore {
                    tier: 1,
                    success_rate: 0.8,
                    mean_reward: 2.0,
                    mean_steps_to_completion: 30.0,
                    episodes_evaluated: 10,
                    scenarios_count: 1,
                },
                TierScore {
                    tier: 2,
                    success_rate: 0.6,
                    mean_reward: 1.5,
                    mean_steps_to_completion: 50.0,
                    episodes_evaluated: 10,
                    scenarios_count: 1,
                },
            ],
            scenario_results: vec![],
            summary: SummaryStats {
                total_episodes: 20,
                total_steps: 800,
                wall_clock_seconds: 2.5,
                mean_decision_latency_ms: 0.3,
            },
        };
        let md = sc.to_markdown();
        assert!(md.contains("MarkdownAgent"));
        assert!(md.contains("75.0%"));
        assert!(md.contains("Per-Tier Results"));
        assert!(md.contains("Summary"));
        assert!(md.contains("Total Episodes"));
        assert!(md.contains("20"));
    }

    #[test]
    fn test_summary_stats_default() {
        let s = SummaryStats::default();
        assert_eq!(s.total_episodes, 0);
        assert_eq!(s.total_steps, 0);
        assert_eq!(s.wall_clock_seconds, 0.0);
        assert_eq!(s.mean_decision_latency_ms, 0.0);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Overall score is always in [0.0, 1.0] when success rates are in [0.0, 1.0].
        #[test]
        fn overall_score_bounded(
            s1 in 0.0_f64..=1.0,
            s2 in 0.0_f64..=1.0,
            s3 in 0.0_f64..=1.0,
        ) {
            let tiers = vec![
                TierScore { tier: 1, success_rate: s1, ..Default::default() },
                TierScore { tier: 2, success_rate: s2, ..Default::default() },
                TierScore { tier: 3, success_rate: s3, ..Default::default() },
            ];
            let score = Scorecard::compute_overall_score(&tiers);
            prop_assert!(score >= 0.0);
            prop_assert!(score <= 1.0);
        }

        /// Scenario success rate is always in [0.0, 1.0].
        #[test]
        fn scenario_success_rate_bounded(
            n_success in 0_u32..=20,
            n_fail in 0_u32..=20,
        ) {
            let total = n_success + n_fail;
            if total == 0 { return Ok(()); }
            let mut episodes = Vec::new();
            for i in 0..n_success {
                episodes.push(EpisodeResult {
                    seed: i as u64, total_reward: 1.0, success: true,
                    steps: 10, terminated: true, truncated: false,
                    mean_decision_time_ms: 1.0,
                });
            }
            for i in 0..n_fail {
                episodes.push(EpisodeResult {
                    seed: (n_success + i) as u64, total_reward: 0.0, success: false,
                    steps: 100, terminated: false, truncated: true,
                    mean_decision_time_ms: 1.0,
                });
            }
            let result = ScenarioResult::from_episodes("test".into(), 1, episodes);
            prop_assert!(result.success_rate >= 0.0);
            prop_assert!(result.success_rate <= 1.0);
        }
    }
}
