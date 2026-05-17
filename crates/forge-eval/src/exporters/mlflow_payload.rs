//! Shared helpers used by every MLflow sink (filesystem today, HTTP in
//! the upcoming Slice 2). Extracted from `mlflow.rs` so both sinks
//! consume one implementation rather than diverging copies.
//!
//! Scope: **pure helpers** only — small fragment-level utilities that
//! either format a string or derive a digest. The composite `RunPayload`
//! struct + `build_run_payload` that ties them into a sink-agnostic
//! intermediate representation arrives in Slice 1.2 alongside the
//! `MlflowFsSink` extraction; until then `mlflow.rs` re-exports each of
//! these so existing call sites stay byte-identical.
//!
//! All helpers are `pub`: when the HTTP sink lands in Slice 2 it consumes
//! the same `child_run_id`, the same `combined_scenario_digest`, the same
//! `render_tier_bar_chart_html`, etc. Drift between sinks is impossible
//! by construction.

use sha2::{Digest, Sha256};

use crate::manifest::RunManifest;
use crate::scorecard::TierScore;

/// Deterministic per-scenario child-run id derived from the parent run id
/// and the scenario id. Re-running export with the same `(parent, scenario)`
/// pair overwrites the same child run rather than creating a new one —
/// the idempotency guarantee both sinks rely on.
///
/// Output is a lower-case hex string of the first 16 bytes of the SHA-256
/// digest (32 hex chars total).
pub fn child_run_id(parent_run_id: &str, scenario_id: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(parent_run_id.as_bytes());
    hasher.update(b"::");
    hasher.update(scenario_id.as_bytes());
    let digest = hasher.finalize();
    hex_short(&digest, 16)
}

/// Combined sha256 over every scenario-file digest in the manifest.
/// Surfaced as the `forge.eval.scenarios_digest` tag in lieu of MLflow's
/// `inputs/` directory tree (which is version-fragile). Shared between
/// the filesystem and HTTP sinks so a re-run with identical scenarios
/// produces a stable digest regardless of transport.
pub fn combined_scenario_digest(manifest: &RunManifest) -> String {
    let mut hasher = Sha256::new();
    for (_, hash) in &manifest.scenario_file_hashes {
        hasher.update(hash.as_bytes());
    }
    hex_short(&hasher.finalize(), 16)
}

/// Self-contained Plotly HTML for the per-tier success-rate + mean-reward
/// chart. Emitted as the `tier_success_rates.html` artefact under each
/// parent run's `artifacts/` directory.
///
/// `run_name` is interpolated into the chart title so a side-by-side view
/// of multiple runs in the MLflow UI is distinguishable at a glance.
pub fn render_tier_bar_chart_html(tier_scores: &[TierScore], run_name: &str) -> String {
    let tiers: Vec<u8> = tier_scores.iter().map(|t| t.tier).collect();
    let success: Vec<f64> = tier_scores.iter().map(|t| t.success_rate).collect();
    let reward: Vec<f64> = tier_scores.iter().map(|t| t.mean_reward).collect();
    format!(
        r#"<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>FORGE eval tier success rates — {run_name}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
<div id="chart" style="width:100%;height:480px;"></div>
<script>
Plotly.newPlot('chart', [
  {{x: {tiers:?}, y: {success:?}, type: 'bar', name: 'Success rate'}},
  {{x: {tiers:?}, y: {reward:?}, type: 'bar', name: 'Mean reward', yaxis: 'y2'}}
], {{
  title: 'Per-tier success rate + mean reward — {run_name}',
  xaxis: {{title: 'Difficulty tier'}},
  yaxis: {{title: 'Success rate', range: [0, 1]}},
  yaxis2: {{title: 'Mean reward', overlaying: 'y', side: 'right'}},
  barmode: 'group'
}});
</script>
</body>
</html>
"#
    )
}

/// MLflow allows alphanumerics + `_ - . / ` (space) in param/metric/tag
/// keys; anything else is rewritten to `_` so the filesystem write never
/// fails on a key that came from agent metadata.
pub fn sanitize(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || matches!(c, '_' | '-' | '.' | '/' | ' ') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Lower-case hex encoding of the first `len` bytes of `bytes`. Centralised
/// so every digest representation across the eval exporters renders the
/// same way (`child_run_id`, `combined_scenario_digest`, future request-id
/// fingerprints in the HTTP sink, ...).
pub fn hex_short(bytes: &[u8], len: usize) -> String {
    let mut out = String::with_capacity(len * 2);
    for b in bytes.iter().take(len) {
        use std::fmt::Write;
        write!(&mut out, "{:02x}", b).expect("write to string");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{RunManifest, MANIFEST_SOURCE_NAME, UNKNOWN};
    use crate::scorecard::TierScore;
    use chrono::Utc;
    use std::path::PathBuf;

    /// Build a minimal RunManifest with `scenario_file_hashes` populated;
    /// all other fields are stable sentinels so tests focus on the helper
    /// under test rather than wall-clock noise.
    fn manifest_with_hashes(hashes: Vec<(PathBuf, String)>) -> RunManifest {
        RunManifest {
            run_id: "test-run".to_string(),
            experiment_name: "test-exp".to_string(),
            timestamp: Utc::now(),
            git_sha: UNKNOWN.to_string(),
            git_branch: UNKNOWN.to_string(),
            rustc_version: UNKNOWN.to_string(),
            user: UNKNOWN.to_string(),
            config_hash: "test-config-hash".to_string(),
            scenario_file_hashes: hashes,
            source_name: MANIFEST_SOURCE_NAME.to_string(),
        }
    }

    /// `child_run_id` MUST be deterministic AND a stable 32-char lower-hex
    /// string. Re-export sites depend on both properties for idempotency.
    #[test]
    fn child_run_id_is_deterministic_and_hex_32() {
        let a = child_run_id("parent-abc", "scenario-1");
        let b = child_run_id("parent-abc", "scenario-1");
        assert_eq!(a, b, "same inputs → same id");
        assert_eq!(a.len(), 32);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));

        let c = child_run_id("parent-abc", "scenario-2");
        assert_ne!(a, c, "different scenario → different id");
        let d = child_run_id("parent-XYZ", "scenario-1");
        assert_ne!(a, d, "different parent → different id");
    }

    /// `combined_scenario_digest` is deterministic + insensitive to the
    /// path component (only the hash matters).
    #[test]
    fn combined_scenario_digest_is_deterministic_and_path_independent() {
        let m1 = manifest_with_hashes(vec![
            (PathBuf::from("a.toml"), "deadbeef".to_string()),
            (PathBuf::from("b.toml"), "cafebabe".to_string()),
        ]);
        let m2 = manifest_with_hashes(vec![
            (PathBuf::from("other/path/a.toml"), "deadbeef".to_string()),
            (PathBuf::from("elsewhere/b.toml"), "cafebabe".to_string()),
        ]);
        assert_eq!(combined_scenario_digest(&m1), combined_scenario_digest(&m2));
        assert_eq!(combined_scenario_digest(&m1).len(), 32);
    }

    /// `combined_scenario_digest` IS sensitive to hash content + order.
    #[test]
    fn combined_scenario_digest_reflects_hash_set_changes() {
        let mut m = manifest_with_hashes(vec![
            (PathBuf::from("a"), "deadbeef".to_string()),
            (PathBuf::from("b"), "cafebabe".to_string()),
        ]);
        let baseline = combined_scenario_digest(&m);

        // Changing a hash value changes the digest.
        m.scenario_file_hashes[0].1 = "feedface".to_string();
        assert_ne!(baseline, combined_scenario_digest(&m));

        // Reordering changes the digest (hash order is observable).
        let m_reordered = manifest_with_hashes(vec![
            (PathBuf::from("a"), "cafebabe".to_string()),
            (PathBuf::from("b"), "deadbeef".to_string()),
        ]);
        let m_original = manifest_with_hashes(vec![
            (PathBuf::from("a"), "deadbeef".to_string()),
            (PathBuf::from("b"), "cafebabe".to_string()),
        ]);
        assert_ne!(
            combined_scenario_digest(&m_reordered),
            combined_scenario_digest(&m_original)
        );
    }

    #[test]
    fn render_tier_bar_chart_html_is_self_contained_html() {
        let tiers = vec![
            TierScore {
                tier: 1,
                success_rate: 0.75,
                mean_reward: 1.2,
                mean_steps_to_completion: 10.0,
                episodes_evaluated: 4,
                scenarios_count: 1,
            },
            TierScore {
                tier: 2,
                success_rate: 0.5,
                mean_reward: 0.8,
                mean_steps_to_completion: 12.0,
                episodes_evaluated: 4,
                scenarios_count: 1,
            },
        ];
        let html = render_tier_bar_chart_html(&tiers, "run-test");
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("plotly-latest.min.js"));
        assert!(html.contains("run-test"));
        // Tier values + success rates make it into the embedded JSON.
        assert!(html.contains("[1, 2]"));
        assert!(html.contains("0.75"));
        assert!(html.contains("0.5"));
    }

    #[test]
    fn sanitize_replaces_disallowed_chars_with_underscore() {
        // Allowed: alphanumerics + _ - . / space.
        assert_eq!(sanitize("a-b.c/d_e f9"), "a-b.c/d_e f9");
        // Disallowed → underscores.
        assert_eq!(sanitize("a!b@c#d$e%f^"), "a_b_c_d_e_f_");
        // Unicode alphanumerics pass through.
        assert_eq!(sanitize("π_α"), "π_α");
        // Empty input returns empty.
        assert_eq!(sanitize(""), "");
    }

    #[test]
    fn hex_short_truncates_to_requested_byte_count() {
        let bytes = [0xde, 0xad, 0xbe, 0xef, 0xca, 0xfe];
        assert_eq!(hex_short(&bytes, 0), "");
        assert_eq!(hex_short(&bytes, 1), "de");
        assert_eq!(hex_short(&bytes, 3), "deadbe");
        assert_eq!(hex_short(&bytes, 6), "deadbeefcafe");
        // Asking for more bytes than available stops at the slice end.
        assert_eq!(hex_short(&bytes, 100), "deadbeefcafe");
    }
}
