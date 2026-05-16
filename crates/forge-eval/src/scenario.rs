//! Scenario and suite definitions for the evaluation harness.
//!
//! A [`Scenario`] pairs a [`ForgeConfig`] with an identifier, difficulty tier,
//! and optional metadata. A [`ScenarioSuite`] is an ordered collection of
//! scenarios, typically loaded from a directory of TOML files.
//!
//! # Backward compatibility
//!
//! All fields except `id`, `tier`, and `forge_config` are optional and carry
//! `#[serde(default)]`, so older scenario files (or programmatically
//! constructed ones) keep parsing as the format grows.

use std::path::{Path, PathBuf};

use forge_types::config::ForgeConfig;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, info, instrument, warn};

/// Minimum valid tier value (inclusive).
pub const TIER_MIN: u8 = 1;

/// Maximum valid tier value (inclusive).
pub const TIER_MAX: u8 = 6;

/// Default tier when none is specified.
pub const DEFAULT_TIER: u8 = 1;

/// File extension recognised by [`ScenarioSuite::load_dir`].
pub const SCENARIO_FILE_EXTENSION: &str = "toml";

/// A single evaluation scenario.
///
/// Scenarios are agent-agnostic — they describe the world, task, and tier
/// only. The agent is supplied separately at evaluation time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scenario {
    /// Stable identifier (used in output paths and scorecard rows).
    pub id: String,
    /// Difficulty tier, in `TIER_MIN..=TIER_MAX`.
    pub tier: u8,
    /// FORGE simulation configuration for this scenario.
    pub forge_config: ForgeConfig,
    /// Optional override for the harness's `max_steps_per_episode`.
    #[serde(default)]
    pub max_steps: Option<u64>,
    /// Optional tags for filtering / grouping.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Optional human-readable description.
    #[serde(default)]
    pub description: Option<String>,
}

impl Scenario {
    /// Builds a scenario from the minimal required fields.
    pub fn new(id: impl Into<String>, tier: u8, forge_config: ForgeConfig) -> Self {
        let id = id.into();
        debug!(%id, tier, "Building scenario");
        Self {
            id,
            tier,
            forge_config,
            max_steps: None,
            tags: Vec::new(),
            description: None,
        }
    }

    /// Returns true if `id` is safe to use as a path component.
    ///
    /// The scenario `id` ends up baked into on-disk artefact paths (via
    /// [`crate::output::OutputConfig::scenario_dir`]), so we reject any
    /// id that could escape the configured output directory.
    pub fn is_safe_path_component(id: &str) -> bool {
        if id.is_empty() {
            return false;
        }
        // Reject parent-directory references, drive letters, NUL, and
        // path separators of either platform. Whitespace-only ids are
        // also rejected to avoid silently-empty directories.
        if id.trim().is_empty() {
            return false;
        }
        for ch in id.chars() {
            // ASCII path separators on Unix/Windows and the NUL byte are
            // never safe regardless of OS.
            if matches!(ch, '/' | '\\' | '\0') {
                return false;
            }
        }
        // Reject components like `..` and `.` even when they only appear
        // by themselves; reject any literal "..", anywhere, since callers
        // may not normalise.
        if id == "." || id == ".." || id.contains("..") {
            return false;
        }
        true
    }

    /// Validates the scenario, returning all issues found.
    ///
    /// An empty `Vec` means the scenario is valid.
    #[instrument(skip_all)]
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.id.is_empty() {
            errors.push("scenario id must not be empty".to_string());
        } else if !Self::is_safe_path_component(&self.id) {
            errors.push(format!(
                "scenario id {:?} contains invalid path characters (must not include '/', '\\\\', NUL, '..', or be a path traversal component)",
                self.id
            ));
        }
        if self.tier < TIER_MIN || self.tier > TIER_MAX {
            errors.push(format!(
                "tier {} is out of valid range {}-{}",
                self.tier, TIER_MIN, TIER_MAX
            ));
        }
        if self.forge_config.world.width == 0 || self.forge_config.world.height == 0 {
            errors.push("world dimensions must be > 0".to_string());
        }
        if self.forge_config.agents.num_agents == 0 {
            errors.push("num_agents must be > 0".to_string());
        }
        if let Some(0) = self.max_steps {
            errors.push("max_steps override must be > 0 when set".to_string());
        }
        errors
    }

    /// Returns true if [`validate`](Self::validate) returns no errors.
    pub fn is_valid(&self) -> bool {
        self.validate().is_empty()
    }

    /// Loads a scenario from a TOML file.
    #[instrument(skip_all, fields(path = %path.as_ref().display()))]
    pub fn load_file(path: impl AsRef<Path>) -> Result<Self, ScenarioError> {
        let path = path.as_ref();
        let raw = std::fs::read_to_string(path).map_err(|e| ScenarioError::Io {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?;
        let scenario: Scenario = toml::from_str(&raw).map_err(|e| ScenarioError::Parse {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?;
        let errors = scenario.validate();
        if !errors.is_empty() {
            return Err(ScenarioError::Invalid {
                path: path.to_path_buf(),
                errors,
            });
        }
        debug!(id = %scenario.id, tier = scenario.tier, "Loaded scenario");
        Ok(scenario)
    }
}

/// A collection of scenarios that share a versioned schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioSuite {
    /// Schema version (informational; matched on by tooling).
    #[serde(default = "default_suite_version")]
    pub version: String,
    /// Ordered list of scenarios.
    pub scenarios: Vec<Scenario>,
}

fn default_suite_version() -> String {
    "1".to_string()
}

impl ScenarioSuite {
    /// Wraps an explicit scenario list into a suite.
    pub fn from_scenarios(scenarios: Vec<Scenario>) -> Self {
        Self {
            version: default_suite_version(),
            scenarios,
        }
    }

    /// Loads every scenario from a directory.
    ///
    /// Reads every `*.toml` file (extension configurable via
    /// [`SCENARIO_FILE_EXTENSION`]) in `dir`, sorted by file name for
    /// deterministic ordering. Sub-directories are *not* traversed.
    #[instrument(skip_all, fields(dir = %dir.as_ref().display()))]
    pub fn load_dir(dir: impl AsRef<Path>) -> Result<Self, ScenarioError> {
        let dir = dir.as_ref();
        if !dir.is_dir() {
            return Err(ScenarioError::NotADirectory {
                path: dir.to_path_buf(),
            });
        }

        let read = std::fs::read_dir(dir).map_err(|e| ScenarioError::Io {
            path: dir.to_path_buf(),
            message: e.to_string(),
        })?;
        // Propagate every per-entry I/O failure instead of silently
        // dropping it via `filter_map(Result::ok)`. A partial suite is
        // worse than a loud failure.
        let mut entries: Vec<PathBuf> = Vec::new();
        for entry in read {
            let entry = entry.map_err(|e| ScenarioError::Io {
                path: dir.to_path_buf(),
                message: format!("read_dir entry failed: {e}"),
            })?;
            let path = entry.path();
            let is_toml = path
                .extension()
                .and_then(|x| x.to_str())
                .map(|x| x.eq_ignore_ascii_case(SCENARIO_FILE_EXTENSION))
                .unwrap_or(false);
            if !is_toml {
                continue;
            }
            // `is_file` can itself fail on broken symlinks or permission
            // issues — surface those rather than silently skip.
            let meta = entry.metadata().map_err(|e| ScenarioError::Io {
                path: path.clone(),
                message: format!("metadata failed: {e}"),
            })?;
            if meta.is_file() {
                entries.push(path);
            }
        }
        entries.sort();

        let mut scenarios = Vec::with_capacity(entries.len());
        for path in &entries {
            scenarios.push(Scenario::load_file(path)?);
        }

        info!(count = scenarios.len(), "Loaded scenario suite");
        let suite = Self::from_scenarios(scenarios);
        let errors = suite.validate();
        if !errors.is_empty() {
            warn!(error_count = errors.len(), "Suite validation failed");
            return Err(ScenarioError::SuiteInvalid { errors });
        }
        Ok(suite)
    }

    /// Validates the suite, returning all issues found.
    #[instrument(skip_all)]
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.scenarios.is_empty() {
            errors.push("suite must contain at least one scenario".to_string());
        }

        let mut seen = std::collections::HashSet::new();
        for s in &self.scenarios {
            if !seen.insert(s.id.clone()) {
                errors.push(format!("duplicate scenario id: {}", s.id));
            }
            for e in s.validate() {
                errors.push(format!("scenario '{}': {}", s.id, e));
            }
        }
        errors
    }

    /// Returns true if [`validate`](Self::validate) returns no errors.
    pub fn is_valid(&self) -> bool {
        self.validate().is_empty()
    }

    /// Returns the unique tiers present in the suite, sorted ascending.
    pub fn tiers(&self) -> Vec<u8> {
        let mut tiers: Vec<u8> = self.scenarios.iter().map(|s| s.tier).collect();
        tiers.sort_unstable();
        tiers.dedup();
        tiers
    }

    /// Returns the scenarios matching the given tier filter.
    ///
    /// Empty filter = all scenarios.
    pub fn filter_tiers(&self, allowed: &[u8]) -> Vec<&Scenario> {
        if allowed.is_empty() {
            return self.scenarios.iter().collect();
        }
        self.scenarios
            .iter()
            .filter(|s| allowed.contains(&s.tier))
            .collect()
    }
}

impl Default for ScenarioSuite {
    fn default() -> Self {
        Self::from_scenarios(Vec::new())
    }
}

/// Errors raised when loading or validating scenarios.
#[derive(Debug, Error)]
pub enum ScenarioError {
    /// Filesystem path is not a directory.
    #[error("not a directory: {path}")]
    NotADirectory {
        /// Offending path.
        path: PathBuf,
    },
    /// I/O failure reading or writing the scenario file.
    #[error("I/O error for {path}: {message}")]
    Io {
        /// Offending path.
        path: PathBuf,
        /// Underlying error message.
        message: String,
    },
    /// TOML parse failure.
    #[error("parse error for {path}: {message}")]
    Parse {
        /// Offending path.
        path: PathBuf,
        /// Underlying error message.
        message: String,
    },
    /// Scenario failed validation.
    #[error("scenario at {path} is invalid: {errors:?}")]
    Invalid {
        /// Offending path.
        path: PathBuf,
        /// List of validation errors.
        errors: Vec<String>,
    },
    /// Suite-level validation failure.
    #[error("suite invalid: {errors:?}")]
    SuiteInvalid {
        /// List of validation errors.
        errors: Vec<String>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn small_config() -> ForgeConfig {
        let mut cfg = ForgeConfig::default();
        cfg.world.width = 8;
        cfg.world.height = 8;
        cfg.agents.num_agents = 1;
        cfg.task.max_episode_length = 32;
        cfg
    }

    #[test]
    fn test_scenario_new_minimal() {
        let s = Scenario::new("alpha", 1, small_config());
        assert_eq!(s.id, "alpha");
        assert_eq!(s.tier, 1);
        assert!(s.tags.is_empty());
        assert!(s.description.is_none());
        assert!(s.max_steps.is_none());
        assert!(s.is_valid());
    }

    #[test]
    fn test_scenario_validate_empty_id() {
        let mut s = Scenario::new("", 1, small_config());
        let errors = s.validate();
        assert!(errors.iter().any(|e| e.contains("id must not be empty")));
        s.id = "ok".into();
        assert!(s.is_valid());
    }

    #[test]
    fn test_scenario_validate_tier_bounds() {
        let s0 = Scenario::new("zero", 0, small_config());
        let s7 = Scenario::new("seven", 7, small_config());
        assert!(!s0.is_valid());
        assert!(!s7.is_valid());
        for tier in TIER_MIN..=TIER_MAX {
            let s = Scenario::new(format!("t{tier}"), tier, small_config());
            assert!(s.is_valid(), "tier {tier} should be valid");
        }
    }

    #[test]
    fn test_scenario_validate_world_dimensions() {
        let mut s = Scenario::new("zero_world", 1, small_config());
        s.forge_config.world.width = 0;
        let errors = s.validate();
        assert!(errors.iter().any(|e| e.contains("world dimensions")));
    }

    #[test]
    fn test_scenario_validate_num_agents_zero() {
        let mut s = Scenario::new("no_agents", 1, small_config());
        s.forge_config.agents.num_agents = 0;
        let errors = s.validate();
        assert!(errors.iter().any(|e| e.contains("num_agents")));
    }

    #[test]
    fn test_scenario_validate_max_steps_zero() {
        let mut s = Scenario::new("zero_steps", 1, small_config());
        s.max_steps = Some(0);
        let errors = s.validate();
        assert!(errors.iter().any(|e| e.contains("max_steps")));
        s.max_steps = Some(10);
        assert!(s.is_valid());
    }

    #[test]
    fn test_scenario_toml_roundtrip() {
        let s = Scenario {
            id: "rt".into(),
            tier: 3,
            forge_config: small_config(),
            max_steps: Some(50),
            tags: vec!["nav".into(), "easy".into()],
            description: Some("round-trip test".into()),
        };
        let toml_str = toml::to_string(&s).unwrap();
        let deser: Scenario = toml::from_str(&toml_str).unwrap();
        assert_eq!(deser.id, "rt");
        assert_eq!(deser.tier, 3);
        assert_eq!(deser.max_steps, Some(50));
        assert_eq!(deser.tags, vec!["nav", "easy"]);
        assert_eq!(deser.description.as_deref(), Some("round-trip test"));
    }

    #[test]
    fn test_scenario_toml_backward_compat_missing_fields() {
        // Older format without max_steps/tags/description should still parse.
        let s = Scenario::new("legacy", 1, small_config());
        let mut value = toml::Value::try_from(&s).unwrap();
        if let Some(t) = value.as_table_mut() {
            t.remove("max_steps");
            t.remove("tags");
            t.remove("description");
        }
        let toml_str = toml::to_string(&value).unwrap();
        let deser: Scenario = toml::from_str(&toml_str).unwrap();
        assert_eq!(deser.id, "legacy");
        assert!(deser.tags.is_empty());
        assert!(deser.description.is_none());
        assert!(deser.max_steps.is_none());
    }

    #[test]
    fn test_scenario_load_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("load.toml");
        let s = Scenario::new("from_disk", 2, small_config());
        std::fs::write(&path, toml::to_string(&s).unwrap()).unwrap();
        let loaded = Scenario::load_file(&path).unwrap();
        assert_eq!(loaded.id, "from_disk");
        assert_eq!(loaded.tier, 2);
    }

    #[test]
    fn test_scenario_load_file_missing() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("absent.toml");
        let err = Scenario::load_file(&path).unwrap_err();
        assert!(matches!(err, ScenarioError::Io { .. }));
    }

    #[test]
    fn test_scenario_load_file_bad_toml() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bad.toml");
        std::fs::write(&path, "= not = valid =").unwrap();
        let err = Scenario::load_file(&path).unwrap_err();
        assert!(matches!(err, ScenarioError::Parse { .. }));
    }

    #[test]
    fn test_scenario_load_file_invalid_content() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("invalid.toml");
        let mut s = Scenario::new("bad", 99, small_config());
        s.id = String::new();
        std::fs::write(&path, toml::to_string(&s).unwrap()).unwrap();
        let err = Scenario::load_file(&path).unwrap_err();
        assert!(matches!(err, ScenarioError::Invalid { .. }));
    }

    #[test]
    fn test_suite_from_scenarios_roundtrip() {
        let suite = ScenarioSuite::from_scenarios(vec![
            Scenario::new("a", 1, small_config()),
            Scenario::new("b", 2, small_config()),
        ]);
        assert_eq!(suite.scenarios.len(), 2);
        assert_eq!(suite.tiers(), vec![1, 2]);
    }

    #[test]
    fn test_suite_validate_empty() {
        let suite = ScenarioSuite::default();
        let errors = suite.validate();
        assert!(errors.iter().any(|e| e.contains("at least one")));
    }

    #[test]
    fn test_suite_validate_duplicate_id() {
        let suite = ScenarioSuite::from_scenarios(vec![
            Scenario::new("dup", 1, small_config()),
            Scenario::new("dup", 2, small_config()),
        ]);
        let errors = suite.validate();
        assert!(errors.iter().any(|e| e.contains("duplicate")));
    }

    #[test]
    fn test_suite_validate_bubbles_scenario_errors() {
        let bad = Scenario::new("x", 99, small_config()); // invalid tier
        let suite = ScenarioSuite::from_scenarios(vec![bad]);
        let errors = suite.validate();
        assert!(errors.iter().any(|e| e.contains("scenario 'x'")));
    }

    #[test]
    fn test_suite_filter_tiers() {
        let suite = ScenarioSuite::from_scenarios(vec![
            Scenario::new("a", 1, small_config()),
            Scenario::new("b", 2, small_config()),
            Scenario::new("c", 1, small_config()),
        ]);
        let only_t1 = suite.filter_tiers(&[1]);
        assert_eq!(only_t1.len(), 2);
        let all = suite.filter_tiers(&[]);
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_suite_load_dir_sorted_deterministic() {
        let dir = tempdir().unwrap();
        for (idx, name) in ["zebra.toml", "alpha.toml", "mike.toml"].iter().enumerate() {
            let path = dir.path().join(name);
            let s = Scenario::new(format!("s{idx}"), 1, small_config());
            std::fs::write(&path, toml::to_string(&s).unwrap()).unwrap();
        }
        let suite = ScenarioSuite::load_dir(dir.path()).unwrap();
        assert_eq!(suite.scenarios.len(), 3);
        // The files were sorted alpha,mike,zebra — match by id ordering
        assert_eq!(suite.scenarios[0].id, "s1"); // alpha.toml
        assert_eq!(suite.scenarios[1].id, "s2"); // mike.toml
        assert_eq!(suite.scenarios[2].id, "s0"); // zebra.toml
    }

    #[test]
    fn test_suite_load_dir_ignores_non_toml() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("ok.toml");
        let s = Scenario::new("ok", 1, small_config());
        std::fs::write(&p, toml::to_string(&s).unwrap()).unwrap();
        std::fs::write(dir.path().join("README.md"), "not a scenario").unwrap();
        std::fs::write(dir.path().join("data.json"), "{}").unwrap();
        let suite = ScenarioSuite::load_dir(dir.path()).unwrap();
        assert_eq!(suite.scenarios.len(), 1);
    }

    #[test]
    fn test_suite_load_dir_not_a_directory() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("file.toml");
        std::fs::write(&p, "").unwrap();
        let err = ScenarioSuite::load_dir(&p).unwrap_err();
        assert!(matches!(err, ScenarioError::NotADirectory { .. }));
    }

    #[test]
    fn test_suite_load_dir_empty_returns_invalid() {
        let dir = tempdir().unwrap();
        let err = ScenarioSuite::load_dir(dir.path()).unwrap_err();
        assert!(matches!(err, ScenarioError::SuiteInvalid { .. }));
    }

    #[test]
    fn test_suite_load_dir_propagates_invalid_scenario() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bad.toml");
        let s = Scenario::new("", 1, small_config()); // empty id
        std::fs::write(&path, toml::to_string(&s).unwrap()).unwrap();
        let err = ScenarioSuite::load_dir(dir.path()).unwrap_err();
        assert!(matches!(err, ScenarioError::Invalid { .. }));
    }

    #[test]
    fn test_suite_tiers_dedup_sorted() {
        let suite = ScenarioSuite::from_scenarios(vec![
            Scenario::new("a", 3, small_config()),
            Scenario::new("b", 1, small_config()),
            Scenario::new("c", 3, small_config()),
            Scenario::new("d", 2, small_config()),
        ]);
        assert_eq!(suite.tiers(), vec![1, 2, 3]);
    }

    #[test]
    fn test_constants_are_consistent() {
        const _: () = {
            assert!(TIER_MIN <= DEFAULT_TIER);
            assert!(DEFAULT_TIER <= TIER_MAX);
        };
        assert_eq!(SCENARIO_FILE_EXTENSION, "toml");
    }

    // ──────────────────────────────────────────────────────────────────
    // Path-traversal hardening (review thread r3252771992).
    // ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_is_safe_path_component_accepts_normal_ids() {
        for id in [
            "alpha",
            "alpha_beta",
            "scenario-1",
            "tier3_easy",
            "v1.2.3",
            "a b c", // spaces are OK; only path separators are rejected
        ] {
            assert!(
                Scenario::is_safe_path_component(id),
                "expected {id:?} to be a safe path component"
            );
        }
    }

    #[test]
    fn test_is_safe_path_component_rejects_traversal_chars() {
        for id in [
            "",
            "   ",
            ".",
            "..",
            "../escape",
            "..\\escape",
            "a/b",
            "a\\b",
            "foo/../bar",
            "with\0null",
            "trail..ing",
        ] {
            assert!(
                !Scenario::is_safe_path_component(id),
                "expected {id:?} to be REJECTED as a path component"
            );
        }
    }

    #[test]
    fn test_scenario_validate_rejects_path_traversal_id() {
        let mut s = Scenario::new("ok", 1, small_config());
        s.id = "../escape".to_string();
        let errors = s.validate();
        assert!(
            errors.iter().any(|e| e.contains("invalid path characters")),
            "missing path-traversal error in {errors:?}"
        );
    }

    #[test]
    fn test_scenario_load_file_rejects_path_traversal_id() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bad.toml");
        let mut s = Scenario::new("ok", 1, small_config());
        s.id = "../escape".to_string();
        std::fs::write(&path, toml::to_string(&s).unwrap()).unwrap();
        let err = Scenario::load_file(&path).unwrap_err();
        match err {
            ScenarioError::Invalid { errors, .. } => {
                assert!(errors.iter().any(|e| e.contains("invalid path characters")));
            }
            other => panic!("expected Invalid, got {other:?}"),
        }
    }

    // ──────────────────────────────────────────────────────────────────
    // load_dir error propagation (review thread r3252813091).
    //
    // Dangling symlinks don't reliably trigger the new metadata() error
    // path because `DirEntry::metadata()` returns the symlink's own
    // metadata, not the target's. We assert the *non-error* invariant
    // instead: a directory with no `.toml` entries surfaces as
    // SuiteInvalid (not silently empty).
    // ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_load_dir_with_only_symlinks_does_not_panic() {
        // Regression for the change away from `filter_map(Result::ok)`:
        // ensure the new entry-by-entry loop handles odd entry kinds
        // (no `.toml` files at all → SuiteInvalid).
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("not-a-scenario.md"), "ignore me").unwrap();
        let err = ScenarioSuite::load_dir(dir.path()).unwrap_err();
        assert!(matches!(err, ScenarioError::SuiteInvalid { .. }));
    }

    mod prop {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn scenario_validate_tier_invariant(tier in 0u8..=10u8) {
                let s = Scenario::new("p", tier, small_config());
                let valid = s.is_valid();
                prop_assert_eq!(valid, (TIER_MIN..=TIER_MAX).contains(&tier));
            }

            #[test]
            fn scenario_toml_roundtrip_arbitrary(
                tier in TIER_MIN..=TIER_MAX,
                id in "[a-z][a-z0-9_]{0,15}",
                num_tags in 0usize..5,
            ) {
                let tags: Vec<String> = (0..num_tags).map(|i| format!("tag{i}")).collect();
                let s = Scenario {
                    id,
                    tier,
                    forge_config: small_config(),
                    max_steps: Some(100),
                    tags: tags.clone(),
                    description: None,
                };
                let toml_str = toml::to_string(&s).unwrap();
                let deser: Scenario = toml::from_str(&toml_str).unwrap();
                prop_assert_eq!(deser.tier, tier);
                prop_assert_eq!(deser.tags, tags);
            }

            #[test]
            fn suite_filter_tiers_subset(
                t1 in 1u8..=3,
                t2 in 4u8..=6,
            ) {
                let suite = ScenarioSuite::from_scenarios(vec![
                    Scenario::new("a", t1, small_config()),
                    Scenario::new("b", t2, small_config()),
                ]);
                let filtered = suite.filter_tiers(&[t1]);
                prop_assert!(filtered.iter().all(|s| s.tier == t1));
            }
        }
    }
}
