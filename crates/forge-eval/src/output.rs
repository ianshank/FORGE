//! Output configuration and on-disk artefact writers for the harness.
//!
//! Phase 1 wires up the four classes of artefact the harness can emit:
//! - Compact replays (`*.bin`, bincode)
//! - Full trajectories (`*.jsonl`)
//! - Per-episode trajectory metadata (`*.metadata.json`)
//! - The aggregate scorecard (`scorecard.json` and/or `scorecard.md`)
//!
//! All path components and behaviour are configurable through [`OutputConfig`]
//! — nothing is hard-coded into the harness.

use std::path::{Path, PathBuf};

use forge_replay::compact::CompactReplay;
use forge_replay::export::export_to_jsonl;
use forge_replay::trajectory::Trajectory;
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument, warn};

/// Default sub-directory name used when no explicit output dir is configured.
pub const DEFAULT_OUTPUT_DIR: &str = "forge-eval-output";

/// Default scorecard file basename (without extension).
pub const DEFAULT_SCORECARD_BASENAME: &str = "scorecard";

/// Sub-directory under the output root that holds per-episode replays.
pub const REPLAY_SUBDIR: &str = "replays";

/// Sub-directory under the output root that holds per-episode trajectories.
pub const TRAJECTORY_SUBDIR: &str = "trajectories";

/// Format choices for the aggregate scorecard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScorecardFormat {
    /// Emit `scorecard.json` only.
    Json,
    /// Emit `scorecard.md` only.
    Markdown,
    /// Emit both JSON and Markdown.
    #[default]
    Both,
}

/// Configuration for harness output artefacts.
///
/// Set [`enabled`](Self::enabled) to `false` (the default) to opt out of all
/// on-disk writes — this keeps backward compatibility with callers that were
/// using the original in-memory-only harness.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OutputConfig {
    /// Master toggle. Defaults to `false`.
    pub enabled: bool,
    /// Root directory for outputs.
    pub dir: PathBuf,
    /// Whether to write compact replays per episode.
    pub write_replays: bool,
    /// Whether to write full trajectories per episode.
    pub write_trajectories: bool,
    /// Whether to write the aggregate scorecard.
    pub write_scorecard: bool,
    /// Format of the aggregate scorecard.
    pub scorecard_format: ScorecardFormat,
    /// Basename (without extension) for the scorecard file.
    pub scorecard_basename: String,
    /// Whether to create the output directory if it does not exist.
    pub create_dir: bool,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            dir: PathBuf::from(DEFAULT_OUTPUT_DIR),
            write_replays: true,
            write_trajectories: true,
            write_scorecard: true,
            scorecard_format: ScorecardFormat::default(),
            scorecard_basename: DEFAULT_SCORECARD_BASENAME.to_string(),
            create_dir: true,
        }
    }
}

impl OutputConfig {
    /// Validates the configuration, returning a list of issues.
    #[instrument(skip_all)]
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.enabled {
            if self.dir.as_os_str().is_empty() {
                errors.push("output dir must not be empty when enabled".to_string());
            }
            if self.scorecard_basename.is_empty() {
                errors.push("scorecard_basename must not be empty".to_string());
            }
            if !self.write_replays && !self.write_trajectories && !self.write_scorecard {
                errors.push("output enabled but no artefact types are toggled on".to_string());
            }
        }
        errors
    }

    /// Returns true if [`validate`](Self::validate) returns no errors.
    pub fn is_valid(&self) -> bool {
        self.validate().is_empty()
    }

    /// Returns the per-scenario output directory under the configured root.
    pub fn scenario_dir(&self, scenario_id: &str) -> PathBuf {
        self.dir.join(scenario_id)
    }

    /// Returns the path used to store a compact replay for a given episode.
    pub fn replay_path(&self, scenario_id: &str, seed: u64) -> PathBuf {
        self.scenario_dir(scenario_id)
            .join(REPLAY_SUBDIR)
            .join(format!("seed_{seed:020}.bin"))
    }

    /// Returns the path used to store a trajectory for a given episode.
    pub fn trajectory_path(&self, scenario_id: &str, seed: u64) -> PathBuf {
        self.scenario_dir(scenario_id)
            .join(TRAJECTORY_SUBDIR)
            .join(format!("seed_{seed:020}.jsonl"))
    }

    /// Returns the path used to store trajectory metadata for an episode.
    pub fn trajectory_metadata_path(&self, scenario_id: &str, seed: u64) -> PathBuf {
        self.scenario_dir(scenario_id)
            .join(TRAJECTORY_SUBDIR)
            .join(format!("seed_{seed:020}.metadata.json"))
    }

    /// Returns the scorecard path for the configured format and extension.
    pub fn scorecard_path(&self, ext: &str) -> PathBuf {
        self.dir
            .join(format!("{}.{}", self.scorecard_basename, ext))
    }

    /// Ensures `path`'s parent directory exists when [`create_dir`](Self::create_dir)
    /// is enabled. Returns the OS-level error message on failure.
    pub fn ensure_parent(&self, path: &Path) -> Result<(), String> {
        if !self.create_dir {
            return Ok(());
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
        }
        Ok(())
    }
}

/// Writes a compact replay for one episode.
#[instrument(skip_all, fields(scenario_id, seed))]
pub fn write_replay(
    cfg: &OutputConfig,
    scenario_id: &str,
    seed: u64,
    replay: &CompactReplay,
) -> Result<PathBuf, String> {
    let path = cfg.replay_path(scenario_id, seed);
    cfg.ensure_parent(&path)?;
    let bytes = replay.to_bytes()?;
    std::fs::write(&path, &bytes).map_err(|e| {
        warn!(error = %e, path = %path.display(), "Failed to write replay");
        format!("failed to write replay {}: {e}", path.display())
    })?;
    debug!(path = %path.display(), bytes = bytes.len(), "Wrote replay");
    Ok(path)
}

/// Writes a trajectory (JSONL + metadata sidecar) for one episode.
#[instrument(skip_all, fields(scenario_id, seed))]
pub fn write_trajectory(
    cfg: &OutputConfig,
    scenario_id: &str,
    seed: u64,
    trajectory: &Trajectory,
) -> Result<PathBuf, String> {
    let path = cfg.trajectory_path(scenario_id, seed);
    cfg.ensure_parent(&path)?;
    export_to_jsonl(trajectory, &path)?;

    let meta_path = cfg.trajectory_metadata_path(scenario_id, seed);
    let meta_json = serde_json::to_string_pretty(&trajectory.metadata)
        .map_err(|e| format!("metadata serialization failed: {e}"))?;
    std::fs::write(&meta_path, meta_json)
        .map_err(|e| format!("failed to write metadata {}: {e}", meta_path.display()))?;

    debug!(path = %path.display(), "Wrote trajectory");
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_replay::trajectory::TrajectoryBuilder;
    use forge_types::agent_interface::AgentResponse;
    use forge_types::config::ForgeConfig;
    use forge_types::constants;
    use forge_types::observation::{InventoryObservation, Observation, TileObservation};
    use tempfile::tempdir;

    fn make_obs() -> Observation {
        Observation {
            grid_view: vec![TileObservation::default()],
            view_width: 1,
            view_height: 1,
            inventory: InventoryObservation {
                slots: vec![(constants::OBS_EMPTY_SLOT_ITEM, 0)],
            },
            health: 1.0,
            stamina: 1.0,
            position: (0, 0),
            messages: vec![],
            day_phase: 0,
            task_progress: vec![],
            altitude: 0,
            battery: 1.0,
            morphology: 0,
            heading: 0,
            crop_scan_results: vec![],
            soil_readings: vec![],
            disease_detections: 0,
            report_ready: false,
        }
    }

    fn make_traj() -> Trajectory {
        let mut b = TrajectoryBuilder::new();
        let r = vec![AgentResponse::from_action(0)];
        b.record_step(0, vec![make_obs()], &r, vec![0.5], false, false);
        b.seed(7).build(vec![0.5])
    }

    fn make_replay() -> CompactReplay {
        let cfg = ForgeConfig::default();
        let mut b = CompactReplay::builder(cfg, 7);
        b.record_tick(vec![0]);
        b.build()
    }

    #[test]
    fn test_default_output_disabled() {
        let cfg = OutputConfig::default();
        assert!(!cfg.enabled);
        assert!(cfg.is_valid()); // disabled is always valid
    }

    #[test]
    fn test_output_validate_enabled_requires_dir() {
        let cfg = OutputConfig {
            enabled: true,
            dir: PathBuf::new(),
            ..Default::default()
        };
        assert!(!cfg.is_valid());
    }

    #[test]
    fn test_output_validate_enabled_requires_some_artefact() {
        let cfg = OutputConfig {
            enabled: true,
            write_replays: false,
            write_trajectories: false,
            write_scorecard: false,
            ..Default::default()
        };
        let errors = cfg.validate();
        assert!(errors.iter().any(|e| e.contains("no artefact")));
    }

    #[test]
    fn test_output_validate_empty_basename() {
        let cfg = OutputConfig {
            enabled: true,
            scorecard_basename: String::new(),
            ..Default::default()
        };
        assert!(!cfg.is_valid());
    }

    #[test]
    fn test_paths_use_configured_root_and_subdirs() {
        let cfg = OutputConfig {
            dir: PathBuf::from("/tmp/forge-out"),
            ..Default::default()
        };
        let rp = cfg.replay_path("alpha", 42);
        let tp = cfg.trajectory_path("alpha", 42);
        let mp = cfg.trajectory_metadata_path("alpha", 42);

        assert!(rp.starts_with("/tmp/forge-out/alpha"));
        assert!(rp.to_string_lossy().contains(REPLAY_SUBDIR));
        assert!(tp.to_string_lossy().contains(TRAJECTORY_SUBDIR));
        assert!(mp.to_string_lossy().ends_with(".metadata.json"));
    }

    #[test]
    fn test_scorecard_path_uses_basename() {
        let cfg = OutputConfig {
            dir: PathBuf::from("/tmp/forge-out"),
            scorecard_basename: "my-card".into(),
            ..Default::default()
        };
        assert_eq!(
            cfg.scorecard_path("json"),
            PathBuf::from("/tmp/forge-out/my-card.json")
        );
        assert_eq!(
            cfg.scorecard_path("md"),
            PathBuf::from("/tmp/forge-out/my-card.md")
        );
    }

    #[test]
    fn test_write_replay_roundtrip() {
        let dir = tempdir().unwrap();
        let cfg = OutputConfig {
            enabled: true,
            dir: dir.path().to_path_buf(),
            ..Default::default()
        };
        let replay = make_replay();
        let path = write_replay(&cfg, "scn", 7, &replay).unwrap();
        assert!(path.exists());
        let bytes = std::fs::read(&path).unwrap();
        let rt = CompactReplay::from_bytes(&bytes).unwrap();
        assert_eq!(rt.seed, 7);
    }

    #[test]
    fn test_write_trajectory_emits_jsonl_and_metadata() {
        let dir = tempdir().unwrap();
        let cfg = OutputConfig {
            enabled: true,
            dir: dir.path().to_path_buf(),
            ..Default::default()
        };
        let traj = make_traj();
        let path = write_trajectory(&cfg, "scn", 7, &traj).unwrap();
        let meta_path = cfg.trajectory_metadata_path("scn", 7);
        assert!(path.exists());
        assert!(meta_path.exists());
        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&meta_path).unwrap()).unwrap();
        assert_eq!(meta["seed"], 7);
    }

    #[test]
    fn test_ensure_parent_respects_create_dir_flag() {
        let dir = tempdir().unwrap();
        let nested = dir.path().join("a").join("b").join("c.bin");
        let cfg = OutputConfig {
            enabled: true,
            dir: dir.path().to_path_buf(),
            create_dir: true,
            ..Default::default()
        };
        cfg.ensure_parent(&nested).unwrap();
        assert!(nested.parent().unwrap().exists());

        // With create_dir=false, ensure_parent should be a no-op.
        let cfg2 = OutputConfig {
            create_dir: false,
            ..cfg.clone()
        };
        let nested2 = dir.path().join("x").join("y").join("z.bin");
        cfg2.ensure_parent(&nested2).unwrap();
        assert!(!nested2.parent().unwrap().exists());
    }

    #[test]
    fn test_scorecard_format_serde() {
        for variant in [
            ScorecardFormat::Json,
            ScorecardFormat::Markdown,
            ScorecardFormat::Both,
        ] {
            let s = serde_json::to_string(&variant).unwrap();
            let back: ScorecardFormat = serde_json::from_str(&s).unwrap();
            assert_eq!(back, variant);
        }
    }

    #[test]
    fn test_output_config_serde_roundtrip() {
        let cfg = OutputConfig {
            enabled: true,
            dir: PathBuf::from("./out"),
            write_replays: false,
            write_trajectories: true,
            write_scorecard: true,
            scorecard_format: ScorecardFormat::Json,
            scorecard_basename: "result".into(),
            create_dir: false,
        };
        let toml_str = toml::to_string(&cfg).unwrap();
        let deser: OutputConfig = toml::from_str(&toml_str).unwrap();
        assert!(deser.enabled);
        assert!(!deser.write_replays);
        assert_eq!(deser.scorecard_format, ScorecardFormat::Json);
        assert_eq!(deser.scorecard_basename, "result");
    }

    #[test]
    fn test_output_config_default_serde_includes_disabled() {
        let cfg = OutputConfig::default();
        let toml_str = toml::to_string(&cfg).unwrap();
        let deser: OutputConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(cfg.enabled, deser.enabled);
        assert_eq!(cfg.write_replays, deser.write_replays);
    }
}
