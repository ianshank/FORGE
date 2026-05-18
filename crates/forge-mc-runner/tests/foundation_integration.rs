//! Integration test for the runner foundation.
//!
//! Proves the four foundation modules — `RunnerConfig`, `ModelManifest`,
//! `HotReloadWatcher`, `TrajectoryWriter` — compose correctly under the
//! lifecycle the eventual `Runner<E, M>` loop will drive:
//!
//! 1. Load config from TOML.
//! 2. Create writer + watcher from config.
//! 3. Trainer writes manifest @ v1 → watcher emits event.
//! 4. Episode 1: writer records steps → finalize → file on disk.
//! 5. Between episodes: watcher polls again, no event (same version).
//! 6. Trainer bumps to v2 → watcher emits event with `previous=1`.
//! 7. Episode 2: writer records steps under same writer instance.
//! 8. No `.tmp` siblings left on disk.

use forge_mc_runner::{
    HotReloadWatcher, ModelManifest, ModelManifestFiles, RunnerConfig, TrajectoryWriter,
};
use forge_replay::v2::{StepV2, TrajectoryV2};

fn make_manifest_at(path: &std::path::Path, version: u64, schema_id: &str) {
    use forge_mc_runner::manifest::ModelFileEntry;
    let m = ModelManifest::new(
        version,
        schema_id,
        "2026-05-17T00:00:00Z",
        ModelManifestFiles {
            representation: ModelFileEntry {
                path: "r.onnx".into(),
                sha256: "r".repeat(64),
            },
            dynamics: ModelFileEntry {
                path: "d.onnx".into(),
                sha256: "d".repeat(64),
            },
            prediction: ModelFileEntry {
                path: "p.onnx".into(),
                sha256: "p".repeat(64),
            },
        },
    );
    m.save_json(path).unwrap();
}

fn make_step(tick: u64, obs_dim: usize, action_count: u32, action: u32) -> StepV2 {
    StepV2 {
        tick,
        obs: vec![0.0; obs_dim],
        action_id: action,
        policy_target: vec![1.0 / action_count as f32; action_count as usize],
        value_target: 0.0,
        reward: 0.1,
        terminated: false,
        truncated: false,
    }
}

/// Drive the four foundation modules through two episodes plus a
/// between-episode manifest bump. Asserts every invariant the eventual
/// `Runner` will rely on.
#[test]
fn config_writer_watcher_compose_through_two_episodes() {
    let tmp = tempfile::tempdir().unwrap();

    // 1. Build a TOML config and parse it. This is what `forge-mc-runner`'s
    //    eventual main() will do — proves serde wiring works end-to-end.
    let traj_dir = tmp.path().join("trajectories");
    let manifest_path = tmp.path().join("model_manifest.json");
    let toml_src = format!(
        r#"
            episodes = 2
            max_steps_per_episode = 4
            trajectory_dir = "{}"
            manifest_path = "{}"
            env_id = "minecraft"
            schema_id = "abc-pinned"
            planning_sims = 0
            action_repeat = 1
            metrics_port = 0
        "#,
        traj_dir.display().to_string().replace('\\', "\\\\"),
        manifest_path.display().to_string().replace('\\', "\\\\"),
    );
    let cfg: RunnerConfig = toml::from_str(&toml_src).expect("config should parse");
    cfg.validate().expect("config should validate");
    assert_eq!(cfg.episodes, 2);
    assert!(
        cfg.metrics_disabled(),
        "metrics_port=0 must disable metrics"
    );

    // 2. Construct writer + watcher straight from config fields. Both
    //    are lazy — neither touches disk yet.
    let obs_dim = 4usize;
    let action_count = 3u32;
    let mut writer = TrajectoryWriter::new(
        cfg.trajectory_dir.clone(),
        cfg.env_id.clone(),
        cfg.schema_id.clone(),
        obs_dim,
        action_count,
    );
    let mut watcher = HotReloadWatcher::new(cfg.manifest_path.clone());
    assert!(!writer.directory().exists(), "writer dir lazy");
    assert!(watcher.last_seen_version().is_none(), "watcher cold");

    // 3. Trainer writes v1 manifest. Watcher must emit on next poll.
    make_manifest_at(&manifest_path, 1, &cfg.schema_id);
    let ev = watcher.poll().unwrap().expect("watcher must emit on v1");
    assert_eq!(ev.new_version, 1);
    assert!(ev.previous_version.is_none());
    assert_eq!(
        ev.manifest.schema_id, cfg.schema_id,
        "watcher must surface the schema_id for the runner to cross-check"
    );

    // 4. Episode 1: run a short episode through the writer.
    writer
        .start_episode("ep-001", cfg.base_seed, "2026-05-17T00:00:01Z")
        .unwrap();
    for tick in 0..cfg.max_steps_per_episode {
        writer
            .record_step(make_step(tick, obs_dim, action_count, (tick % 3) as u32))
            .unwrap();
    }
    let ep1_path = writer.finalize_and_save("2026-05-17T00:00:02Z").unwrap();
    assert!(ep1_path.exists());
    assert!(ep1_path.ends_with("ep-001.json"));
    assert!(!writer.has_episode_in_progress());

    // 5. Between-episode poll with same manifest version → no event.
    assert!(
        watcher.poll().unwrap().is_none(),
        "no event when version unchanged"
    );

    // 6. Trainer bumps to v2. Watcher emits with previous=1.
    make_manifest_at(&manifest_path, 2, &cfg.schema_id);
    let ev2 = watcher.poll().unwrap().expect("watcher must emit on v2");
    assert_eq!(ev2.new_version, 2);
    assert_eq!(ev2.previous_version, Some(1));

    // 7. Episode 2 — same writer instance, fresh episode id.
    writer
        .start_episode("ep-002", Some(7), "2026-05-17T00:00:03Z")
        .unwrap();
    writer
        .record_step(make_step(0, obs_dim, action_count, 0))
        .unwrap();
    let ep2_path = writer.finalize_and_save("2026-05-17T00:00:04Z").unwrap();
    assert!(ep2_path.exists());
    assert!(ep2_path.ends_with("ep-002.json"));

    // 8. No tmp siblings. Trajectories save atomically.
    let tmp_traj = traj_dir.join(".ep-001.json.tmp");
    let tmp_manifest = tmp.path().join(".model_manifest.json.tmp");
    assert!(!tmp_traj.exists(), "trajectory .tmp left behind");
    assert!(!tmp_manifest.exists(), "manifest .tmp left behind");

    // 9. Cross-check trajectories are round-trippable through
    //    `TrajectoryV2::load_json` — proves the writer's output is
    //    consumable by `forge-replay`-aware trainers.
    let ep1 = TrajectoryV2::load_json(&ep1_path).unwrap();
    let ep2 = TrajectoryV2::load_json(&ep2_path).unwrap();
    assert_eq!(ep1.episode_id, "ep-001");
    assert_eq!(ep1.steps.len(), cfg.max_steps_per_episode as usize);
    assert_eq!(ep2.episode_id, "ep-002");
    assert_eq!(ep2.steps.len(), 1);
    assert_eq!(ep1.schema_id, cfg.schema_id);
    assert_eq!(ep2.schema_id, cfg.schema_id);
}

/// Sanity-check: a runner whose config disagrees with the manifest's
/// schema_id can detect drift between the two without any custom
/// glue — `manifest.schema_id` and `config.schema_id` are simple `&str`
/// comparisons. This is what the eventual `Runner` will do at startup.
#[test]
fn config_and_manifest_schema_id_can_be_compared_directly() {
    let tmp = tempfile::tempdir().unwrap();
    let manifest_path = tmp.path().join("model_manifest.json");
    make_manifest_at(&manifest_path, 1, "manifest-schema");

    let cfg = RunnerConfig {
        schema_id: "config-schema".into(),
        manifest_path: manifest_path.clone(),
        ..RunnerConfig::default()
    };

    let mut watcher = HotReloadWatcher::new(&manifest_path);
    let ev = watcher.poll().unwrap().expect("event");
    assert_ne!(
        cfg.schema_id, ev.manifest.schema_id,
        "this is the drift the runner must catch and refuse to start"
    );
}
