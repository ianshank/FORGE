//! Live runner wiring — connects a real
//! [`forge_env_mc::MinecraftEnv`] to a real
//! [`forge_agent::OnnxMuZeroModel`] and drives the runner loop. Closes
//! the v0.3-pre `ExitCode 64` gap.
//!
//! Gated behind the `mc-live` Cargo feature so the runner library
//! stays buildable on hosts without ONNX Runtime / `forge-env-mc`.
//! The binary at `crates/forge-mc-runner/src/main.rs` consumes this
//! module from its `async_main` else-branch via `tokio::task::
//! spawn_blocking` (mirroring the `--dry-run` shape — both env
//! `connect()` and ONNX `load()` are synchronous blocking calls).
//!
//! ## Shape
//!
//! ```text
//! main.rs::async_main
//!   ├── tokio::task::spawn_blocking(move || run_live(cfg, metrics))
//!   │   └── live::run_live
//!   │       ├── MinecraftEnvConfig::from_toml(cfg.mc_env_config_path)
//!   │       ├── ActionMap::load(env_cfg.action_map_path)
//!   │       ├── (xlang cross-check schema_id vs cfg.schema_id)
//!   │       ├── MinecraftEnv::connect(env_cfg, action_map)
//!   │       ├── ModelManifest::load_json(cfg.manifest_path)
//!   │       ├── OnnxModelConfig := config_from_manifest(...)
//!   │       ├── OnnxMuZeroModel::load(onnx_cfg)
//!   │       ├── LatentMctsSearch::new(model, mcts_cfg)
//!   │       ├── TrajectoryWriter::new(...).with_compression(...)
//!   │       ├── HotReloadWatcher::new(cfg.manifest_path)
//!   │       ├── into_reload_fn(...) wrapped to also call
//!   │       │     recorder.set_model_version(manifest.version)
//!   │       ├── Runner::new(...).with_metrics(rec).with_reload_fn(fn)
//!   │       └── runner.run(None)
//!   └── (back to main; metrics tear-down via shutdown oneshot)
//! ```

#![cfg(feature = "mc-live")]

#[cfg(feature = "onnx-reload")]
use std::path::PathBuf;

#[cfg(feature = "onnx-reload")]
use forge_agent::latent_mcts::onnx_model::{OnnxMuZeroModel, OnnxReloadError};
use forge_agent::latent_mcts::search::{LatentMctsConfig, LatentMctsSearch};
use forge_env::Env;
use forge_env_mc::{ActionMap, MinecraftEnv, MinecraftEnvConfig};
use tracing::info;
#[cfg(feature = "onnx-reload")]
use tracing::warn;

#[cfg(feature = "onnx-reload")]
use crate::config::OnnxRuntimeConfig;
use crate::config::RunnerConfig;
use crate::error::RunnerError;
use crate::hot_reload::HotReloadWatcher;
#[cfg(feature = "onnx-reload")]
use crate::manifest::ModelManifest;
use crate::metrics::MetricsRecorder;
#[cfg(feature = "onnx-reload")]
use crate::onnx_reload::config_from_manifest;
use crate::random_baseline::RandomLatentModel;
#[cfg(feature = "onnx-reload")]
use crate::runner::ReloadFn;
use crate::runner::Runner;
use crate::trajectory::TrajectoryWriter;

/// Drive a single live session: connect to the bot, load the model
/// bundle, run `cfg.episodes` episodes. Returns when the runner
/// reaches its episode budget or hits a fatal error.
///
/// Sync function — call from `tokio::task::spawn_blocking` so the
/// blocking WS handshake doesn't stall the async runtime.
pub fn run_live(cfg: RunnerConfig, metrics: Option<MetricsRecorder>) -> Result<(), RunnerError> {
    // 1. Load MinecraftEnvConfig.
    let env_cfg_path = cfg.mc_env_config_path.as_ref().ok_or_else(|| {
        RunnerError::ConfigLoad(
            "mc_env_config_path is required for live runs; \
             set the field in runner.toml or pass `--mc-config <path>`"
                .into(),
        )
    })?;
    let env_cfg_text = std::fs::read_to_string(env_cfg_path)
        .map_err(|e| RunnerError::ConfigLoad(format!("read {}: {e}", env_cfg_path.display())))?;
    let env_cfg: MinecraftEnvConfig = toml::from_str(&env_cfg_text)
        .map_err(|e| RunnerError::ConfigLoad(format!("parse {}: {e}", env_cfg_path.display())))?;

    // 2. Load ActionMap (path inside env_cfg).
    let action_map = ActionMap::load(&env_cfg.action_map_path).map_err(|e| {
        RunnerError::ConfigLoad(format!(
            "load action map {}: {e}",
            env_cfg.action_map_path.display()
        ))
    })?;
    let action_count = action_map.action_count();
    info!(
        action_count,
        path = %env_cfg.action_map_path.display(),
        "action map loaded"
    );

    // 3. Connect to the bot. The handshake cross-checks
    // `expected_schema_id` against the bot's `Hello.schema_id` when
    // set in env_cfg; cfg.schema_id (with env-var override applied by
    // main.rs) is the runner's local truth.
    let env = MinecraftEnv::connect(env_cfg, action_map)
        .map_err(|e| RunnerError::EnvSetup(format!("MinecraftEnv::connect failed: {e}")))?;
    // Belt-and-braces: drift between the bot's hash and the runner's
    // expected hash MUST stop the loop. The env's `connect` already
    // does this check when `expected_schema_id` is set on the env
    // config, but the runner config carries a SEPARATE `schema_id`
    // field (operators may set it without populating the env one).
    if !cfg.schema_id.is_empty() && cfg.schema_id != env.schema_id() && cfg.schema_id != "unset" {
        return Err(RunnerError::EnvSetup(format!(
            "schema_id drift: runner.schema_id={} != bot.schema_id={}",
            cfg.schema_id,
            env.schema_id()
        )));
    }

    // 4. Branch on random_actions BEFORE the ONNX bundle load. In
    // random-baseline mode the runner never invokes the model, so we
    // skip the manifest + bundle + ORT session entirely and run the
    // loop with a `RandomLatentModel` stub for the type-generic
    // bound.  This lets the v0.5 baseline-capture script run against
    // a host without ONNX Runtime installed.
    if cfg.random_actions {
        return run_live_random(cfg, env, action_count, metrics);
    }

    #[cfg(not(feature = "onnx-reload"))]
    {
        let _ = (action_count, metrics);
        tracing::error!(
            "trained-mode live runner requires the `onnx-reload` Cargo \
             feature; rebuild with `--features mc-live --features onnx-reload` \
             (or `--features mc-live-bundled` to also bundle libonnxruntime.so). \
             Random-baseline mode (`random_actions = true` in runner.toml) \
             does not need ORT and works in this build."
        );
        Err(RunnerError::ConfigLoad(
            "trained-mode live runner requires the `onnx-reload` Cargo \
             feature; rebuild with `--features mc-live --features onnx-reload` \
             (or `--features mc-live-bundled`)."
                .into(),
        ))
    }

    #[cfg(feature = "onnx-reload")]
    {
        run_live_trained(cfg, env, action_count, metrics)
    }
}

/// Trained-mode live-runner branch: loads the ONNX manifest + bundle,
/// installs the hot-reload wrapper, and drives the runner with a real
/// `OnnxMuZeroModel`. Pulled into its own function so the ORT bits
/// can be feature-gated together (`#[cfg(feature = "onnx-reload")]`).
#[cfg(feature = "onnx-reload")]
fn run_live_trained(
    cfg: RunnerConfig,
    env: MinecraftEnv,
    action_count: u32,
    metrics: Option<MetricsRecorder>,
) -> Result<(), RunnerError> {
    // 4'. Load manifest + resolve bundle dir.
    let manifest = ModelManifest::load_json(&cfg.manifest_path).map_err(|e| {
        RunnerError::ConfigLoad(format!(
            "load manifest {}: {e}",
            cfg.manifest_path.display()
        ))
    })?;
    let bundle_dir = cfg
        .onnx
        .bundle_dir
        .clone()
        .or_else(|| cfg.manifest_path.parent().map(PathBuf::from))
        .ok_or_else(|| {
            RunnerError::ConfigLoad(format!(
                "cannot derive bundle_dir from manifest_path {} (no parent); \
                 set onnx.bundle_dir explicitly",
                cfg.manifest_path.display()
            ))
        })?;

    // 5. Build initial OnnxModelConfig. `action_space_size = 0` in the
    // runner config means "auto-derive from action_map" — handled here
    // BEFORE the `OnnxRuntimeConfig` lookup loses the bot-side count.
    let onnx_invariants = resolve_onnx_invariants(&cfg.onnx, action_count);
    let onnx_cfg = config_from_manifest(
        &manifest,
        &bundle_dir,
        onnx_invariants.action_space_size,
        onnx_invariants.latent_dim,
        onnx_invariants.num_threads,
    );

    // 6. Load the model.
    let model = OnnxMuZeroModel::load(onnx_cfg)
        .map_err(|e| RunnerError::EnvSetup(format!("OnnxMuZeroModel::load failed: {e}")))?;
    info!(
        manifest_version = manifest.version,
        bundle_dir = %bundle_dir.display(),
        "runner mode: trained (ONNX model loaded)"
    );
    // Stamp the initial model version into the metrics gauge so
    // observers see `forge_mc_model_version >= 1` before the first
    // hot-reload.
    if let Some(rec) = metrics.as_ref() {
        rec.set_model_version(manifest.version);
    }

    // 7. Build planner + writer + watcher.
    let mut mcts_cfg = LatentMctsConfig::default();
    mcts_cfg.base.num_simulations = cfg.planning_sims;
    mcts_cfg.add_exploration_noise = false;
    let search = LatentMctsSearch::new(model, mcts_cfg);

    // The runner expects a flat observation. `forge-env-mc` advertises
    // `ObsSpec::flat_f32(...)` at connect time, so `num_elements()`
    // gives the flat length directly. The shape itself is a 1-D
    // vector; any future multi-dim env would need its own runner.
    let obs_dim = env.obs_spec().num_elements();
    if obs_dim == 0 {
        return Err(RunnerError::EnvSetup(
            "MinecraftEnv obs_spec.num_elements() == 0; bot did not advertise an obs_dim".into(),
        ));
    }
    let writer = TrajectoryWriter::new(
        &cfg.trajectory_dir,
        &cfg.env_id,
        &cfg.schema_id,
        obs_dim,
        action_count,
    )
    .with_compression(cfg.trajectory_compression, cfg.trajectory_gzip_level);

    let watcher = HotReloadWatcher::new(&cfg.manifest_path);

    // 8. Build reload-fn wrapper that ALSO stamps the metric gauge on
    // each successful reload.
    let reload_fn = build_reload_fn_with_metrics(
        bundle_dir,
        onnx_invariants.action_space_size,
        onnx_invariants.latent_dim,
        onnx_invariants.num_threads,
        metrics.clone(),
    );

    // 9. Assemble + run.
    let mut runner = Runner::new(cfg, env, search, writer, watcher);
    if let Some(rec) = metrics {
        runner = runner.with_metrics(rec);
    }
    runner = runner.with_reload_fn(reload_fn);
    let outcome = runner.run(None)?;
    info!(?outcome, "live runner finished");
    Ok(())
}

/// Random-baseline live-runner branch — no ONNX, no manifest, no
/// hot-reload. Constructs a `Runner<MinecraftEnv, RandomLatentModel>`
/// where the runner's planning step samples actions uniformly at
/// random (gated by `cfg.random_actions`) and the
/// `RandomLatentModel` exists only to satisfy the type generic.
///
/// Used by the v0.5 first-real-run baseline-capture flow (T3/T4)
/// to produce a calibrated random-policy comparison snapshot for
/// the trained agent. Returns when the runner reaches its episode
/// budget or hits a fatal env error.
fn run_live_random(
    cfg: RunnerConfig,
    env: MinecraftEnv,
    action_count: u32,
    metrics: Option<MetricsRecorder>,
) -> Result<(), RunnerError> {
    info!(
        action_count,
        episodes = cfg.episodes,
        "runner mode: random (no ONNX load, MCTS bypassed)"
    );

    let obs_dim = env.obs_spec().num_elements();
    if obs_dim == 0 {
        return Err(RunnerError::EnvSetup(
            "MinecraftEnv obs_spec.num_elements() == 0; bot did not advertise an obs_dim".into(),
        ));
    }

    // RandomLatentModel uses the resolved latent_dim purely as
    // a zero-init `LatentState` size when inference is *accidentally*
    // invoked. The runner's planning branch short-circuits before
    // we ever call into the model in random mode.
    let model = RandomLatentModel::new(action_count, cfg.onnx.latent_dim);
    let mut mcts_cfg = LatentMctsConfig::default();
    mcts_cfg.base.num_simulations = 0;
    mcts_cfg.add_exploration_noise = false;
    let search = LatentMctsSearch::new(model, mcts_cfg);

    let writer = TrajectoryWriter::new(
        &cfg.trajectory_dir,
        &cfg.env_id,
        &cfg.schema_id,
        obs_dim,
        action_count,
    )
    .with_compression(cfg.trajectory_compression, cfg.trajectory_gzip_level);

    let watcher = HotReloadWatcher::new(&cfg.manifest_path);

    let mut runner = Runner::new(cfg, env, search, writer, watcher);
    if let Some(rec) = metrics {
        // Stamp a constant `model_version=0` gauge so Prometheus
        // observers see the random-baseline run as a distinct "zero"
        // version from any trained-mode run that bumped > 0.
        rec.set_model_version(0);
        runner = runner.with_metrics(rec);
    }
    let outcome = runner.run(None)?;
    info!(?outcome, "live runner (random baseline) finished");
    Ok(())
}

/// Snapshot of the ONNX runtime invariants resolved at startup.
/// Resolved once so `into_reload_fn` and the initial `OnnxModelConfig`
/// share one source of truth.
#[cfg(feature = "onnx-reload")]
struct ResolvedOnnx {
    action_space_size: u32,
    latent_dim: usize,
    num_threads: usize,
}

#[cfg(feature = "onnx-reload")]
fn resolve_onnx_invariants(cfg: &OnnxRuntimeConfig, action_count_from_map: u32) -> ResolvedOnnx {
    let action_space_size = if cfg.action_space_size == 0 {
        action_count_from_map
    } else {
        if cfg.action_space_size != action_count_from_map {
            warn!(
                cfg = cfg.action_space_size,
                action_map = action_count_from_map,
                "onnx.action_space_size != action_map.action_count — using cfg value; \
                 verify the action map and bundle match"
            );
        }
        cfg.action_space_size
    };
    ResolvedOnnx {
        action_space_size,
        latent_dim: cfg.latent_dim,
        num_threads: cfg.num_threads,
    }
}

/// Build a `ReloadFn<OnnxMuZeroModel>` that wraps
/// `into_reload_fn(...)` to ALSO stamp `recorder.set_model_version`
/// on each successful manifest swap. Pulled into its own function
/// so the test surface can verify the wrapping discipline
/// independent of the rest of `run_live`.
#[cfg(feature = "onnx-reload")]
fn build_reload_fn_with_metrics(
    bundle_dir: PathBuf,
    action_space_size: u32,
    latent_dim: usize,
    num_threads: usize,
    metrics: Option<MetricsRecorder>,
) -> ReloadFn<OnnxMuZeroModel> {
    Box::new(
        move |model: &mut OnnxMuZeroModel, manifest: &ModelManifest| {
            let new_config = config_from_manifest(
                manifest,
                &bundle_dir,
                action_space_size,
                latent_dim,
                num_threads,
            );
            model
                .reload(new_config)
                .map_err(|e: OnnxReloadError| RunnerError::Reload(e.to_string()))?;
            if let Some(rec) = metrics.as_ref() {
                rec.set_model_version(manifest.version);
            }
            Ok(())
        },
    )
}

#[cfg(all(test, feature = "onnx-reload"))]
mod tests {
    use super::*;

    #[test]
    fn resolve_onnx_invariants_uses_map_action_count_when_cfg_zero() {
        let cfg = OnnxRuntimeConfig {
            action_space_size: 0,
            latent_dim: 256,
            num_threads: 1,
            bundle_dir: None,
        };
        let r = resolve_onnx_invariants(&cfg, 12);
        assert_eq!(r.action_space_size, 12);
        assert_eq!(r.latent_dim, 256);
        assert_eq!(r.num_threads, 1);
    }

    #[test]
    fn resolve_onnx_invariants_warns_but_uses_cfg_when_mismatched() {
        // The warning surfaces in `tracing` logs; we can't easily
        // capture that without a test logger. The behavioural assert
        // — `cfg` wins over `action_map.action_count` — is the
        // structurally important part.
        let cfg = OnnxRuntimeConfig {
            action_space_size: 99,
            latent_dim: 256,
            num_threads: 1,
            bundle_dir: None,
        };
        let r = resolve_onnx_invariants(&cfg, 12);
        assert_eq!(r.action_space_size, 99);
    }

    #[test]
    fn build_reload_fn_with_metrics_returns_callable_box() {
        // Type-shape assertion only. The `ort` incompatibility this
        // comment used to cite is fixed: the `onnx-features` CI job
        // builds and tests `--features mc-live-bundled` (which implies
        // `onnx-reload`) on every push. What remains unverified here is
        // the reload path end-to-end against a real bundle -- the
        // callable is only checked for its type shape.
        let _fn: ReloadFn<OnnxMuZeroModel> =
            build_reload_fn_with_metrics(PathBuf::from("."), 4, 8, 1, None);
    }
}
