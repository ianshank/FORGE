//! `forge-mc-runner` — CLI driver for the latent-MCTS Minecraft runner.
//!
//! Loads a [`RunnerConfig`] from a TOML file, wires a runner together,
//! and runs `--episodes` episodes (or the value baked into the config,
//! whichever wins).
//!
//! This binary is intentionally thin: the heavy lifting lives in
//! [`forge_mc_runner::Runner`]. Concrete env + model construction is
//! deferred to follow-up wiring with `forge-env-mc::MinecraftEnv` and
//! `forge_agent::latent_mcts::onnx_model::OnnxMuZeroModel` — the
//! `--dry-run` mode below exercises the loop with the in-process stub
//! env/model and is useful for smoke-testing the CLI plumbing without a
//! Minecraft server.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use forge_mc_runner::{
    serve_metrics, HotReloadWatcher, MetricsRecorder, Runner, RunnerConfig, TrajectoryWriter,
};
use tokio::runtime::Builder as TokioBuilder;
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(
    name = "forge-mc-runner",
    version,
    about = "Episode runner driving a FlatObsEnv + latent-MCTS planner"
)]
struct Cli {
    /// Path to a TOML config that deserialises into [`RunnerConfig`].
    /// Any unspecified field falls back to its `Default`.
    #[arg(long, value_name = "TOML_PATH")]
    config: Option<PathBuf>,

    /// Override the number of episodes to run. `0` means run forever.
    #[arg(long)]
    episodes: Option<u64>,

    /// Run the loop with an in-process stub env + stub model. Useful
    /// for CLI smoke tests without a live Minecraft server.
    #[arg(long)]
    dry_run: bool,

    /// Override the runner config's `mc_env_config_path`. Lets
    /// operators point the runner at a sibling `env.toml` without
    /// editing `runner.toml`. Only consumed in the live path; ignored
    /// for `--dry-run`.
    #[arg(long, value_name = "TOML_PATH")]
    mc_config: Option<PathBuf>,
}

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("forge_mc_runner=info,forge_agent=info")),
        )
        .with_target(true)
        .init();

    let cli = Cli::parse();

    let config = match load_config(cli.config.as_deref()) {
        Ok(c) => c,
        Err(e) => {
            error!("failed to load config: {e}");
            return ExitCode::from(2);
        }
    };

    let config = match cli.episodes {
        Some(n) => RunnerConfig {
            episodes: n,
            ..config
        },
        None => config,
    };
    // `--mc-config` CLI flag overrides `runner.toml`'s
    // `mc_env_config_path`. Env-var ladder on `schema_id` is applied
    // last so the orchestrator's `FORGE_MC_SCHEMA_ID` wins over both
    // TOML and `--mc-config`.
    let config = if let Some(path) = cli.mc_config.clone() {
        RunnerConfig {
            mc_env_config_path: Some(path),
            ..config
        }
    } else {
        config
    };
    let config = config.with_env_var_overrides();

    if let Err(e) = config.validate() {
        error!("invalid config: {e}");
        return ExitCode::from(2);
    }

    info!(?config, "runner configuration");

    // The tokio worker count flows through config — no hard-coded
    // literal in the binary. Production deployments override via TOML
    // when the metrics endpoint + heavier hot-reload work warrants
    // more threads.
    let runtime = match TokioBuilder::new_multi_thread()
        .worker_threads(config.tokio_worker_threads)
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            error!("failed to build tokio runtime: {e}");
            return ExitCode::from(2);
        }
    };

    runtime.block_on(async_main(cli, config))
}

async fn async_main(cli: Cli, config: RunnerConfig) -> ExitCode {
    let (metrics_recorder, metrics_handle, metrics_shutdown_tx) =
        match maybe_spawn_metrics_server(&config).await {
            Ok(parts) => parts,
            Err(rc) => return rc,
        };

    if cli.dry_run {
        let runner_result = tokio::task::spawn_blocking({
            let cfg = config.clone();
            let metrics = metrics_recorder.clone();
            move || run_dry(cfg, metrics)
        })
        .await;

        // Always shut the metrics server down, regardless of the
        // runner's exit code.
        if let Some(tx) = metrics_shutdown_tx {
            let _ = tx.send(());
        }
        if let Some(handle) = metrics_handle {
            if let Err(e) = handle.await {
                warn!("metrics server task join error: {e}");
            }
        }

        match runner_result {
            Ok(Ok(())) => ExitCode::SUCCESS,
            Ok(Err(e)) => {
                error!("dry-run failed: {e}");
                ExitCode::from(1)
            }
            Err(e) => {
                error!("runner task panicked: {e}");
                ExitCode::from(1)
            }
        }
    } else {
        // Live runner path. Mirrors `--dry-run`'s `spawn_blocking`
        // shape because `MinecraftEnv::connect` + `OnnxMuZeroModel::load`
        // are synchronous blocking calls that would stall the async
        // runtime otherwise.
        #[cfg(feature = "mc-live")]
        let runner_result = tokio::task::spawn_blocking({
            let cfg = config.clone();
            let metrics = metrics_recorder.clone();
            move || forge_mc_runner::run_live(cfg, metrics)
        })
        .await;
        #[cfg(not(feature = "mc-live"))]
        let runner_result: Result<
            Result<(), forge_mc_runner::RunnerError>,
            tokio::task::JoinError,
        > = {
            error!(
                "live runner not available: re-compile with `--features mc-live` \
                 (transitively pulls forge-env-mc + ort). Re-run with --dry-run for the stub path."
            );
            Ok(Err(forge_mc_runner::RunnerError::ConfigLoad(
                "binary built without `mc-live` feature".into(),
            )))
        };

        // Tear the metrics server down regardless of the runner's
        // exit code — same pattern as the dry-run branch.
        if let Some(tx) = metrics_shutdown_tx {
            let _ = tx.send(());
        }
        if let Some(handle) = metrics_handle {
            if let Err(e) = handle.await {
                warn!("metrics server task join error: {e}");
            }
        }

        match runner_result {
            Ok(Ok(())) => ExitCode::SUCCESS,
            Ok(Err(e)) => {
                error!("live runner failed: {e}");
                ExitCode::from(1)
            }
            Err(e) => {
                error!("live runner task panicked: {e}");
                ExitCode::from(1)
            }
        }
    }
}

type MetricsParts = (
    Option<MetricsRecorder>,
    Option<tokio::task::JoinHandle<Result<(), forge_mc_runner::MetricsError>>>,
    Option<tokio::sync::oneshot::Sender<()>>,
);

async fn maybe_spawn_metrics_server(config: &RunnerConfig) -> Result<MetricsParts, ExitCode> {
    if config.metrics_disabled() {
        return Ok((None, None, None));
    }
    let recorder = match MetricsRecorder::new(&config.metrics_histogram_buckets) {
        Ok(r) => r,
        Err(e) => {
            error!("failed to build metrics recorder: {e}");
            return Err(ExitCode::from(2));
        }
    };
    let bind = format!("{}:{}", config.metrics_bind, config.metrics_port);
    let addr: std::net::SocketAddr = match bind.parse() {
        Ok(a) => a,
        Err(e) => {
            error!("invalid metrics bind addr {bind:?}: {e}");
            return Err(ExitCode::from(2));
        }
    };
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let shutdown = async move {
        // Future resolves when either the oneshot fires OR ctrl-c is
        // received — whichever comes first triggers graceful drain.
        let ctrl_c = async {
            if let Err(e) = tokio::signal::ctrl_c().await {
                warn!("ctrl-c handler install failed: {e}");
            }
        };
        tokio::select! {
            _ = rx => {},
            _ = ctrl_c => {},
        }
    };
    let handle = match serve_metrics(addr, recorder.clone(), shutdown).await {
        Ok(h) => h,
        Err(e) => {
            error!("metrics server failed to start: {e}");
            return Err(ExitCode::from(2));
        }
    };
    Ok((Some(recorder), Some(handle), Some(tx)))
}

fn load_config(path: Option<&std::path::Path>) -> Result<RunnerConfig, String> {
    match path {
        Some(p) => {
            let text =
                std::fs::read_to_string(p).map_err(|e| format!("read {}: {e}", p.display()))?;
            toml::from_str(&text).map_err(|e| format!("parse {}: {e}", p.display()))
        }
        None => Ok(RunnerConfig::default()),
    }
}

fn run_dry(
    config: RunnerConfig,
    metrics: Option<MetricsRecorder>,
) -> Result<(), forge_mc_runner::RunnerError> {
    use forge_agent::latent_mcts::model::StubLatentModel;
    use forge_agent::latent_mcts::search::{LatentMctsConfig, LatentMctsSearch};

    // Dry-run dims flow through config (no hard-coded values at the
    // call site). Defaults match the historical literal values
    // (`obs_dim = 8`, `action_count = 4`, `latent_dim = 16`,
    // `max_episode_len = 8`) so existing `--dry-run` smoke tests
    // behave identically.
    let obs_dim = config.dry_run.obs_dim;
    let action_count = config.dry_run.action_count;
    let env = dry_run::StubEnv::new(obs_dim, action_count, Some(config.dry_run.max_episode_len));

    let mut mcts_cfg = LatentMctsConfig::default();
    mcts_cfg.base.num_simulations = config.planning_sims;
    mcts_cfg.add_exploration_noise = false;
    let search = LatentMctsSearch::new(
        StubLatentModel::new(action_count, config.dry_run.latent_dim),
        mcts_cfg,
    );

    let writer = TrajectoryWriter::new(
        &config.trajectory_dir,
        &config.env_id,
        &config.schema_id,
        obs_dim,
        action_count,
    )
    .with_compression(config.trajectory_compression, config.trajectory_gzip_level);
    let watcher = HotReloadWatcher::new(&config.manifest_path);

    let mut runner = Runner::new(config, env, search, writer, watcher);
    if let Some(rec) = metrics {
        runner = runner.with_metrics(rec);
    }
    let outcome = runner.run(None)?;
    info!(?outcome, "dry-run complete");
    Ok(())
}

mod dry_run {
    use std::borrow::Cow;

    use forge_env::spec::{ActionSpec, ObsSpec};
    use forge_env::{Env, EnvError, FlatObsEnv, StepOutput};

    /// Stub env used by `--dry-run` smoke testing. Mirrors the shape of
    /// the test-only `StubFlatEnv` in `runner.rs` but lives here so the
    /// binary doesn't depend on `#[cfg(test)]` code.
    pub(super) struct StubEnv {
        obs_spec: ObsSpec,
        action_spec: ActionSpec,
        obs_dim: usize,
        action_count: u32,
        tick: u64,
        terminate_at: Option<u64>,
    }

    impl StubEnv {
        pub fn new(obs_dim: usize, action_count: u32, terminate_at: Option<u64>) -> Self {
            Self {
                obs_spec: ObsSpec::flat_f32("dry-run-stub", obs_dim, 0.0, 1.0),
                action_spec: ActionSpec::discrete(action_count),
                obs_dim,
                action_count,
                tick: 0,
                terminate_at,
            }
        }
    }

    impl Env for StubEnv {
        type Obs = Vec<f32>;
        type Action = u32;
        type Info = ();
        type Error = EnvError;

        fn reset_into(
            &mut self,
            _seed: Option<u64>,
            out: &mut Vec<f32>,
        ) -> Result<(), Self::Error> {
            self.tick = 0;
            out.clear();
            out.resize(self.obs_dim, 0.0);
            Ok(())
        }

        fn step_into(
            &mut self,
            action: u32,
            out: &mut StepOutput<Vec<f32>, ()>,
        ) -> Result<(), Self::Error> {
            if action >= self.action_count {
                return Err(EnvError::InvalidAction {
                    action_id: action,
                    space_n: self.action_count,
                });
            }
            self.tick += 1;
            out.obs.clear();
            out.obs.resize(self.obs_dim, self.tick as f32);
            out.reward = 1.0;
            out.terminated = matches!(self.terminate_at, Some(t) if self.tick >= t);
            out.truncated = false;
            Ok(())
        }

        fn obs_spec(&self) -> &ObsSpec {
            &self.obs_spec
        }
        fn action_spec(&self) -> &ActionSpec {
            &self.action_spec
        }
        fn name(&self) -> Cow<'_, str> {
            Cow::Borrowed("dry-run-stub")
        }
    }

    impl FlatObsEnv for StubEnv {
        fn obs_dim(&self) -> usize {
            self.obs_dim
        }
        fn num_actions(&self) -> u32 {
            self.action_count
        }
    }
}
