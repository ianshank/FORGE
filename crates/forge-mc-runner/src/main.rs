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
use forge_mc_runner::{HotReloadWatcher, Runner, RunnerConfig, TrajectoryWriter};
use tracing::{error, info};
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

    if let Err(e) = config.validate() {
        error!("invalid config: {e}");
        return ExitCode::from(2);
    }

    info!(?config, "runner configuration");

    if cli.dry_run {
        match run_dry(config) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                error!("dry-run failed: {e}");
                ExitCode::from(1)
            }
        }
    } else {
        error!(
            "live runner wiring (MinecraftEnv + OnnxMuZeroModel) is not yet \
             integrated in this binary. Re-run with --dry-run for now."
        );
        ExitCode::from(64)
    }
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

fn run_dry(config: RunnerConfig) -> Result<(), forge_mc_runner::RunnerError> {
    use forge_agent::latent_mcts::model::StubLatentModel;
    use forge_agent::latent_mcts::search::{LatentMctsConfig, LatentMctsSearch};

    // Construct an in-process stub env that mirrors the writer's dims.
    let obs_dim: usize = 8;
    let action_count: u32 = 4;
    let env = dry_run::StubEnv::new(obs_dim, action_count, Some(8));

    let mut mcts_cfg = LatentMctsConfig::default();
    mcts_cfg.base.num_simulations = config.planning_sims;
    mcts_cfg.add_exploration_noise = false;
    let search = LatentMctsSearch::new(StubLatentModel::new(action_count, 16), mcts_cfg);

    let writer = TrajectoryWriter::new(
        &config.trajectory_dir,
        &config.env_id,
        &config.schema_id,
        obs_dim,
        action_count,
    );
    let watcher = HotReloadWatcher::new(&config.manifest_path);

    let mut runner = Runner::new(config, env, search, writer, watcher);
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
