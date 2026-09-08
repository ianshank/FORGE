//! [`Runner`] — drives episodes against a [`FlatObsEnv`] using
//! [`LatentMctsSearch`], writes [`TrajectoryV2`] per episode via the
//! shared [`TrajectoryWriter`], and applies hot-reloads emitted by the
//! [`HotReloadWatcher`] strictly between episodes.
//!
//! ## Why it lives here (not in `forge-agent`)
//!
//! The runner stitches four orthogonal pieces (env, planner, replay
//! writer, reload watcher) into one loop. Hosting it in `forge-mc-runner`
//! keeps `forge-agent` focused on planning and avoids pulling
//! `forge-env` / `forge-replay` into the planner crate.
//!
//! ## Zero-allocation discipline
//!
//! The loop reuses two `Vec<f32>` buffers — `obs_buf` (the
//! pre-step observation fed into the planner) and `step_out.obs` (the
//! post-step observation returned by the env). They are swapped at the
//! end of every step via [`std::mem::swap`], avoiding a per-step
//! allocation on the hot path. The remaining per-step allocations are:
//!
//! - one `Vec<f32>` for the policy target (length `action_count`),
//! - one [`StepV2`] struct copy stored in the writer's in-memory buffer
//!   (atomically flushed at episode end).
//!
//! Both are intrinsic to the trajectory format and cannot be avoided
//! without changing the on-disk schema. The `obs_buf` clone fed into
//! the recorded `StepV2.obs` is unavoidable: trajectories own their
//! observations independently of the runner buffer.
//!
//! ## Hot-reload contract (plan §3.4)
//!
//! [`HotReloadWatcher::poll`] is invoked exactly once at the top of
//! [`Runner::run`]'s outer episode loop — never mid-episode. When a
//! [`ReloadEvent`] is emitted, the runner takes an `&mut` borrow on the
//! model via [`LatentMctsSearch::model_mut`] and invokes the reload
//! callback (set via [`Runner::with_reload_fn`]). Models that do not
//! support reload (e.g. `StubLatentModel` in tests) simply leave the
//! callback unset and version bumps are recorded without changing
//! weights.

use chrono::Utc;
use forge_agent::latent_mcts::model::LatentForwardModel;
use forge_agent::latent_mcts::search::{LatentMctsSearch, LatentSearchResult};
use forge_env::{FlatObsEnv, StepOutput};
use forge_replay::v2::StepV2;
use rand_pcg::Pcg64Mcg;
use tracing::{debug, info, instrument, warn};

use crate::config::RunnerConfig;
use crate::error::RunnerError;
use crate::hot_reload::{HotReloadWatcher, ReloadEvent};
use crate::manifest::ModelManifest;
use crate::metrics::{
    MetricsRecorder, METRIC_REASON_ENV_RESET, METRIC_REASON_ENV_STEP, METRIC_REASON_PLANNER,
};
use crate::random_baseline::sample_random_action;
use crate::trajectory::TrajectoryWriter;

/// Prefix used for the trajectory-file episode identifier (e.g.
/// `ep-000001.json`). The Python-side `TrajectoryReader` (in
/// `python/forge/training/muzero_mc/replay.py`) globs files using
/// this prefix, so any change MUST be mirrored there and version-
/// pinned by a cross-language test.
pub const EPISODE_ID_PREFIX: &str = "ep-";

/// Zero-pad width for the episode sequence number in the formatted
/// episode id (e.g. `ep-000001`). Matches the Python-side
/// `TrajectoryReader` glob and trajectory naming convention.
pub const EPISODE_ID_PAD_WIDTH: usize = 6;

/// Format a 1-based episode sequence number into the canonical
/// trajectory-id string (`ep-NNNNNN`). Pinned through
/// [`EPISODE_ID_PREFIX`] and [`EPISODE_ID_PAD_WIDTH`] so no caller
/// has to know the layout.
pub fn format_episode_id(seq: u64) -> String {
    format!(
        "{prefix}{seq:0width$}",
        prefix = EPISODE_ID_PREFIX,
        seq = seq,
        width = EPISODE_ID_PAD_WIDTH,
    )
}

/// Signature for the model hot-reload callback.
///
/// Invoked by [`Runner`] between episodes whenever
/// [`HotReloadWatcher`] reports a strictly-monotonic manifest version
/// bump. The runner hands the callback an exclusive borrow on the
/// model so it can swap ONNX session handles (or perform any
/// equivalent in-place mutation).
///
/// Callbacks must be `Send` so the runner stays `Send`.
pub type ReloadFn<M> = Box<dyn FnMut(&mut M, &ModelManifest) -> Result<(), RunnerError> + Send>;

/// Summary of a single completed episode.
#[derive(Debug, Clone)]
pub struct EpisodeOutcome {
    /// Episode identifier (also the trajectory file stem).
    pub episode_id: String,
    /// Number of env `step_into` calls that completed (≤
    /// `RunnerConfig::max_steps_per_episode`).
    pub steps: u64,
    /// Sum of per-step rewards recorded in the trajectory.
    pub total_reward: f32,
    /// `true` iff the env reported `terminated` at episode end.
    pub terminated: bool,
    /// `true` iff the env reported `truncated` at episode end **or** the
    /// runner hit `max_steps_per_episode` without termination.
    pub truncated: bool,
    /// Path the trajectory file was atomically written to.
    pub trajectory_path: std::path::PathBuf,
}

/// Aggregate summary returned by [`Runner::run`].
///
/// Per-run counters (`episodes_completed`, `total_steps`,
/// `terminated_count`, `truncated_count`) are scoped to the current
/// [`Runner::run`] call; they start at zero on every invocation.
///
/// `reloads_applied` and `last_model_version` are **lifetime totals**
/// snapshotted from the runner — they reflect every reload the runner
/// has seen since construction, not just those applied during the
/// current `run`. This matches the integration tests at
/// `tests/runner_integration.rs` which assert across successive calls
/// to `runner.run(Some(1))`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RunnerOutcome {
    /// Episodes finished during *this* `run` call (regardless of
    /// terminated vs truncated outcome).
    pub episodes_completed: u64,
    /// Sum of `EpisodeOutcome.steps` across the episodes finished
    /// during *this* `run` call.
    pub total_steps: u64,
    /// Episodes that finished with `terminated=true` during this
    /// `run` call.
    pub terminated_count: u64,
    /// Episodes that finished with `truncated=true` (env-side or
    /// runner-side) during this `run` call.
    pub truncated_count: u64,
    /// **Lifetime total** of reloads the runner has applied since
    /// construction — not limited to this `run` call.
    pub reloads_applied: u64,
    /// **Lifetime** last manifest version the runner observed via the
    /// watcher; `None` until the first reload.
    pub last_model_version: Option<u64>,
    /// Episodes discarded because of a transient env error
    /// (`RECONNECTING` / `BUSY`) during *this* `run` call. These do
    /// not count toward [`Self::episodes_completed`].
    pub transient_discards: u64,
}

/// Episode-driving runner. Generic over the env (`E: FlatObsEnv`) and
/// the planner's latent forward model (`M: LatentForwardModel`).
pub struct Runner<E: FlatObsEnv, M: LatentForwardModel>
where
    E::Info: Default + serde::Serialize,
{
    config: RunnerConfig,
    env: E,
    search: LatentMctsSearch<M>,
    writer: TrajectoryWriter,
    watcher: HotReloadWatcher,
    reload_fn: Option<ReloadFn<M>>,
    metrics: Option<MetricsRecorder>,
    obs_buf: Vec<f32>,
    step_out: StepOutput<Vec<f32>, E::Info>,
    episode_seq: u64,
    last_model_version: Option<u64>,
    reloads_applied: u64,
    /// Lifetime count of hot reloads that were *rejected* (integrity
    /// failure, unreadable bundle, ORT error). Kept separate from
    /// `reloads_applied` because a rejected reload is not a failed run:
    /// the runner keeps serving the last verified model.
    reloads_rejected: u64,
    /// Per-runner deterministic RNG used by the random-actions baseline
    /// branch. Seeded lazily on first use from `config.base_seed`
    /// (falls back to a wall-clock-derived seed); per-episode mixing
    /// happens via the episode-seq stir below.
    random_rng: Option<Pcg64Mcg>,
}

impl<E: FlatObsEnv, M: LatentForwardModel> Runner<E, M>
where
    E::Info: Default + serde::Serialize,
{
    /// Build a new runner. Buffer sizes are derived from
    /// `writer.obs_dim()` — the writer is the single source of truth
    /// for `obs_dim` and `action_count`.
    pub fn new(
        config: RunnerConfig,
        env: E,
        search: LatentMctsSearch<M>,
        writer: TrajectoryWriter,
        watcher: HotReloadWatcher,
    ) -> Self {
        let obs_dim = writer.obs_dim();
        Self {
            config,
            env,
            search,
            writer,
            watcher,
            reload_fn: None,
            metrics: None,
            obs_buf: vec![0.0; obs_dim],
            step_out: StepOutput {
                obs: vec![0.0; obs_dim],
                reward: 0.0,
                terminated: false,
                truncated: false,
                info: E::Info::default(),
            },
            episode_seq: 0,
            last_model_version: None,
            reloads_applied: 0,
            reloads_rejected: 0,
            random_rng: None,
        }
    }

    /// Builder helper to install a reload callback. Without it, the
    /// runner still tracks manifest version bumps (advances
    /// `last_model_version`) but performs no model mutation.
    pub fn with_reload_fn(mut self, reload_fn: ReloadFn<M>) -> Self {
        self.reload_fn = Some(reload_fn);
        self
    }

    /// Builder helper to install a [`MetricsRecorder`]. Without it,
    /// every per-episode / per-step metric call site is a no-op,
    /// preserving the binary's zero-dependency story when
    /// `RunnerConfig::metrics_disabled()` is true.
    pub fn with_metrics(mut self, recorder: MetricsRecorder) -> Self {
        self.metrics = Some(recorder);
        self
    }

    /// Pre-seed the watcher so a manifest version already on disk does
    /// not trigger a spurious first-poll reload.
    pub fn prime_watcher_with(&mut self, version: u64) {
        self.watcher.prime_with(version);
        self.last_model_version = Some(version);
    }

    /// Last manifest version the runner has observed (via the watcher).
    pub fn last_model_version(&self) -> Option<u64> {
        self.last_model_version
    }

    /// Number of reloads applied so far across the runner's lifetime.
    pub fn reloads_applied(&self) -> u64 {
        self.reloads_applied
    }

    /// Lifetime count of hot reloads rejected before they were applied.
    ///
    /// Non-zero means a manifest bump pointed at a bundle that failed
    /// verification; the runner continued on the previous model.
    #[must_use]
    pub fn reloads_rejected(&self) -> u64 {
        self.reloads_rejected
    }

    /// Number of episodes started since construction.
    pub fn episode_seq(&self) -> u64 {
        self.episode_seq
    }

    /// Borrow the underlying env (read-only).
    pub fn env(&self) -> &E {
        &self.env
    }

    /// Borrow the runner config (read-only).
    pub fn config(&self) -> &RunnerConfig {
        &self.config
    }

    /// Lazily-initialised RNG for the random-actions baseline branch.
    ///
    /// Seed source:
    ///   * `config.base_seed`           — deterministic across runs
    ///   * `SystemTime` ns fallback     — non-deterministic but
    ///     reproducible within a single runner process
    ///
    /// Returns `&mut Pcg64Mcg` so the planning branch can call
    /// `sample_random_action(rng, action_count)` without re-seeding.
    fn random_rng_mut(&mut self) -> &mut Pcg64Mcg {
        if self.random_rng.is_none() {
            let seed: u128 = match self.config.base_seed {
                Some(s) => u128::from(s).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                None => {
                    use std::time::{SystemTime, UNIX_EPOCH};
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map(|d| d.as_nanos())
                        .unwrap_or(0xDEAD_BEEF_CAFE_F00D)
                }
            };
            self.random_rng = Some(Pcg64Mcg::new(seed));
        }
        self.random_rng.as_mut().expect("just seeded")
    }

    /// Consume the runner and return its parts.
    pub fn into_parts(self) -> (E, LatentMctsSearch<M>, TrajectoryWriter, HotReloadWatcher) {
        (self.env, self.search, self.writer, self.watcher)
    }

    /// Poll the watcher; if a fresh manifest version landed, invoke the
    /// reload callback. Must only be called between episodes.
    fn maybe_reload(&mut self) -> Result<(), RunnerError> {
        let Some(event) = self.watcher.poll()? else {
            return Ok(());
        };
        let ReloadEvent {
            new_version,
            previous_version,
            manifest,
        } = event;
        info!(
            new_version,
            previous_version = ?previous_version,
            "applying hot reload"
        );
        if let Some(reloader) = self.reload_fn.as_mut() {
            let model = self.search.model_mut();
            // Propagate the error unchanged. Collapsing it into
            // `RunnerError::Reload(e.to_string())` destroyed exactly the
            // structure `integrity` exists to provide: an operator matching on
            // `ModelDigestMismatch` to tell bundle tampering apart from a
            // transient ORT failure would never see that variant, despite
            // `onnx_reload`'s doc promising it surfaces "uncollapsed".
            reloader(model, &manifest)?;
        } else {
            debug!("no reload_fn installed; version recorded only");
        }
        self.last_model_version = Some(new_version);
        self.reloads_applied += 1;
        if let Some(rec) = self.metrics.as_ref() {
            rec.set_model_version(new_version);
            rec.record_model_reload();
        }
        Ok(())
    }

    /// Run a single episode end-to-end: reset → plan → step → record →
    /// finalize. Returns the per-episode summary.
    #[instrument(skip(self), fields(episode_seq = self.episode_seq + 1))]
    pub fn run_episode(&mut self) -> Result<EpisodeOutcome, RunnerError> {
        self.episode_seq += 1;
        let episode_id = format_episode_id(self.episode_seq);
        let seed = self
            .config
            .base_seed
            .map(|s| s.wrapping_add(self.episode_seq));
        let started_at = Utc::now().to_rfc3339();

        self.writer.start_episode(&episode_id, seed, &started_at)?;

        // Reset env into reused buffer.
        self.env
            .reset_into(seed, &mut self.obs_buf)
            .map_err(|e| env_err(e, METRIC_REASON_ENV_RESET))?;

        let max_steps = self.config.max_steps_per_episode;
        let action_repeat = self.config.action_repeat.max(1);

        let mut steps_taken: u64 = 0;
        let mut total_reward: f32 = 0.0;
        let mut terminated = false;
        let mut truncated = false;

        // Resolve action_count up-front for the random-actions branch.
        // `FlatObsEnv::num_actions` is the canonical source — every
        // env impl computes it from its `ActionSpec::discrete_n()`.
        let action_count = self.env.num_actions();

        for tick in 0..max_steps {
            // Plan from the *current* (pre-step) observation.
            // Wall-clock per planning call is recorded into the
            // `forge_mc_planning_latency_seconds` histogram if a
            // metrics recorder is installed.
            let plan_start = std::time::Instant::now();
            let plan_result = if self.config.random_actions {
                // Random-actions baseline: skip MCTS entirely and
                // sample uniformly from the action space. Returns a
                // synthetic LatentSearchResult whose `visit_counts`
                // is a uniform distribution (so the recorded
                // `policy_target` reflects the random policy) and
                // `root_value = 0.0`.
                let rng = self.random_rng_mut();
                let action = sample_random_action(rng, action_count);
                Ok(LatentSearchResult {
                    action,
                    visit_counts: vec![1; action_count as usize],
                    root_value: 0.0,
                })
            } else {
                self.search
                    .search(&self.obs_buf)
                    .map_err(|e| RunnerError::Planner(e.to_string()))
            };
            if let Some(rec) = self.metrics.as_ref() {
                rec.record_planning_latency_seconds(plan_start.elapsed().as_secs_f64());
            }
            let LatentSearchResult {
                action,
                visit_counts,
                root_value,
            } = match plan_result {
                Ok(r) => r,
                Err(e) => {
                    if let Some(rec) = self.metrics.as_ref() {
                        rec.record_protocol_error(METRIC_REASON_PLANNER);
                    }
                    return Err(e);
                }
            };

            // Visit counts → policy target (normalised distribution).
            let policy_target = normalize_visits(&visit_counts);

            // Apply the chosen action `action_repeat` times, accumulating
            // reward. Stop early if the env reports terminated/truncated.
            let mut step_reward: f32 = 0.0;
            for _ in 0..action_repeat {
                self.env
                    .step_into(action, &mut self.step_out)
                    .map_err(|e| env_err(e, METRIC_REASON_ENV_STEP))?;
                step_reward += self.step_out.reward;
                if self.step_out.terminated || self.step_out.truncated {
                    break;
                }
            }

            terminated = self.step_out.terminated;
            truncated = self.step_out.truncated;
            total_reward += step_reward;

            if let Some(rec) = self.metrics.as_ref() {
                if let Ok(info_val) = serde_json::to_value(&self.step_out.info) {
                    if let Some(breakdown) =
                        info_val.get("reward_breakdown").and_then(|v| v.as_object())
                    {
                        for (component, val) in breakdown {
                            if let Some(v_f64) = val.as_f64() {
                                rec.record_reward_component(component, v_f64 as f32);
                            }
                        }
                    }
                }
            }

            // Runner-side truncation prediction: if this is the final
            // iteration the loop will execute (last `tick` before the
            // cap) and the env did not flag terminated/truncated on its
            // own, set `truncated = true` BEFORE building the StepV2 so
            // the recorded trajectory's last step matches the
            // `EpisodeOutcome.truncated` value the caller sees. Without
            // this, downstream trainers that rely on `StepV2.truncated`
            // for n-step bootstrap-cut decisions would silently treat
            // a runner-side truncation as a normal continuation.
            let last_iteration = tick + 1 >= max_steps;
            if last_iteration && !terminated && !truncated {
                truncated = true;
            }

            // Record the step using the PRE-step observation (obs_buf).
            let step = StepV2 {
                tick,
                obs: self.obs_buf.clone(),
                action_id: action,
                policy_target,
                value_target: root_value,
                reward: step_reward,
                terminated,
                truncated,
            };
            self.writer.record_step(step)?;

            // Swap obs_buf ↔ step_out.obs so the next iteration plans
            // off the post-step observation without allocating.
            std::mem::swap(&mut self.obs_buf, &mut self.step_out.obs);

            steps_taken += 1;
            if terminated || truncated {
                break;
            }
        }

        // Belt-and-braces post-loop check. With the in-loop
        // last-iteration guard above this should always be a no-op,
        // but the assertion remains so that future loop refactors
        // still produce a consistent `EpisodeOutcome.truncated`.
        if steps_taken >= max_steps && !terminated && !truncated {
            truncated = true;
        }

        let ended_at = Utc::now().to_rfc3339();
        let trajectory_path = self.writer.finalize_and_save(&ended_at)?;

        if let Some(rec) = self.metrics.as_ref() {
            rec.record_episode_complete(total_reward);
            rec.record_episode_length(steps_taken as usize);
        }

        Ok(EpisodeOutcome {
            episode_id,
            steps: steps_taken,
            total_reward,
            terminated,
            truncated,
            trajectory_path,
        })
    }

    /// Run a sequence of episodes, polling the hot-reload watcher
    /// between each. The episode limit is the first of:
    ///
    /// 1. `max_episodes` if provided,
    /// 2. `config.episodes` if non-zero (`0` means "run forever"),
    /// 3. otherwise unbounded.
    #[instrument(skip(self), fields(max_episodes = ?max_episodes))]
    pub fn run(&mut self, max_episodes: Option<u64>) -> Result<RunnerOutcome, RunnerError> {
        let limit = match max_episodes {
            Some(n) => n,
            None if !self.config.runs_forever() => self.config.episodes,
            None => u64::MAX,
        };

        let mut outcome = RunnerOutcome::default();
        let mut consecutive_transient: u32 = 0;
        let max_transient = self.config.max_consecutive_transient_failures;
        let backoff = std::time::Duration::from_millis(self.config.transient_failure_backoff_ms);

        loop {
            if outcome.episodes_completed >= limit {
                break;
            }

            // Reload check at the top of the outer loop — strictly
            // between episodes per plan §3.4.
            //
            // A failed reload must NOT end the run. The trainer and runner
            // share `models/` as a host bind-mount, so a bad or tampered
            // bundle is exactly the case integrity checking exists to catch —
            // and propagating here would have turned "reject the bad bundle"
            // into "kill the training run", repeatable by anyone able to write
            // that directory. Keep serving the previously verified model and
            // retry on the next poll; `last_model_version` is left unchanged
            // by `maybe_reload`'s early return, so the bump is not consumed.
            if let Err(e) = self.maybe_reload() {
                warn!(
                    error = %e,
                    "hot reload rejected; continuing on the previously verified model"
                );
                self.reloads_rejected += 1;
            }

            match self.run_episode() {
                Ok(ep) => {
                    consecutive_transient = 0;
                    outcome.episodes_completed += 1;
                    outcome.total_steps += ep.steps;
                    if ep.terminated {
                        outcome.terminated_count += 1;
                    }
                    if ep.truncated {
                        outcome.truncated_count += 1;
                    }
                }
                Err(RunnerError::TransientEnv {
                    code,
                    message,
                    reason,
                }) => {
                    // Reconnect tears down mineflayer and starts a new
                    // world. The WS Error frame completed the pair —
                    // retrying recv hangs, resending Step stitches two
                    // MDPs. Discard the partial trajectory and Reset
                    // into a fresh episode.
                    self.writer.discard_current();
                    if let Some(rec) = self.metrics.as_ref() {
                        rec.record_protocol_error(reason);
                    }
                    consecutive_transient = consecutive_transient.saturating_add(1);
                    outcome.transient_discards = outcome.transient_discards.saturating_add(1);
                    warn!(
                        code = %code,
                        message = %message,
                        consecutive_transient,
                        max_transient,
                        episodes_completed = outcome.episodes_completed,
                        transient_discards = outcome.transient_discards,
                        "transient env error; discarding episode and continuing the run"
                    );
                    if max_transient == 0 || consecutive_transient >= max_transient {
                        return Err(RunnerError::TooManyTransientFailures {
                            count: consecutive_transient,
                            code,
                            message,
                        });
                    }
                    if !backoff.is_zero() {
                        std::thread::sleep(backoff);
                    }
                }
                Err(e) => {
                    warn!(error = %e, "episode failed");
                    self.writer.discard_current();
                    return Err(e);
                }
            }
        }

        outcome.reloads_applied = self.reloads_applied;
        outcome.last_model_version = self.last_model_version;
        Ok(outcome)
    }
}

/// Normalise raw visit counts into a probability distribution.
///
/// Zero-sum visit vectors (degenerate searches with `num_simulations=0`)
/// fall back to a uniform distribution so the recorded trajectory still
/// satisfies the `policy_target.iter().sum() ≈ 1.0` invariant downstream
/// trainers rely on.
fn normalize_visits(visits: &[u32]) -> Vec<f32> {
    let sum: u64 = visits.iter().map(|&v| v as u64).sum();
    if sum == 0 {
        let n = visits.len().max(1) as f32;
        return vec![1.0 / n; visits.len()];
    }
    let inv = 1.0 / sum as f32;
    visits.iter().map(|&v| v as f32 * inv).collect()
}

fn env_err<E>(e: E, reason: &'static str) -> RunnerError
where
    E: std::error::Error + Send + Sync + 'static,
{
    let msg = e.to_string();
    if let Some((code, message)) = crate::error::parse_transient_env_error(&msg) {
        RunnerError::TransientEnv {
            code,
            message,
            reason,
        }
    } else {
        RunnerError::Env(msg)
    }
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
#[path = "runner/tests.rs"]
mod tests;
