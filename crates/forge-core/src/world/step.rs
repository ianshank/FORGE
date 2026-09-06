//! Simulation step methods and step result generation.

use forge_types::observation::{Observation, StepResult};
use forge_types::Action;
use tracing::{instrument, trace};

use super::WorldState;
use crate::systems;

impl WorldState {
    /// Advances the simulation by one tick with the given actions.
    ///
    /// Returns a freshly allocated [`StepResult`]. This is the convenience
    /// API; for the **zero-allocation hot path**, call [`Self::step_into`]
    /// with a caller-owned, reused `StepResult` buffer instead.
    #[instrument(skip_all)]
    pub fn step(&mut self, actions: &[Action]) -> StepResult {
        let mut result = StepResult::default();
        self.step_into(actions, &mut result);
        result
    }

    /// Advances the simulation by one tick, filling the supplied
    /// [`StepResult`] in place.
    ///
    /// This is the **hot-path entry point**: every internal buffer used by
    /// the systems pipeline lives on `WorldState` and is reused across
    /// ticks; the supplied `result` buffer is also reused (its inner
    /// `Vec`s are cleared and re-extended rather than reallocated).
    ///
    /// # Zero-allocation contract
    ///
    /// After a single warm-up call, subsequent invocations execute without
    /// any heap allocations **in the audited configuration**: a single
    /// `WorldState` with `tasks` empty and the agri pipeline disabled.
    /// This is what CI enforces today via
    /// `crates/forge-bench/src/bin/allocation_audit.rs` +
    /// `benchmarks/runner/check_zero_alloc.py`.
    ///
    /// Known caveats outside the audited configuration:
    /// - When `self.tasks` is non-empty, `forge_task::evaluator::evaluate_tasks`
    ///   allocates a fresh `Vec<f32>` per tick. Refactoring the task
    ///   evaluator to write into a `WorldState`-owned reward scratch is
    ///   tracked as tech debt in `docs/next_steps.md`.
    /// - Logging via `tracing` may allocate when fields are recorded;
    ///   keep `RUST_LOG=warn` (or below `info`) on the hot path.
    ///
    /// Pass an arbitrary `StepResult` (e.g. `StepResult::default()`); on
    /// the first call the inner buffers will allocate to fit the agent
    /// count, and every subsequent call will reuse that capacity.
    #[instrument(skip_all)]
    pub fn step_into(&mut self, actions: &[Action], result: &mut StepResult) {
        if self.terminated || self.truncated {
            self.fill_step_result(result);
            return;
        }

        trace!(tick = self.tick, num_actions = actions.len(), "step");

        // Pad or truncate actions to match agent count, reusing the
        // pre-allocated `step_actions` buffer.
        let agent_count = self.agents.len();
        self.step_actions.clear();
        let take = actions.len().min(agent_count);
        self.step_actions.extend_from_slice(&actions[..take]);
        self.step_actions.resize(agent_count, Action::Noop);

        // Run all systems
        systems::run_systems(self);

        // Check truncation (max episode length)
        if self.config.task.max_episode_length > 0
            && self.tick >= self.config.task.max_episode_length
        {
            self.truncated = true;
        }

        // Check termination (all agents dead)
        if self.agents.iter().all(|a| !a.alive) {
            self.terminated = true;
        }

        self.fill_step_result(result);
    }

    /// Fills the supplied [`StepResult`] in place from the current state.
    ///
    /// The result's inner `Vec`s are reused — `clear`+`extend`/`resize`
    /// rather than reallocated — so repeated calls with the same `out`
    /// buffer perform no heap allocations after the first warm call.
    pub(crate) fn fill_step_result(&mut self, out: &mut StepResult) {
        let n = self.agents.len();

        // Reuse the existing observation slots; per-slot inner Vecs are
        // cleared and re-extended by `fill_observation`.
        if out.observations.len() < n {
            out.observations.resize_with(n, Observation::default);
        } else if out.observations.len() > n {
            out.observations.truncate(n);
        }
        for (i, agent) in self.agents.iter().enumerate() {
            // Safe: bounds guaranteed by the resize above.
            self.fill_observation(agent, &mut out.observations[i]);
        }

        // Rewards: prefer the task evaluator's Vec when present (move it
        // out via take); otherwise zero-fill the reused buffer.
        out.rewards.clear();
        match self.last_task_rewards.take() {
            Some(rewards) => out.rewards.extend_from_slice(&rewards),
            None => out.rewards.resize(n, 0.0),
        }

        out.terminated = self.terminated;
        out.truncated = self.truncated;

        let info = &mut out.info;
        info.tick = self.tick;
        info.agents_alive.clear();
        info.agents_alive
            .extend(self.agents.iter().map(|a| a.alive));
        // tasks_completed is currently always per-agent empty inner Vecs
        // (no system populates it). Resize keeps the outer capacity and
        // replaces inner Vecs with empty ones — empty Vec::new() does not
        // allocate, so this is zero-alloc.
        info.tasks_completed.clear();
        info.tasks_completed.resize_with(n, Vec::new);
        info.total_resources = self.resources.iter().map(|r| r.quantity as u32).sum();
        info.day_phase = self.day_phase;
    }
}
