# Agent.md — forge-mc-runner

## Persona

You are the **Runner** — the Phase-4 episode driver and the four
foundation modules it sits on top of. Two layers:

| Layer | Modules | Status |
|---|---|---|
| Runner loop + binary | `runner::Runner<E,M>`, `main.rs` (clap CLI), `EpisodeOutcome`, `RunnerOutcome`, `ReloadFn<M>` | Landed `2026-05-20` (commit `b1cc7f8`) |
| Foundation | `config::RunnerConfig`, `manifest::ModelManifest`, `hot_reload::HotReloadWatcher`, `trajectory::TrajectoryWriter` | Landed PR #56 |

The Runner stitches the four foundation modules together with the
existing `forge-env::FlatObsEnv` trait and `forge-agent::latent_mcts::
LatentMctsSearch` to drive episodes end-to-end. Every interface is
**stable**: the foundation modules' public API is unchanged from PR #56;
the new Runner / binary / `ReloadFn` types are purely additive.

## Why the loop landed after the foundation

The runner-side and trainer-side wire formats stabilised first
(`ModelManifest`, `TrajectoryV2`) so `python/forge/training/muzero_mc/`
and the runner could be developed in parallel against a pinned
contract. The loop now consumes those contracts without modifying
them.

## Module-by-module contract

### RunnerConfig

- All fields `#[serde(default)]`. `Default` produces a smoke-run
  config that `validate()` accepts; production callers override
  `episodes`, `trajectory_dir`, `manifest_path`, `schema_id`.
- `validate()` rejects: empty `env_id` / `schema_id` / path fields;
  `action_repeat == 0` (would divide by zero in the runner loop).
- `runs_forever() == (episodes == 0)`; `metrics_disabled()
  == (metrics_port == 0)`. Caller gates Prometheus serving on the
  latter.

### ModelManifest

- `MANIFEST_SCHEMA_VERSION = 1` pinned. Any breaking change to the
  manifest shape requires bumping this constant — `load_json` rejects
  mismatched versions with `RunnerError::ManifestSchemaMismatch`.
- `validate()` runs **before** any IO in `save_json`; an invalid
  manifest never lands on disk.
- Atomic write: dotted `.tmp` sibling in the same directory, then
  `rename`. This matches the trainer's Python `exporter.py` discipline
  so the runner never observes a half-written manifest.
- `version` is the swap signal: trainer exports increment it
  monotonically; `HotReloadWatcher` rejects downgrades silently.
- `schema_id` is the cross-language sha256 that the runner must
  cross-check against the env handshake's `Hello.schema_id`.

### HotReloadWatcher

- **Contract:** poll only between episodes. The watcher doesn't
  enforce this — its doc comment does. Mid-episode polling breaks
  the lock-ordering story for `OnnxMuZeroModel::reload`
  (representation → dynamics → prediction).
- `poll()` returns `Ok(None)` when the manifest is missing — first
  start-up before trainer bootstrap is a non-error state.
- `prime_with(version)` lets the runner skip the initial event for an
  already-bootstrapped manifest.
- Lower-version manifests are ignored: the watcher refuses to
  downgrade `last_seen_version`. Surfaces a `RunnerError::Json` on a
  syntactically-bad manifest rather than hiding it as "no update".

### TrajectoryWriter

- Lifecycle: `start_episode → record_step* → finalize_and_save`.
  Wrong-order calls return `RunnerError::WriterState`.
- `record_step` forwards `TrajectoryV2::push` validation errors as
  `RunnerError::Trajectory` (obs-dim, policy-dim, action-range).
- Directory created on first `finalize_and_save` if missing;
  constructing a writer in a test never touches disk.
- Empty episodes are legal (0 steps recorded). The header invariants
  still hold; `final_reward = 0.0`.
- `discard_current()` aborts an in-progress episode without writing.
  Useful for tests and for cleanly skipping a partial episode the
  planner couldn't finish.

## Error model

A single `RunnerError` enum spans every failure mode:

| Variant | When |
|---|---|
| `MissingPath` | Configured path doesn't exist (currently unused — reserved for the runner main) |
| `ManifestSchemaMismatch` | `MANIFEST_SCHEMA_VERSION` drift |
| `InvalidManifest(String)` | Other manifest invariant violation |
| `WriterState(String)` | Out-of-order writer call |
| `Io { path, source }` | IO failure with the path that failed |
| `Json(serde_json::Error)` | JSON parse failure |
| `Trajectory(TrajectoryError)` | Forwarded from `forge-replay::v2` |

`io(path, source)` helper preserves the path so log messages always
identify the file that broke.

## Runner — the loop on top of the foundation

- `Runner<E: FlatObsEnv, M: LatentForwardModel>` owns the env, the
  `LatentMctsSearch`, the `TrajectoryWriter`, the `HotReloadWatcher`,
  the `ReloadFn` callback (optional), and two pre-allocated obs
  buffers swapped via `std::mem::swap` at the end of every step
  (zero per-step alloc on the hot path).
- `run_episode()` drives one episode: `writer.start_episode → env.reset_into → loop { search.search → env.step_into × action_repeat → writer.record_step → swap obs ↔ step_out.obs } → writer.finalize_and_save`. Visit counts are normalised
  (via the internal `normalize_visits` helper) into the policy target
  `StepV2.policy_target`; `LatentSearchResult.root_value` becomes the
  `value_target`. Runner-side truncation when `steps == max_steps`.
- `run(max_episodes)` polls the watcher at the top of every iteration
  (strictly between episodes, plan §3.4). On a `ReloadEvent` the
  runner takes `self.search.model_mut()` and invokes the installed
  `ReloadFn` — the borrow checker prevents this colliding with a
  live `search()` borrow because `search()` takes `&self`.
- `with_reload_fn(reload_fn)` is the builder hook. Without a callback
  the watcher still records `last_model_version` and increments
  `reloads_applied`, but the model is left untouched (useful for
  smoke testing or for envs whose models don't support reload, like
  `StubLatentModel` in tests).
- `prime_watcher_with(version)` suppresses the initial reload against
  an already-bootstrapped manifest.

## Binary — `forge-mc-runner`

- clap CLI: `--config <TOML>`, `--episodes <n>`, `--dry-run`,
  `--log-level`.
- `--dry-run` constructs an in-process stub env + `StubLatentModel`
  via the `dry_run::StubEnv` module so the loop can be smoke-tested
  without docker or a Minecraft server. The CI job
  `forge-mc-runner-bin` runs `--dry-run --episodes 1` on every push.
- Live wiring against `forge-env-mc::MinecraftEnv` and
  `OnnxMuZeroModel` is the next follow-up. Without it the binary
  exits with code 64 and a clear error.

## Additive `RunnerError` variants (loop-land)

| Variant | When |
|---|---|
| `Env(String)` | Wrapped `forge_env::Env::Error` from the underlying env (Display form — concrete type isn't nameable from this crate) |
| `Planner(String)` | Wrapped anyhow error from `LatentMctsSearch::search` |
| `Reload(String)` | The installed reload callback returned an error |

## What's deliberately still NOT here

- `OnnxMuZeroModel::reload()` impl — additive method on
  `crates/forge-agent/src/latent_mcts/onnx_model.rs` with fixed
  mutex-acquisition order (representation → dynamics → prediction).
  Lives in `forge-agent`, not here. Phase-4 deferred follow-up.
- Live `MinecraftEnv` + `OnnxMuZeroModel` wiring in `main.rs` — the
  binary's non-dry-run path currently exits 64. Will land alongside
  the ONNX reload impl.
- Prometheus `/metrics` HTTP endpoint — `metrics_port` is wired in
  `RunnerConfig` but the server itself ships with a separate
  Phase-6 follow-up.

## Test layout

- `src/runner.rs` — 13 unit tests (normalize_visits, run_episode,
  run, reload callback semantics, prime_watcher_with, no-callback
  recording, callback-error propagation, between-episode-poll
  contract).
- `src/config.rs|manifest.rs|hot_reload.rs|trajectory.rs` — per-module
  `#[cfg(test)] mod tests` blocks (42 tests, unchanged from PR #56).
- `tests/runner_integration.rs` — 3 end-to-end tests exercising the
  public API only: manifest bump v1 → v2 with trajectory round-trip,
  reload-callback error propagation, and zero-sim degenerate
  uniform-policy fallback.
- `tests/foundation_integration.rs` — 2 foundation integration tests
  unchanged from PR #56.

## Build & test

| Command | Purpose |
|---|---|
| `cargo build -p forge-mc-runner` | Build the crate (lib + bin) |
| `cargo test -p forge-mc-runner` | All unit + integration tests |
| `cargo clippy -p forge-mc-runner --all-targets -- -D warnings` | Lint |
| `cargo bench -p forge-bench --bench latent_mcts_inference` | Companion bench (lives in forge-bench) |
| `cargo run -p forge-mc-runner -- --dry-run --episodes 1` | Smoke-test the binary end-to-end without docker |

## Related docs

- `docs/architecture.md` §3.10.1 (foundation), §3.10.2 (runner loop +
  binary), §3.10.3 (Python `muzero_mc`), §3.10.4 (Phase-6
  orchestration) — C4-style component diagrams.
- `docs/plans/minecraft_rl_integration_plan_v2.md` Phase 4–6 — the
  full plan the runner is built against.
- `docs/next_steps.md` Phase 4 / 5 / 6 — what's landed vs. follow-up.
- `CHANGELOG.md` "Phase 4 — Runner<E,M> episode loop" + "Phase 5 —
  Python muzero_mc" + "Phase 6 — Compose stack + mc-bot CI" entries.
- `examples/minecraft/quickstart.md` — operator walkthrough.
