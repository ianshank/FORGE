# Agent.md — forge-mc-runner

## Persona

You are the **Runner Foundation** — the building blocks of the Phase 4
episode-runner crate for the FORGE Minecraft RL integration. You do
not yet ship a runnable `Runner<E, M>` loop; you ship the four
self-contained modules that the loop will assemble:

| Module | Role |
|---|---|
| `config::RunnerConfig` | TOML-driven knobs for the episode loop |
| `manifest::ModelManifest` | The `model_manifest.json` swap signal |
| `hot_reload::HotReloadWatcher` | Between-episode version-bump poller |
| `trajectory::TrajectoryWriter` | Episode-scoped `TrajectoryV2` writer |

Each module is independently testable; together they compose into the
foundation that `forge-mc-runner`'s eventual `main()` will glue into a
working `Runner`. Every interface here is **stable** — the upcoming
`Runner` PR adds new types but does not break the four below.

## Why the foundation ships before the loop

The runner-side and trainer-side wire formats need to stop diverging
before either side can integrate. Landing the manifest schema and
trajectory writer first lets `python/forge/training/muzero_mc/
exporter.py` and the future Rust runner be developed in parallel
against a pinned contract.

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

## What's deliberately NOT here

- `Runner<E: FlatObsEnv, M: LatentForwardModel>` — the actual episode
  loop. Follow-up PR; will compose the four modules above.
- `LatentPlanner` adapter around `LatentMctsSearch` — produces
  `(action, policy_target, value_target)` triples for `TrajectoryWriter`.
- `OnnxMuZeroModel::reload()` — additive method on
  `crates/forge-agent/src/latent_mcts/onnx_model.rs` with fixed
  mutex-acquisition order. Lives in `forge-agent`, not here.
- Prometheus `/metrics` HTTP endpoint — `metrics_port` is wired but
  the server itself ships with the runner main.

## Test layout

- `src/*.rs` — per-module `#[cfg(test)] mod tests` blocks (42 tests).
- `tests/foundation_integration.rs` — 2 end-to-end tests proving the
  four modules compose under the lifecycle the eventual `Runner` will
  drive (TOML load → two episodes → between-episode manifest bump →
  trajectory round-trip → schema_id drift detection).

## Build & test

| Command | Purpose |
|---|---|
| `cargo build -p forge-mc-runner` | Build the crate |
| `cargo test -p forge-mc-runner` | All unit + integration tests |
| `cargo clippy -p forge-mc-runner --all-targets -- -D warnings` | Lint |
| `cargo bench -p forge-bench --bench latent_mcts_inference` | Companion bench (lives in forge-bench) |

## Related docs

- `docs/architecture.md` §3.10.1 — C4-style component diagram for
  the runner foundation.
- `docs/plans/minecraft_rl_integration_plan_v2.md` Phase 4 — the
  full plan the foundation is built against.
- `docs/next_steps.md` Phase 4 — what's landed vs. follow-up.
- `CHANGELOG.md` "Phase 4 foundation" entry — per-module summary.
