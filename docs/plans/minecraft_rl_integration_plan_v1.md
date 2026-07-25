# FORGE Minecraft RL Integration — Implementation Plan v1

**Date:** 2026-05-17
**Branch:** `claude/minecraft-rl-agent-integration-xnJjt`
**Status:** Draft for review, not yet implemented

---

## 0. Preamble

### Goal
Plug FORGE's `latent_mcts` planner into a real Minecraft instance, watchable in a browser, with a Python trainer that consumes Minecraft trajectories and produces ONNX weights the Rust planner hot-reloads. Introduce a reusable `Env` trait so future environments compose cleanly. Preserve every existing FORGE workflow without behavioural drift.

### How this plan was produced
Two parallel subagent passes:

1. **Audit pass** (Explore agent) — characterised coupling in `forge-agent` and `forge-python`, then filled gap-analysis blanks on `forge-replay`, `forge-types` schemas, `forge-integration`, `forge-cognitive`, testing conventions, logging patterns, config patterns, PyO3 idioms, ONNX loader layout, and existing benchmark coverage.
2. **Architecture pass** (Plan agent) — designed the phase breakdown given the audited ground truth.

Audit findings of note that shape this plan:
- `forge-replay::Trajectory` is closer to env-agnostic than expected (`Vec<Observation>` + `Vec<u32>` + `Vec<f32>` + flags), but still types `Observation` from `forge-types`. A new `v2` flat-tensor variant is needed.
- `forge-agent::latent_mcts` is genuinely env-free; only `&[f32]` and `Vec<f32>` cross its API. Zero changes required to its core.
- `OnnxMuZeroModel` uses `ort` v2.0.0-rc.9 with `Mutex<Session>` per net — three files on disk, loaded via `Session::builder().commit_from_file(path)`. Hot-reload is feasible via an additive `reload()` method.
- Logging convention: `#[tracing::instrument(skip_all)]` on functions with allocation-heavy args; `tracing::debug!`/`warn!` macros; spans labelled at function granularity.
- Config convention: `#[derive(Debug, Clone, Default, Serialize, Deserialize)]` with `#[serde(default)]`; nested sub-configs; defaults in `impl Default`.
- Testing convention: bare `#[test]` for units, `proptest!` blocks for properties, `criterion` for benchmarks. No tokio/rstest/async_test.
- **Benchmark gap noticed:** `forge-bench` has no `latent_mcts` inference latency benchmark. This plan adds one in Phase 4.

### Hard constraints (CLAUDE.md compliance)
- **No hardcoded values.** Every numeric is a documented `Default`-impl config field.
- **Zero alloc on hot path.** New code obeys the existing `step_into` contract; CI gate at `crates/forge-bench/src/bin/allocation_audit.rs` extended where applicable.
- **Deterministic.** Same seed + same actions ⇒ identical state. Existing PRNG choices (`rand_pcg`) reused.
- **Structured logging.** `tracing` only, never `println!`. `#[instrument]` on public functions.
- **Property tests** alongside unit tests for invariants.
- **`thiserror` derive** for error types; `anyhow` for binaries.
- **Backwards compatibility.** Existing FORGE training/eval flows must keep passing every test unchanged.

---

## 1. Convention Adherence Checklist

Each new crate enforces these without exception. Items map 1:1 to audited patterns.

| Concern | Pattern | Source of pattern |
|---|---|---|
| Logging | `#[tracing::instrument(skip_all, fields(...))]` on public fns; `debug!`/`warn!`/`error!` for events | `forge-replay/trajectory.rs:66,117,175` |
| Errors | `#[derive(thiserror::Error, Debug)]` for libs; `anyhow::Result` only in `bin/` | `forge-agent` and root-crate convention |
| Config | `#[derive(Debug, Clone, Default, Serialize, Deserialize)]` + `#[serde(default)]`; defaults in `impl Default` | `forge-types/config.rs:26-51` |
| Tests | `#[test]` unit; `proptest! { #[test] ... }` properties; `criterion` benches | `forge-replay/trajectory.rs:323-371`, `forge-bench/benches/step_throughput.rs` |
| PyO3 | `#[pyclass]` + `#[pymethods]` + `py.allow_threads(\|\| ...)` around Rust compute; numpy via `numpy::PyArray*` | `forge-python/src/env.rs` |
| ONNX | `ort` v2.x; `Mutex<Session>` per net; load via `Session::builder().commit_from_file(path)` | `forge-agent/src/latent_mcts/onnx_model.rs:81-101` |
| Workspace registration | Add new crate to `crates/` + workspace `members` + `[workspace.dependencies]` | Root `Cargo.toml` |
| Public doc comments | Every public item documented (CLAUDE.md hard rule) | CLAUDE.md |

---

## 2. Architecture at a Glance

```
                         +-----------------------------------+
                         |        forge-env (NEW crate)      |
                         |   Env, ObsSpec, ActionSpec, ...   |
                         +-----------------------------------+
                                |                |
        +-----------------------+                +-----------------------+
        |                                                                |
+-------v---------+                                            +---------v--------+
| forge-env-forge |  (NEW, backwards-compat shim)              | forge-env-mc     |  (NEW)
| wraps WorldState|                                            | WS client to Node|
+-----------------+                                            +---------+--------+
                                                                         |
                                                            WebSocket    |
                                                            (JSON+bin)   v
                                                          +--------------+----------+
                                                          | forge-mc-bot (Node)     |
                                                          | mineflayer + viewer     |
                                                          +-------------------------+

forge-agent::latent_mcts (UNCHANGED core)
        ^
        |  drives via Env trait through new LatentPlanner adapter
        |
+-------+---------------+      reload weights      +-----------------------+
| forge-mc-runner (bin) | <----------------------- | python/forge/training |
| Rust planning loop    |   ONNX files + manifest  | (MuZero trainer, ext.)|
+-----------------------+                          +-----------------------+
                |                                                    ^
                |  trajectories (TrajectoryV2)                       |
                +----------------------------------------------------+
                           (file replay buffer)
```

Every cross-component wire is versioned:
- WebSocket: `SCHEMA_VERSION` constant in `protocol.rs`; `Hello` handshake aborts on mismatch.
- Trajectory: `TRAJECTORY_FORMAT_VERSION = 2` pinned in `forge-replay::v2`.
- ONNX bundles: `model_manifest.json` with monotonic `version`, content-hashed `schema_id`, sha256 of each net.

---

## 3. Phase Breakdown

Six phases, each independently shippable & testable. Phase N never breaks Phase N-1.

### Phase 1 — `forge-env` crate (generic Env trait)

**New crate:** `crates/forge-env/`

Files: `Cargo.toml`, `src/lib.rs`, `src/env.rs`, `src/spec.rs`, `src/spaces.rs`, `src/error.rs`, `tests/env_trait.rs`.

**Modified files:**
- `Cargo.toml` (workspace root): add `crates/forge-env` to members and `[workspace.dependencies]`.
- `crates/forge-agent/Cargo.toml`: optional dev-dep on `forge-env` for an integration test demonstrating `latent_mcts` driving an `Env`. Why: prove the trait composes with the existing planner without modifying `latent_mcts` itself.

**Public API surface (final form):**

```rust
// crates/forge-env/src/env.rs
pub trait Env: Send {
    type Obs;
    type Action: Copy + Send;
    type Info: Default + Send;
    type Error: std::error::Error + Send + Sync + 'static;

    fn reset(&mut self, seed: Option<u64>) -> Result<StepOutput<Self>, Self::Error>;
    fn step(&mut self, action: Self::Action) -> Result<StepOutput<Self>, Self::Error>;
    fn obs_spec(&self) -> &ObsSpec;
    fn action_spec(&self) -> &ActionSpec;
    fn name(&self) -> &'static str { "env" }
    fn close(&mut self) -> Result<(), Self::Error> { Ok(()) }
}

pub struct StepOutput<E: Env + ?Sized> {
    pub obs: E::Obs,
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
    pub info: E::Info,
}

/// Convenience marker for envs whose obs is a flat float vector and whose
/// actions are a discrete index. This is the surface `latent_mcts` consumes.
pub trait FlatObsEnv: Env<Obs = Vec<f32>, Action = u32> {
    fn obs_dim(&self) -> usize;
    fn num_actions(&self) -> u32;
}
```

```rust
// crates/forge-env/src/spec.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObsSpec {
    pub shape: Vec<usize>,
    pub low: f32,
    pub high: f32,
    pub dtype: DType,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionSpec {
    Discrete { n: u32, labels: Option<Vec<String>> },
    MultiDiscrete { nvec: Vec<u32> },
    Box { low: Vec<f32>, high: Vec<f32> },
}
```

Design choices:
- **Associated types, not generics.** Keeps `Box<dyn FlatObsEnv>` viable for downstream code.
- **`Info: Default`** so `StepOutput` is ergonomic to construct.
- **`Error: std::error::Error + Send + Sync + 'static`** interops with `anyhow`.
- **No `Clone` on `Env`.** Preserves wire-bound `MinecraftEnv` semantics (TCP sockets aren't cloneable). Planners needing rollouts use `ForwardModel` or latent MCTS, neither of which requires env cloning.
- **`obs_spec`/`action_spec` return `&`** — zero-alloc on accessor calls.

**Tests:**
- Unit: `env_spec_serde_roundtrip`, `flat_obs_env_via_dyn`, `step_output_default_info`.
- Property (`proptest!`): `discrete_action_in_range`, `obs_shape_product_matches`.
- Integration `tests/env_trait.rs`: a `MockFlatEnv` driven through `latent_mcts::LatentMctsSearch` with `StubLatentModel`, asserting planner can `search()` against its observations.

**Logging spans:** Trait-method impls add `#[tracing::instrument(skip_all, fields(env = self.name()))]`. `name()` default returns `"env"` static str for cheap span labelling without allocation.

**Config knobs added:** None (pure trait crate).

**Rollback:** Delete the crate, revert workspace `Cargo.toml`. Nothing references it yet.

---

### Phase 2 — Backwards-compat shim for FORGE's `WorldState`

**New crate:** `crates/forge-env-forge/`

Files: `Cargo.toml`, `src/lib.rs`, `src/world_env.rs`, `src/flat_adapter.rs`, `tests/forge_env_parity.rs`.

**Modified files:**
- `crates/forge-python/src/env.rs`: untouched at runtime. Add a `#[cfg(test)]` parity test that drives `ForgeEnv` and `WorldEnv` in lockstep and asserts equal rewards/terminated/truncated.
- `crates/forge-agent/src/forward_model.rs`: untouched. The shim is a parallel path, not a replacement.

**Public API surface:**

```rust
// crates/forge-env-forge/src/world_env.rs
pub struct WorldEnv {
    state: forge_core::WorldState,
    config: forge_types::config::ForgeConfig,
    obs_spec: ObsSpec,
    action_spec: ActionSpec,
}

impl Env for WorldEnv {
    type Obs = forge_types::observation::Observation;
    type Action = forge_types::Action;
    type Info = forge_types::observation::StepInfo;
    type Error = ForgeEnvError;
    // reset/step delegate to WorldState; action_spec built from
    // Action::space_size_full(comm_vocab, drone, agri, hex) at construction.
}

// crates/forge-env-forge/src/flat_adapter.rs
pub struct FlatForgeEnv {
    inner: WorldEnv,
    flattener: ObsFlattener,
}
impl FlatObsEnv for FlatForgeEnv { /* obs_dim = product(obs_spec.shape) */ }
```

`ObsFlattener` is config-driven via `FlatObsConfig`. Every observation field that becomes a float is toggle-able. Action-space size is **always** sourced from `Action::space_size_full()` — never a literal in this crate.

**Tests:**
- Unit: `flattener_dim_matches_spec`, `flattener_round_trip_preserves_inventory`, `world_env_action_space_matches_space_size_full`.
- Integration `tests/forge_env_parity.rs`: 1k deterministic steps comparing `WorldEnv` to direct `WorldState` rewards/term flags.
- Property: `seeded_reset_is_idempotent`, `flatten_dim_invariant_across_observations`.

**Logging spans:** `WorldEnv::step` with `tick`, `agent_count`. `FlatForgeEnv::step` adds `obs_dim`.

**Config knobs added (`forge-types::config` extension or local sub-config):**
```rust
pub struct FlatObsConfig {
    pub grid_radius: u32,
    pub inventory_slots: u32,
    pub include_battery: bool,
    pub include_morphology: bool,
    pub include_task_progress: bool,
    pub normalize: bool,
}
```
All defaults pulled from `ForgeConfig`; never hardcoded inline.

**Rollback:** Delete the crate. `forge-python` and `forge-agent` untouched. Existing PPO/SAC/Gym flows keep working.

---

### Phase 3 — Mineflayer bot + WebSocket protocol

**New JS package:** `mc-bot/` at repo root (out of Cargo workspace).

Files: `mc-bot/package.json`, `mc-bot/src/index.js`, `mc-bot/src/protocol.js`, `mc-bot/src/observation.js`, `mc-bot/src/action_map.js`, `mc-bot/src/viewer.js`, `mc-bot/test/protocol.test.js`, `mc-bot/test/action_map.test.js`, `mc-bot/test/fixtures/*.json`, `mc-bot/README.md`.

**New crate:** `crates/forge-env-mc/`

Files: `Cargo.toml`, `src/lib.rs`, `src/protocol.rs`, `src/client.rs`, `src/mc_env.rs`, `src/error.rs`, `src/config.rs`, `tests/protocol_schema.rs`, `tests/mc_env_mock.rs`.

**Modified files:**
- Root `Cargo.toml`: add `forge-env-mc` member.
- `docker/`: placeholder note for `docker/mc-bot.Dockerfile` — Phase 6 ops work, not authored here.

**Public API surface:**

```rust
// crates/forge-env-mc/src/mc_env.rs
pub struct MinecraftEnv {
    client: ProtocolClient,
    config: MinecraftEnvConfig,
    obs_spec: ObsSpec,
    action_spec: ActionSpec,
    action_map: ActionMap,
    last_obs: Option<Vec<f32>>,
}

impl Env for MinecraftEnv {
    type Obs = Vec<f32>;
    type Action = u32;
    type Info = MinecraftStepInfo;
    type Error = McEnvError;
}
impl FlatObsEnv for MinecraftEnv { /* */ }

// crates/forge-env-mc/src/protocol.rs
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientMsg {
    Reset { seed: Option<u64> },
    Step { action_id: u32 },
    Close,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMsg {
    Hello { schema_version: u32, action_count: u32, obs_dim: usize },
    Observation {
        tick: u64,
        obs: Vec<f32>,
        reward: f32,
        terminated: bool,
        truncated: bool,
        info: serde_json::Value,
    },
    Error { code: String, message: String },
}

pub const SCHEMA_VERSION: u32 = 1;
```

Action map loaded from `configs/minecraft/action_map.toml`. Example entries:
```toml
[[action]]
id = 0
kind = "noop"
ticks = 1

[[action]]
id = 1
kind = "move"
direction = "forward"
ticks = 4

[[action]]
id = 12
kind = "place"
hotbar_slot = 0
```
No discrete-action ids anywhere in Rust or JS source code.

**Tests:**
- Unit (Rust): `protocol_serde_roundtrip`, `hello_handshake_version_match`, `action_map_loads_from_toml`, `obs_dim_from_hello`, `schema_id_stable_across_loads`.
- Unit (JS, `node:test` or `vitest`): `clientMsg_parses_reset_step_close`, `observation_builder_deterministic_per_tick`, `actionMap_rejects_unknown_id`, `schema_id_matches_rust_fixture`.
- Integration `tests/mc_env_mock.rs`: mock WS server speaks `ServerMsg`, drives `MinecraftEnv::reset/step` for 50 steps, asserts obs dimensions match spec and no panics.
- E2E smoke `tests/e2e_mc_smoke.rs` (`#[ignore]`, run with `--ignored`): boots `mc-bot` against `flying-squid` or `mineflayer-mock-server` (no Minecraft binary needed), confirms one full episode.

**Logging spans:**
- `ProtocolClient::send` (`msg_kind`)
- `ProtocolClient::recv` (`tick`, `bytes`)
- `MinecraftEnv::step` (`action_id`, `reward`)
- `ActionMap::load` (`path`, `entry_count`, `schema_id`)

**Config knobs added (`configs/minecraft/env.toml`):**
```toml
[server]
host = "127.0.0.1"
port = 25565

[bot]
username = "ForgeBot"
auth = "offline"
version = "1.20.4"

[viewer]
enabled = true
port = 3000

[episode]
max_ticks = 6000
tick_rate_hz = 20
action_repeat = 4

[observation]
radius = 8
include_inventory = true
include_biome = true
flatten_dim = 960  # validated against Hello.obs_dim at handshake

[protocol]
schema_version = 1
heartbeat_ms = 2000
reconnect_backoff_ms = [500, 1000, 2000, 4000, 8000]
obs_binary_threshold = 256
```

**Rollback:** Delete `forge-env-mc`, `mc-bot/`, workspace edit. Nothing else regresses.

---

### Phase 4 — Rust runner: `latent_mcts` driving Minecraft

**New crate:** `crates/forge-mc-runner/`

Files: `Cargo.toml`, `src/main.rs`, `src/runner.rs`, `src/planner.rs`, `src/trajectory_writer.rs`, `src/hot_reload.rs`, `src/config.rs`, `tests/runner_smoke.rs`.

**New benchmark:** `crates/forge-bench/benches/latent_mcts_inference.rs` — fills the gap noted in audit (no existing latent_mcts benchmark). Measures per-decision latency for `LatentMctsSearch::search` at varying simulation budgets (1, 8, 25, 50, 100, 200) using `StubLatentModel`, plus an `OnnxMuZeroModel` variant gated on a `bench-onnx` feature.

**Modified files:**
- `crates/forge-agent/src/latent_mcts/onnx_model.rs`: add **only** the additive method
  ```rust
  pub fn reload(&self, new_paths: &OnnxModelConfig) -> Result<(), ort::Error>;
  ```
  Implementation: build three new `Session` instances first (failing without touching live state), then briefly acquire all three `Mutex<Session>` locks and `std::mem::replace` each. No signature breaks; no behavioural change for callers that never call `reload`.
- `crates/forge-agent/src/latent_mcts/mod.rs`: re-export the new symbol.
- `crates/forge-bench/Cargo.toml`: register the new bench, add optional `bench-onnx` feature.

**Public API surface:**

```rust
// crates/forge-mc-runner/src/planner.rs
pub struct LatentPlanner<M: LatentForwardModel> {
    search: LatentMctsSearch<M>,
    config: LatentMctsConfig,
}
impl<M: LatentForwardModel> LatentPlanner<M> {
    pub fn act<E: FlatObsEnv>(&self, env: &E, obs: &[f32]) -> anyhow::Result<ActDecision>;
}

pub struct ActDecision {
    pub action_id: u32,
    pub root_value: f32,
    pub policy_target: Vec<f32>, // normalised visit counts; consumed by trainer
}

// crates/forge-mc-runner/src/runner.rs
pub struct Runner<E: FlatObsEnv> {
    env: E,
    planner: LatentPlanner<OnnxMuZeroModel>,
    writer: TrajectoryWriter,
    hot_reload: HotReloadWatcher,
    config: RunnerConfig,
}
impl<E: FlatObsEnv> Runner<E> {
    pub fn run_episode(&mut self, seed: Option<u64>) -> anyhow::Result<EpisodeStats>;
    pub fn run(&mut self, max_episodes: u64) -> anyhow::Result<()>;
}

// crates/forge-mc-runner/src/hot_reload.rs
pub struct HotReloadWatcher { /* watches model_manifest.json mtime + version */ }
impl HotReloadWatcher {
    pub fn poll(&mut self) -> Option<OnnxModelConfig>; // Some when manifest version increments AND schema_id matches
}
```

**Tests:**
- Unit: `planner_returns_action_in_range`, `hot_reload_triggers_on_manifest_bump`, `hot_reload_rejects_schema_id_mismatch`, `trajectory_writer_appends_atomically`.
- Integration `tests/runner_smoke.rs`: `Runner` against `MockFlatEnv` for 200 ticks with `StubLatentModel`; asserts trajectories written, no leaks, deterministic with same seed.
- E2E `tests/e2e_runner.rs` (`#[ignore]`): real `forge-env-mc` + mock WS server + real `OnnxMuZeroModel` loaded from small fixture nets exported from a synthetic PyTorch model.

**Logging spans:**
- `Runner::run_episode` (`episode`, `seed`, `total_reward`, `steps`)
- `LatentPlanner::act` (`obs_dim`, `chosen_action`, `root_value`)
- `HotReloadWatcher::poll` (`manifest_version`, `schema_id`)
- `OnnxMuZeroModel::reload` (`old_version`, `new_version`)

**Config knobs added (`configs/minecraft/runner.toml`):**
```toml
[mcts]
num_simulations = 50
c_puct = 1.25
dirichlet_alpha = 0.3
dirichlet_epsilon = 0.25
discount = 0.997

[model]
representation_path = "models/representation.onnx"
dynamics_path      = "models/dynamics.onnx"
prediction_path    = "models/prediction.onnx"
latent_dim = 256
action_space_size = 64  # cross-checked against Hello.action_count

[hot_reload]
manifest_path = "models/model_manifest.json"
poll_interval_ms = 5000

[runner]
max_episodes = 0       # 0 = unbounded
replay_dir = "replays/minecraft"
flush_every_n_steps = 200
```

**Rollback:** Delete the crate, revert the `reload()` addition + bench. Phase 3 still runs (env usable from Python or examples).

---

### Phase 5 — Python trainer + replay format v2

**New module:** `python/forge/training/muzero_mc/`

Files: `__init__.py`, `trainer.py`, `replay.py`, `exporter.py`, `manifest.py`, `cli.py`.

**New tests:** `tests/python/training/test_muzero_mc_trainer.py`, `tests/python/training/test_replay_v2.py`, `tests/python/training/test_onnx_exporter.py`.

**New Rust module:** `crates/forge-replay/src/v2.rs` — additive within the existing crate. Keeps replay format ownership in one place; v1 stays.

**Modified files:**
- `crates/forge-replay/src/lib.rs`: add `pub mod v2;`.
- `crates/forge-replay/src/trajectory.rs`: untouched — v1 remains default.
- `python/forge/training/__init__.py`: register `muzero_mc` subpackage.
- Root `pyproject.toml`: add `[project.optional-dependencies] minecraft = ["onnx>=1.16", "onnxruntime>=1.18", "torch>=2.3", "onnxsim>=0.4"]`.

**Public API surface:**

```rust
// crates/forge-replay/src/v2.rs
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TrajectoryV2 {
    pub format_version: u32,           // = TRAJECTORY_FORMAT_VERSION
    pub env_id: String,                // "minecraft", "forge", ...
    pub schema_id: String,             // sha256 of (obs_dim, action_count, action_map_canonical)
    pub episode_id: String,            // ulid
    pub seed: Option<u64>,
    pub steps: Vec<StepV2>,
    pub final_reward: f32,
    pub started_at: String,            // RFC3339
    pub ended_at: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StepV2 {
    pub tick: u64,
    pub obs: Vec<f32>,
    pub action_id: u32,
    pub policy_target: Vec<f32>,       // visit-count distribution from MCTS
    pub value_target: f32,
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
}

pub const TRAJECTORY_FORMAT_VERSION: u32 = 2;

/// Convert a legacy v1 trajectory into v2 (loses some FORGE-specific obs structure
/// — caller must supply a flattener). Used only for migration tooling.
pub fn from_v1(t: &Trajectory, flatten: &dyn Fn(&Observation) -> Vec<f32>) -> TrajectoryV2;
```

```python
# python/forge/training/muzero_mc/trainer.py
@dataclass
class MuZeroMcTrainerConfig:
    replay_dir: pathlib.Path
    out_dir: pathlib.Path
    manifest_path: pathlib.Path
    batch_size: int
    unroll_steps: int
    n_step_return: int
    learning_rate: float
    weight_decay: float
    target_update_interval: int
    onnx_opset: int
    obs_dim: int
    action_count: int
    latent_dim: int
    schema_id: str
    min_episodes_before_publish: int
    compare_against_previous: bool

class MuZeroMcTrainer:
    def __init__(self, cfg: MuZeroMcTrainerConfig): ...
    def step(self) -> dict: ...        # one optimizer step, returns metrics
    def export(self) -> ModelManifest: ...

# python/forge/training/muzero_mc/manifest.py
@dataclass
class ModelManifest:
    version: int
    schema_id: str
    representation: str
    dynamics: str
    prediction: str
    obs_dim: int
    action_count: int
    latent_dim: int
    exported_at: str
    sha256: dict[str, str]
```

**Tests:**
- Rust unit: `trajectoryv2_serde`, `trajectoryv2_format_version_pinned`, `schema_id_stable`, `from_v1_preserves_rewards`.
- Python unit: `test_replay_iterator_yields_unrolls`, `test_unroll_value_target_matches_bootstrap`, `test_onnx_export_passes_onnxsim_and_inferencesession`, `test_manifest_version_monotonic`, `test_trainer_step_decreases_loss_on_overfit_batch`.
- E2E `tests/e2e_train_loop.py`: trainer runs 5 steps on synthetic `TrajectoryV2` dump and exports a valid ONNX bundle that `OnnxMuZeroModel::load` (via a small PyO3 test helper exposed under `cfg(test)`) accepts.

**Logging spans:** Python via existing `forge.utils.logging`. Rust v2 module uses `#[instrument(skip_all, fields(format_version, schema_id))]` on `read_dir`, `append`, `validate_schema`.

**Config knobs added (`configs/minecraft/training.toml`):**
```toml
batch_size = 256
unroll_steps = 5
n_step_return = 10
learning_rate = 1e-3
weight_decay = 1e-4
target_update_interval = 100
onnx_opset = 17
obs_dim = 960
action_count = 64
latent_dim = 256
schema_id = "mc-v1-obs960-act64"

[export]
compare_against_previous = true
min_episodes_before_publish = 50
```

**Rollback:** Delete `python/forge/training/muzero_mc/`, revert `forge-replay::lib.rs` + drop `v2.rs`. v1 replays untouched. Phase 4 runner keeps running with the bootstrap ONNX.

---

### Phase 6 — End-to-end glue, browser viewer wiring, ops scripts

**New files:**
- `scripts/mc_run.sh` — orchestrates `docker compose up mc-bot`, `cargo run -p forge-mc-runner`, `python -m forge.training.muzero_mc.cli`.
- `dashboard/src/minecraft_panel/` — viewer-embed panel iframing prismarine-viewer on `viewer.port`; live runner-metrics WebSocket connection.
- `examples/minecraft/quickstart.md` — 10-minute getting-started runbook.
- `tests/e2e_full_loop.py` — full vertical slice: bot up, runner steps, trainer trains, runner reloads.
- `docker/mc-bot.Dockerfile` — Node 20 base, installs mineflayer + prismarine-viewer.
- `docker/compose.minecraft.yml` — composes mc-bot + a Paper Minecraft server.

**Modified files:**
- `docs/minecraft.md` (new) — architecture + ops runbook.
- `CHANGELOG.md` — append integration line.
- `.github/workflows/`: new job runs unit tests for new crates + JS package. E2E `#[ignore]`d in PR CI; gated on manual `workflow_dispatch`.

**Public API surface:** none new. Wiring only.

**Tests:** `tests/e2e_full_loop.py` (`@pytest.mark.skip` in PR CI, nightly job). Asserts a 3-episode loop ending with `model_manifest.version == 1` reload taking effect.

**Logging spans:** none new.

**Config knobs added (`configs/minecraft/orchestration.toml`):** paths, ports, container image tags.

**Rollback:** Drop the scripts and dashboard panel. Crates from prior phases remain usable individually.

---

## 4. Cross-Cutting Concerns

### 4.1 Backwards compatibility

- `forge-python::ForgeEnv` (Python pyclass) — **byte-identical** runtime; only `#[cfg(test)]` additions for parity testing.
- `forge-agent::mcts` (classical) — **untouched.** Out of scope for env-trait refactor. Coexists with `Env` indefinitely; `ForwardModel` and `Env` are complementary, not competing.
- `forge-replay::Trajectory` (v1) — **untouched.** v2 lives alongside.
- `OnnxMuZeroModel` — gains `reload()`; existing constructor + `infer*` methods unchanged.

CI gate `crates/forge-env-forge/tests/forge_env_parity.rs` runs 1k deterministic steps via both `ForgeEnv` and `WorldEnv`; identical rewards/terminated/truncated required.

### 4.2 Replay buffer format (env-agnostic, versioned)

- `TRAJECTORY_FORMAT_VERSION = 2` is the canonical going-forward.
- File layout: JSONL, one episode per file at `replay_dir/<env_id>/<YYYY-MM-DD>/<episode_id>.jsonl`.
- Header line (first JSONL record) carries `{"format_version": 2, "schema_id": "...", "env_id": "minecraft"}` so readers fail fast.
- `schema_id` = sha256 of canonical-form `(obs_dim, action_count, action_map_entries)`. Recomputed at runner startup AND trainer startup; mismatch raises a hard error before any work happens.
- v1 → v2 converter (`forge-replay::v2::from_v1`) supplied for one-time migrations only; not part of the runtime path.

### 4.3 Hot-reload protocol for ONNX weights

Single `model_manifest.json` is source of truth:

```json
{
  "version": 7,
  "schema_id": "mc-v1-obs960-act64",
  "representation": "models/rep_v7.onnx",
  "dynamics":       "models/dyn_v7.onnx",
  "prediction":     "models/pred_v7.onnx",
  "obs_dim": 960,
  "action_count": 64,
  "latent_dim": 256,
  "exported_at": "2026-05-17T12:00:00Z",
  "sha256": {
    "representation": "...",
    "dynamics":       "...",
    "prediction":     "..."
  }
}
```

`HotReloadWatcher::poll`:
1. Stat manifest; if mtime unchanged → `None`.
2. Parse manifest, compare `version` to last loaded.
3. If higher: verify sha256 of each ONNX file.
4. Verify `schema_id` matches runner's compiled-in (or config-loaded) value; mismatch → `tracing::error!` and refuse reload.
5. Call `OnnxMuZeroModel::reload`.
6. **Reload happens between episodes only**, never mid-search, to eliminate latent-dim mismatch surprises.

Trainer-side `manifest.py` enforces atomic write (tmp file + rename) and monotonically increasing version.

### 4.4 WebSocket protocol contract (Node bot ↔ Rust adapter)

Transport: WebSocket on `tcp://{host}:{port}`. JSON text frames for control; binary frames carrying little-endian f32 arrays when `obs_dim > obs_binary_threshold` (config knob, default 256).

Handshake:
1. Client (Rust) opens connection.
2. Server (Node) sends `Hello { schema_version, action_count, obs_dim }`.
3. Client compares to compiled `SCHEMA_VERSION`. Mismatch → close + error.
4. Client sends `Reset { seed }`.

Per-tick:
- Client → `Step { action_id }`
- Server → `Observation { tick, obs, reward, terminated, truncated, info }`

Error paths:
- Server `Error { code, message }` ends the episode; runner records `truncated=true`, increments `mc_protocol_error_total` (tracing-derived metric).
- Heartbeat: client expects at least one `Observation` per `protocol.heartbeat_ms`; otherwise reconnect with exponential backoff per `protocol.reconnect_backoff_ms`.

Versioning: `SCHEMA_VERSION` ratchets only on incompatible changes. Backwards-compatible additions (new optional `info` fields) allowed within a major version.

Snapshot test (`tests/protocol_schema.rs`) serialises one of each variant and asserts bytes match a golden file under `tests/fixtures/protocol/`. JS unit test parses the same fixture, catching schema drift in code review.

---

## 5. Gap Analysis

| Capability | Where it lives today | What's missing | Where it'll live |
|---|---|---|---|
| Generic env abstraction | None (concrete `WorldState`) | Trait + spec types | `forge-env` (Phase 1) |
| FORGE WorldState as generic Env | `forge-python::ForgeEnv` only | Native Rust `Env` impl | `forge-env-forge::WorldEnv` (Phase 2) |
| Minecraft connection (game side) | None | Mineflayer bot + viewer | `mc-bot/` (Phase 3) |
| Minecraft env (Rust client) | None | WS protocol + `Env` impl | `forge-env-mc` (Phase 3) |
| Latent MCTS planner | `forge-agent::latent_mcts` ✅ | — | Reused as-is |
| ONNX inference | `latent_mcts::onnx_model::OnnxMuZeroModel` ✅ | Hot-reload swap | Add `reload()` (Phase 4) |
| Runner loop (env + planner + replay) | None | Binary that ties them | `forge-mc-runner` (Phase 4) |
| Trajectory format env-agnostic | `forge-replay::Trajectory` (FORGE-typed `Observation`) | `Vec<f32>` obs variant + format_version | `forge-replay::v2` (Phase 5) |
| MuZero training from MC trajectories | `python/forge/training/muzero_trainer.py` (WorldState-typed) | MC-typed trainer over `TrajectoryV2` | `python/forge/training/muzero_mc/` (Phase 5) |
| ONNX export from trained nets | `python/forge/agents/muzero_agent.py` (FORGE-specific) | Reusable exporter with manifest | `muzero_mc/exporter.py` (Phase 5) |
| Browser viewer | None | Prismarine viewer iframe | `mc-bot/src/viewer.js` + dashboard panel (Phase 3 + 6) |
| Ops orchestration | None | docker-compose + scripts | Phase 6 |
| CI gating on new crates | Existing workflows FORGE-only | New crate jobs + JS lint/test | Phase 6 |
| **Latent MCTS inference benchmark** | **None — gap identified in audit** | Criterion bench for `search()` at varying sim budgets | `forge-bench/benches/latent_mcts_inference.rs` (Phase 4) |

---

## 6. Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | **Tick-rate mismatch** — Minecraft @ 20 TPS vs MCTS planning ~100ms/move | High | Med | `episode.action_repeat` config; server holds last action N ticks until next `Step`. Runner emits `mc_planning_latency_ms`; alarm if > tick budget. Test in `runner_smoke` asserts no queued outstanding action. |
| 2 | **Action-space drift** between Node `action_map.toml`, Rust constants, Python trainer | High | High | Single source of truth: `configs/minecraft/action_map.toml`. Rust + Node both load it at boot, both compute `schema_id` (sha256 of canonical form). Runner refuses to start on mismatch. Trainer refuses to publish if `schema_id` differs from one in replay buffer. |
| 3 | **ONNX export pitfalls** — missing ops in onnxruntime, dynamic-axis surprises | Med | High | Exporter pins `opset=17`; runs `onnxsim` + `onnxruntime.InferenceSession` smoke before writing manifest. CI test `test_onnx_export_passes_onnxsim_and_inferencesession`. |
| 4 | **Replay format churn** as policy/value-target encoding evolves | High | Med | `format_version` pinned; bumping requires converter + deprecation notice. Trainer reads `format_version` first; errors on unknown. v1 untouched. |
| 5 | **JS↔Rust schema drift** in WS protocol | Med | High | Golden-file snapshot test (`tests/protocol_schema.rs`) + JS unit test against the same JSON fixture from `mc-bot/test/fixtures/`. Both must pass in CI. `Hello` handshake aborts on `SCHEMA_VERSION` mismatch. |
| 6 | **Hot-reload race** — manifest swapped mid-search → wrong-dim latent | Med | High | Reload only between episodes. `OnnxMuZeroModel::reload` builds new sessions first (failure-isolated), then briefly acquires all three mutexes to swap. `schema_id` check refuses dim changes. Phase 4 unit test `hot_reload_rejects_schema_id_mismatch`. |

---

## 7. Out of Scope

Explicitly **not** addressed by this plan:

- Refactoring `forge-agent::mcts` (the `WorldState`-coupled classical MCTS) onto the new `Env` trait.
- Training MuZero **from scratch** on Minecraft. Plan ships bootstrap ONNX (random-init or small-dataset BC).
- Pixel observations. Phase 3 obs is structured (block ids in radius, inventory, hp/food, biome, time). A future `forge-env-mc::pixel_obs` module can extend.
- Multi-agent Minecraft. One bot per runner in v1. `MultiBotEnv` wrapper is future work.
- Mojang online-auth. v1 targets `offline-mode=true` LAN servers.
- Replacing existing FORGE PPO/SAC trainers. Phase 5 trainer is additive.
- Distributed self-play (parallel runners feeding one replay). Single runner first; sharding via `replay_dir/shard_<id>/` later.

---

## 8. Dead-Code / Redundancy Detection

The plan adds two abstraction crates (`forge-env`, `forge-env-forge`) whose primary consumers are `forge-env-mc` and `forge-mc-runner`. If a future revert ever removes those consumers, the abstraction crates risk going unused.

Detection layered into CI:
- **`cargo-udeps`** on the workspace (already used per CLAUDE.md spirit; add if absent). Catches unused `Cargo.toml` deps.
- **`cargo clippy -- -D dead_code -D unused_imports`** gating in CI for new crates (CLAUDE.md mandates zero clippy warnings).
- **`cargo-machete`** nightly job warns on any crate with zero in-workspace dependents.
- **JS**: `mc-bot/` runs `eslint --max-warnings 0`; `eslint-plugin-import/no-unused-modules` flags unused exports.
- **Python**: `ruff` with `F401`, `F841` denied; `vulture` nightly.

---

## 9. Vertical Slices — What's Runnable After Each Phase

| End of Phase | Runnable artifact |
|---|---|
| 1 | `cargo test -p forge-env` — trait shape proven via mock env driving real `latent_mcts`. |
| 2 | `cargo test -p forge-env-forge` — parity vs `WorldState`; `latent_mcts` plans over a real FORGE world via the generic trait. |
| 3 | `node mc-bot/src/index.js` connects to local MC server, browser viewer renders bot; `cargo run -p forge-env-mc --example random_walk` drives random actions. |
| 4 | `forge-mc-runner` binary plays Minecraft with bootstrap ONNX, writes `TrajectoryV2` files, hot-reloads on manifest bump. New criterion bench measures `latent_mcts` inference latency. |
| 5 | `python -m forge.training.muzero_mc.cli train --config configs/minecraft/training.toml` consumes replay, produces new ONNX manifest. |
| 6 | `bash scripts/mc_run.sh` brings up the full loop; dashboard shows live viewer + training metrics. |

---

## 10. Open Questions / Decision Points

These need a human decision before or during implementation:

1. **JS test runner choice**: `node:test` (zero-dep, slower) vs `vitest` (faster, adds dev-dep). Audit didn't reveal an existing JS testing convention.
2. **Mineflayer version pinning**: which MC protocol version do we target? `1.20.4` recommended (broad mod compatibility) but should match the Paper server image we ship in Phase 6.
3. **Where do bootstrap ONNX fixtures live?** Proposal: `tests/fixtures/onnx/minecraft/` checked in as small synthetic nets (~100KB total), generated by a `make fixtures` target.
4. **Manifest write location**: same dir as ONNX files, or separate `models/manifests/`? Affects atomicity story (cross-fs rename can fail).
5. **Replay storage compression**: JSONL today; gzip per file? zstd? Defer to Phase 5 with a measurement-first approach.
6. **Dashboard panel framework**: existing `dashboard/` stack (audit didn't probe this) — Phase 6 implementation must verify before committing to React/Svelte/etc.

---

## Appendix A — Audit Findings Reference

Key source-of-truth observations from the gap-analysis pass that this plan honours:

- `forge-replay::TrajectoryStep` at `crates/forge-replay/src/trajectory.rs:14-34` already carries `Vec<Observation>`, `Vec<u32>` actions, `Vec<f32>` rewards, `terminated`, `truncated`. v2 only needs to replace `Vec<Observation>` with `Vec<f32>` and add `format_version`/`schema_id`/`policy_target`/`value_target`.
- `forge-types::Action::space_size_full(comm_vocab, drone, agri, hex)` is the authoritative discrete-action-space size for FORGE. `forge-env-forge::WorldEnv` calls it; never hardcodes the size.
- `OnnxMuZeroModel` at `crates/forge-agent/src/latent_mcts/onnx_model.rs:81-101` uses `ort` v2.0.0-rc.9 with three `Mutex<Session>` (representation/dynamics/prediction) loaded from individual `.onnx` files. The `reload()` addition follows this exact pattern.
- Logging idiom is `#[instrument(skip_all)]` on hot-path functions (see `forge-replay/src/trajectory.rs:66,117,175`). All new public functions follow this.
- Config idiom is `#[derive(Debug, Clone, Default, Serialize, Deserialize)]` with `#[serde(default)]` plus nested sub-configs (see `forge-types::config::ForgeConfig`).
- PyO3 idiom is `py.allow_threads(|| rust_compute())` to release the GIL during Rust work (see `crates/forge-python/src/env.rs:94`).
- No tokio or async — single-threaded deterministic replay is the convention. Our WebSocket client uses a sync `tungstenite` rather than `tokio-tungstenite` to match.
- `forge-bench` has **no** existing `latent_mcts` benchmark — Phase 4 explicitly fills this gap with `latent_mcts_inference.rs`.

---

## Appendix B — Critical Files for Implementation

Files implementers must read first, in priority order:

1. `crates/forge-agent/src/latent_mcts/onnx_model.rs` — central to hot-reload contract.
2. `crates/forge-agent/src/latent_mcts/model.rs` — defines `LatentForwardModel`, the contract `OnnxMuZeroModel` honours.
3. `crates/forge-agent/src/latent_mcts/search.rs` — the PUCT loop that consumes `&[f32]`.
4. `crates/forge-replay/src/trajectory.rs` — v1 format that v2 extends.
5. `crates/forge-replay/src/lib.rs` — module surface to expand.
6. `crates/forge-python/src/env.rs` — PyO3 idiom reference.
7. `crates/forge-types/src/action.rs` — discrete action encoding (`to_discrete`, `from_discrete`, `space_size_full`).
8. `crates/forge-types/src/config.rs` — config-struct template.
9. `crates/forge-types/src/observation.rs` — flat-tensor layout reference for the FORGE-side flattener.
10. Root `Cargo.toml` — workspace member list (every new crate registers here).
11. `python/forge/training/muzero_trainer.py` — reference architecture for the new MC trainer.
