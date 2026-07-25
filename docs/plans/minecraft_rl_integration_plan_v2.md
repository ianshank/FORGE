# FORGE Minecraft RL Integration — Implementation Plan v2

**Date:** 2026-05-17
**Branch:** `claude/minecraft-rl-agent-integration-xnJjt`
**Supersedes:** [`minecraft_rl_integration_plan_v1.md`](./minecraft_rl_integration_plan_v1.md)
**Status:** Draft for review, not yet implemented

---

## 0. Changes from v1

Peer-review pass found ten substantive issues. v2 addresses all of them; v1 is kept for diff/history. Critical changes:

| # | Severity | Issue | Resolution location |
|---|---|---|---|
| 1 | Critical | Reward signal source undefined | New Phase 3 reward subsystem (§3.3.1) |
| 2 | Critical | Episode reset semantics for persistent MC world | New Phase 3 reset strategy (§3.3.2) |
| 3 | Critical | `type Action: Copy` rejects FORGE's own `Action::Communicate` | Phase 1 trait revision (§3.1) |
| 4 | Significant | Binary obs-frame protocol announced but unspecified | Removed from v1; deferred to protocol v2 (§4.4) |
| 5 | Significant | Zero-alloc claim incompatible with wire-bound `MinecraftEnv` | Required buffer-filling `Env::step_into` plus explicit wire-bound allocation carve-out (§1, §3.1) |
| 6 | Significant | ONNX opset / ort runtime version compat unverified | Phase 5 round-trip CI test (§3.5) |
| 7 | Worth fixing | Bootstrap ONNX origin unspecified | Phase 5 `bootstrap.py` (§3.5) |
| 8 | Worth fixing | Multi-agent `from_v1` flatten ambiguity | Signature change to `&[Observation] -> Vec<f32>` (§3.5) |
| 9 | Worth fixing | `name() -> &'static str` too restrictive | Changed to `Cow<'static, str>` (§3.1) |
| 10 | Worth fixing | Phase 6 dashboard chicken-and-egg with §10 open question | Phase 6 narrowed to viewer-iframe-only (§3.6) |

Everything not listed above is unchanged from v1.

---

## 1. Convention Adherence Checklist

Same as v1 with one explicit clarification:

**Zero-alloc carve-out.** CLAUDE.md mandates zero allocation on the hot path, enforced via `crates/forge-bench/src/bin/allocation_audit.rs`. The audit gate applies to `WorldState::step_into` and Rust-internal simulation paths. Every `Env` now implements buffer-filling `reset_into` and `step_into`; allocating `reset` and `step` are convenience wrappers. Wire-bound environments (`MinecraftEnv`) reuse caller observation buffers through `Env::step_into`, but internal WebSocket I/O and JSON parsing still allocate per step and are explicitly excluded from the allocation audit by module path. In-process envs such as `FlatForgeEnv` must honor the full no-allocation hot-path contract.

| Concern | Pattern | Source |
|---|---|---|
| Logging | `#[tracing::instrument(skip_all, fields(...))]` on public fns; `debug!`/`warn!`/`error!` for events | `forge-replay/trajectory.rs:66,117,175` |
| Errors | `#[derive(thiserror::Error, Debug)]` for libs; `anyhow::Result` only in `bin/` | `forge-agent` convention |
| Config | `#[derive(Debug, Clone, Default, Serialize, Deserialize)]` + `#[serde(default)]` | `forge-types/config.rs:26-51` |
| Tests | `#[test]` unit; `proptest! { #[test] ... }`; `criterion` benches | `forge-replay/trajectory.rs:323-371`, `forge-bench/benches/step_throughput.rs` |
| PyO3 | `#[pyclass]` + `#[pymethods]` + `py.allow_threads(\|\| ...)`; numpy via`numpy::PyArray*` | `forge-python/src/env.rs` |
| ONNX | `ort` v2.x; `Mutex<Session>` per net; `Session::builder().commit_from_file(path)` | `forge-agent/src/latent_mcts/onnx_model.rs:81-101` |
| Zero-alloc | Hot-path Rust paths only; wire-bound envs exempt with documented carve-out | This document §1 |
| Workspace registration | Add new crate to `crates/` + workspace `members` + `[workspace.dependencies]` | Root `Cargo.toml` |
| Public doc comments | Every public item documented | CLAUDE.md |

---

## 2. Architecture at a Glance

Unchanged from v1 — same six-component topology, same versioning at every wire.

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
                                                            (JSON only)  v
                                                          +--------------+----------+
                                                          | forge-mc-bot (Node)     |
                                                          | mineflayer + viewer     |
                                                          | + reward.js + reset.js  |
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

---

## 3. Phase Breakdown

### 3.1 Phase 1 — `forge-env` crate (revised trait)

Same crate layout as v1. **Trait signature changes** from peer review:

```rust
// crates/forge-env/src/env.rs
pub trait Env: Send {
    type Obs;
    type Action: Send;            // CHANGED: removed `Copy` bound
    type Info: Default + Send;
    type Error: std::error::Error + Send + Sync + 'static;

    fn reset_into(&mut self, seed: Option<u64>, out: &mut Self::Obs)
        -> Result<(), Self::Error>;
    fn step_into(&mut self, action: Self::Action, out: &mut StepOutput<Self::Obs, Self::Info>)
        -> Result<(), Self::Error>;
    fn obs_spec(&self) -> &ObsSpec;
    fn action_spec(&self) -> &ActionSpec;

    // CHANGED: was &'static str; now allows dynamic names like
    // "minecraft-v1.20.4-paper-r3"
    fn name(&self) -> Cow<'_, str> { Cow::Borrowed("env") }

    fn close(&mut self) -> Result<(), Self::Error> { Ok(()) }

    fn reset(&mut self, seed: Option<u64>) -> Result<Self::Obs, Self::Error>
    where
        Self::Obs: Default;
    fn step(&mut self, action: Self::Action) -> Result<StepOutput<Self::Obs, Self::Info>, Self::Error>
    where
        Self::Obs: Default;
}

pub struct StepOutput<Obs, Info> {
    pub obs: Obs,
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
    pub info: Info,
}

/// Marker for envs feeding latent_mcts. Obs is a flat float vec; action a discrete index.
/// Action = u32 is Copy, so the hot-path action remains trivially copyable.
pub trait FlatObsEnv: Env<Obs = Vec<f32>, Action = u32> {
    fn obs_dim(&self) -> usize;
    fn num_actions(&self) -> u32;
}
```

Rationale:
- **Drop `Copy` on `Action`** — `forge_types::Action::Communicate { message: String, ... }` is not `Copy`. `FlatObsEnv` constrains `Action = u32` so the hot path stays `Copy`.
- **`Cow<'_, str>` for `name`** — allows static defaults *and* runtime-formatted names (server version, world seed).
- **Required buffer-filling methods** — `reset_into` and `step_into` are part of `Env`, while `reset` and `step` are allocating convenience wrappers. `FlatForgeEnv` must satisfy the full zero-alloc buffer contract; `MinecraftEnv` reuses caller observation buffers but remains excluded from the allocation audit because WebSocket I/O and JSON parsing allocate.

**Tests:** Same as v1 plus:
- `dynamic_name_via_cow_owned` — verifies `name()` can return an owned formatted string.
- `step_into_reuses_buffer_capacity` — `MockFlatEnv` implements `Env::step_into`; test asserts buffer reuse via `Vec::capacity` invariance across calls.

Everything else (spec types, error type, integration test) is identical to v1.

---

### 3.2 Phase 2 — `forge-env-forge` (backwards-compat shim)

Unchanged from v1 except:
- `WorldEnv::name()` now returns `Cow::Owned(format!("forge-{}", config.world.world_id))` instead of a static string.
- `FlatForgeEnv` implements `Env::step_into` — it reuses cached typed-observation buffers and writes flattened observations into the caller's `Vec<f32>` buffer without allocating on the hot path. `step_into_keeps_buffer_dim_stable` gates this behavior.

Parity test scope unchanged: 1k deterministic steps vs `WorldState` direct path.

---

### 3.3 Phase 3 — Mineflayer bot + WebSocket protocol (revised)

Two new subsystems were missing from v1: reward computation and episode reset.

#### 3.3.1 Reward subsystem (was missing in v1)

**New files in `mc-bot/`:**
- `src/reward/index.js` — registry + composition
- `src/reward/builtins/` — built-in `RewardFn` impls
- `mc-bot/test/reward.test.js`

**Contract:**
```js
// mc-bot/src/reward/index.js
// RewardFn shape:
//   ({ bot, prev, curr, action, tick }) => number
// `prev` and `curr` are snapshots from observation.js. Pure function;
// no side effects (so a CompositeReward can call many in sequence).
```

**Built-in `RewardFn`s (each a separate file under `builtins/`):**

| Name | Signal |
|---|---|
| `block_broken` | +R per block of configured type broken |
| `block_placed` | +R per block placed in configured set |
| `distance_to_goal` | -ΔL₁ to a configured `{x,y,z}` waypoint |
| `inventory_acquired` | +R first time each configured item appears in inventory |
| `health_delta` | sign-of-delta(health) × R |
| `survival` | +R per tick alive |
| `composite` | weighted sum of any of the above |

**Config (`configs/minecraft/rewards.toml`):**
```toml
schema_version = 1

[[reward]]
kind = "composite"
weights = { distance_to_goal = 1.0, inventory_acquired = 10.0, survival = 0.01 }

[reward.distance_to_goal]
target = { x = 0, y = 64, z = 0 }
clip = 100.0

[reward.inventory_acquired]
items = ["minecraft:oak_log", "minecraft:cobblestone"]
```

**Schema-id discipline:** The reward config's canonical-form sha256 is folded into `schema_id` (alongside `obs_dim` and `action_count`). Changing rewards mid-experiment invalidates prior replay buffers — surfaced as a hard error at trainer startup, with a documented migration path: "rewards changed; rerun data collection or branch the schema_id namespace."

**Tests:**
- Unit (JS): one test per built-in `RewardFn` plus `composite` aggregation.
- Property (JS): `survival_reward_monotonic_in_ticks`, `inventory_acquired_idempotent_per_item`.
- Integration (`tests/reward_replay.test.js`): replays a recorded mineflayer event trace through the reward fn and asserts numeric reproducibility.

#### 3.3.2 Episode reset strategy (was missing in v1)

A Paper server doesn't reset between episodes. v2 picks the simplest working option:

**Reset = teleport + state restore, NOT world regeneration.** For v1 of the integration:
1. Bot is teleported to `spawn.x/y/z` (config-driven).
2. Inventory is fully cleared (`/clear @s`).
3. Health and food restored (`/effect give @s minecraft:instant_health`, `/effect give @s minecraft:saturation`).
4. Episode tick counter resets to 0.
5. Nearby block state is *not* restored — episodes are independent in agent state but share a slowly-evolving world.

**Why not full regeneration:** chunk regen via `/chunky` or world reset is 10s+; episodic learning at the scale we want can't tolerate that. Drift is bounded because the bot only modifies blocks in its action radius and episodes are short (~5min).

**Optional Phase 3+ extension:** "arena mode" — a flat structured arena at fixed coordinates, periodically rebuilt by a server-side `WorldEdit` schematic restore on episode N (config-driven). Not in v1.

**Config (`configs/minecraft/reset.toml`):**
```toml
strategy = "teleport"        # or "arena" (future)

[teleport]
spawn = { x = 0, y = 64, z = 0 }
yaw = 0
pitch = 0
clear_inventory = true
restore_health = true
restore_food = true

[arena]                       # used when strategy = "arena"
enabled = false
schematic_path = "schematics/arena.schem"
rebuild_every_n_episodes = 100
```

**Tests:**
- Integration `tests/reset_smoke.test.js`: drives 5 reset cycles against `flying-squid` mock server; asserts bot position, inventory empty, health full each time.
- Property: `reset_is_idempotent_over_repeated_calls`.

#### 3.3.3 Protocol — JSON only, no binary frame

v1 mentioned a binary-frame optimization but didn't specify how `ServerMsg` distinguishes binary from JSON. v2 **drops binary frames entirely** from protocol v1. All observations are JSON `Vec<f32>`. This costs ~3× bandwidth vs raw f32 but keeps the wire format trivially debuggable. A future protocol v2 can add binary frames with explicit framing once we have throughput numbers proving it's needed.

The `obs_binary_threshold` config knob is **removed** from v2.

Otherwise protocol unchanged:
```rust
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMsg {
    Hello { schema_version: u32, action_count: u32, obs_dim: usize, schema_id: String },
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

Note `Hello` now also carries `schema_id` so the runner can fail fast on action-map or reward-config drift versus the bot.

**Config (`configs/minecraft/env.toml`):** same as v1 minus the removed `obs_binary_threshold`.

---

### 3.4 Phase 4 — Rust runner

Unchanged from v1, with one clarification on lock ordering for hot-reload:

**Lock-order discipline in `OnnxMuZeroModel::reload`:** all three `Mutex<Session>` are acquired in a fixed order — `representation`, `dynamics`, `prediction`. Inference methods (`infer_representation`, `infer_dynamics`, `infer_prediction`) each lock only one mutex at a time, so deadlock is structurally impossible. Test `reload_no_deadlock_under_concurrent_inference` spawns 4 threads doing `infer_*` while a 5th calls `reload` 100×; asserts completion within timeout. Doc comment on `reload` calls out the invariant for future maintainers.

`HotReloadWatcher::poll` is called **only between episodes**, by `Runner::run_episode` after the loop exits but before the next episode starts. The watcher itself doesn't enforce this; the doc comment does, and Phase 4 unit test `poll_called_mid_episode_is_caller_error` verifies the runner doesn't call it mid-loop.

Everything else (new `latent_mcts` inference bench, `LatentPlanner`, `TrajectoryWriter`) is identical to v1.

---

### 3.5 Phase 5 — Python trainer + replay v2 (revised)

Three changes from v1:

#### 3.5.1 Bootstrap ONNX (was missing in v1)

**New file:** `python/forge/training/muzero_mc/bootstrap.py`

```python
"""Produce a random-init MuZero ONNX bundle the runner can load on first start."""

class BootstrapConfig:
    obs_dim: int
    action_count: int
    latent_dim: int
    schema_id: str
    output_dir: pathlib.Path
    seed: int  # for reproducible random init

def bootstrap(cfg: BootstrapConfig) -> ModelManifest:
    """Initialise three MuZeroNet modules, export ONNX, write manifest with version=0."""
```

**CLI:**
```
python -m forge.training.muzero_mc.cli bootstrap \
    --obs-dim 960 --action-count 64 --latent-dim 256 \
    --schema-id mc-v1-obs960-act64 \
    --output models/
```

`forge-mc-runner` refuses to start if `model_manifest.json` is missing — first-run instructions in `examples/minecraft/quickstart.md` direct users to run `bootstrap` first.

**Tests:**
- `test_bootstrap_produces_loadable_bundle`: runs bootstrap; asserts the three .onnx files load via `onnxruntime.InferenceSession`; calls `OnnxMuZeroModel::load` via a PyO3 test helper.
- `test_bootstrap_deterministic_under_seed`: same seed twice → byte-identical .onnx files (modulo non-deterministic exporter metadata).

#### 3.5.2 ONNX opset / runtime compatibility test (was missing in v1)

**New CI test:** `tests/python/training/test_onnx_compat.py` runs on every PR.

Flow:
1. Bootstrap a tiny bundle (latent_dim=8, action_count=4).
2. Round-trip through Rust: spawn a subprocess running `cargo run -p forge-agent --example load_and_infer -- <manifest_path>`.
3. Subprocess loads the bundle via `OnnxMuZeroModel::new`, runs one inference per net, prints output shapes.
4. Test asserts subprocess exit 0 and output shapes match expected.

Catches opset mismatch between Python export (`onnx_opset = 17`) and `ort` v2.0.0-rc.9's bundled onnxruntime version. Pinning the `ort` rc version in `Cargo.toml` plus this test together prevent silent compatibility regressions.

#### 3.5.3 Multi-agent flatten signature fix

```rust
// crates/forge-replay/src/v2.rs
// CHANGED in v2: takes a slice of observations so multi-agent v1 trajectories
// can be converted unambiguously (caller decides whether to concat, select
// agent 0, or error).
pub fn from_v1(
    t: &Trajectory,
    flatten_step: &dyn Fn(&[Observation]) -> Vec<f32>,
) -> TrajectoryV2;
```

Reference flatteners under `forge-replay::v2::flatteners`:
- `agent_zero_only(obs: &[Observation]) -> Vec<f32>` — picks agent 0, errors if zero agents.
- `concat_agents(obs: &[Observation]) -> Vec<f32>` — concats in order; requires homogeneous obs.

Trainer-side `replay.py` only consumes `TrajectoryV2` directly; the `from_v1` path is migration tooling, not runtime.

Everything else in Phase 5 is identical to v1.

---

### 3.6 Phase 6 — End-to-end glue (narrowed)

v1 included a dashboard panel as a Phase 6 deliverable while leaving the dashboard framework as an §10 open question — chicken-and-egg. v2 narrows Phase 6:

**In scope for v2:**
- `scripts/mc_run.sh` — orchestrates docker-compose + runner + trainer.
- `docker/mc-bot.Dockerfile` + `docker/compose.minecraft.yml`.
- `mc-bot/src/viewer.js` exposes prismarine-viewer on `viewer.port` (already in Phase 3).
- `examples/minecraft/quickstart.md` documents pointing a browser at `http://localhost:<viewer.port>` and the runner's `/metrics` endpoint at `http://localhost:<runner.metrics_port>`.
- `tests/e2e_full_loop.py` — full vertical slice test.
- `docs/minecraft.md` runbook.

**Out of scope (deferred to v3):**
- Dashboard integration. v2 ships *standalone* viewer + metrics endpoints, viewable in any browser. Whether/how to embed them in the existing `dashboard/` stack is a separate workstream once the framework decision lands.

Runner exposes a minimal Prometheus-format metrics endpoint on `runner.metrics_port` (config knob, default 9090) with:
- `forge_mc_episode_total{result="terminated|truncated"}`
- `forge_mc_episode_reward_sum`
- `forge_mc_planning_latency_ms` (histogram)
- `forge_mc_model_version` (gauge)
- `forge_mc_protocol_error_total`

This is independently useful without any dashboard.

---

## 4. Cross-Cutting Concerns

### 4.1 Backwards compatibility

Unchanged from v1.

### 4.2 Replay buffer format

Unchanged from v1. The `schema_id` content hash in v2 additionally folds in the canonical-form sha256 of `rewards.toml`, so changing rewards mid-experiment is detected as a schema mismatch.

### 4.3 Hot-reload protocol for ONNX weights

Unchanged from v1 except for the explicit **lock ordering** doc-and-test in §3.4.

### 4.4 WebSocket protocol contract

Unchanged from v1 except:
- **No binary frames in protocol v1.** All `Observation.obs` is JSON `Vec<f32>`. Binary framing is a deferred protocol v2 feature.
- `Hello` carries `schema_id` so runner detects bot-side config drift at handshake.
- `obs_binary_threshold` config knob removed.

---

## 5. Gap Analysis (revised)

| Capability | Where it lives today | What's missing | Where it'll live |
|---|---|---|---|
| Generic env abstraction | None | Trait + spec types | `forge-env` (Phase 1) |
| FORGE WorldState as generic Env | `forge-python::ForgeEnv` only | Native Rust `Env` impl | `forge-env-forge::WorldEnv` (Phase 2) |
| Minecraft connection (game side) | None | mineflayer bot + viewer | `mc-bot/` (Phase 3) |
| **Minecraft reward signal** | **None (v1 GAP)** | **Pluggable `RewardFn` registry + config** | **`mc-bot/src/reward/` (Phase 3.3.1)** |
| **Minecraft episode reset** | **None (v1 GAP)** | **Teleport + state restore strategy** | **`mc-bot/src/reset.js` (Phase 3.3.2)** |
| Minecraft env (Rust client) | None | WS protocol + `Env` impl | `forge-env-mc` (Phase 3) |
| Latent MCTS planner | `forge-agent::latent_mcts` ✅ | — | Reused as-is |
| ONNX inference | `latent_mcts::onnx_model::OnnxMuZeroModel` ✅ | Hot-reload swap | Add `reload()` (Phase 4) |
| Runner loop | None | Binary that ties them | `forge-mc-runner` (Phase 4) |
| Trajectory format env-agnostic | `forge-replay::Trajectory` (FORGE-typed) | `Vec<f32>` obs variant + format_version | `forge-replay::v2` (Phase 5) |
| MuZero training from MC trajectories | `python/forge/training/muzero_trainer.py` (WorldState-typed) | MC-typed trainer over `TrajectoryV2` | `python/forge/training/muzero_mc/` (Phase 5) |
| ONNX export from trained nets | `python/forge/agents/muzero_agent.py` (FORGE-specific) | Reusable exporter with manifest | `muzero_mc/exporter.py` (Phase 5) |
| **Bootstrap ONNX origin** | **None (v1 GAP)** | **Random-init exporter script** | **`muzero_mc/bootstrap.py` (Phase 5.3.1)** |
| **ONNX opset / ort runtime compat** | **None (v1 GAP)** | **Round-trip CI test** | **`tests/python/training/test_onnx_compat.py` (Phase 5.3.2)** |
| Browser viewer | None | prismarine-viewer iframe | `mc-bot/src/viewer.js` (Phase 3) |
| Runner observability | None | Prometheus metrics endpoint | `forge-mc-runner::metrics` (Phase 6) |
| Ops orchestration | None | docker-compose + scripts | Phase 6 |
| Latent MCTS inference benchmark | None | Criterion bench for `search()` at varying sim budgets | `forge-bench/benches/latent_mcts_inference.rs` (Phase 4) |

---

## 6. Risk Register (revised)

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | Tick-rate mismatch (MC @ 20 TPS vs MCTS ~100ms/move) | High | Med | `episode.action_repeat`; server holds last action N ticks; `forge_mc_planning_latency_ms` metric + alarm. |
| 2 | Action-space drift between Node, Rust, Python | High | High | Single `action_map.toml`; sha256 `schema_id` cross-checked in `Hello` + replay header + trainer startup. |
| 3 | ONNX export pitfalls (missing ops, dynamic-axis) | Med | High | `opset=17` pin; `onnxsim` + `onnxruntime.InferenceSession` smoke; **Phase 5.3.2 cross-runtime round-trip test**. |
| 4 | Replay format churn | High | Med | `format_version` pinned; trainer errors on unknown; v1 untouched. |
| 5 | JS↔Rust schema drift in WS protocol | Med | High | Golden-file snapshot test; JS unit test on same fixture; `Hello.schema_id` handshake. |
| 6 | Hot-reload race (manifest swap mid-search) | Med | High | Reload only between episodes (doc + test); fixed mutex lock order (`representation→dynamics→prediction`); `schema_id` rejects dim changes. |
| 7 | **Reward-config drift invalidates replays** (NEW) | High | Med | `rewards.toml` sha256 folded into `schema_id`; trainer fails fast on mismatch with documented migration path. |
| 8 | **Persistent-world drift across episodes** (NEW) | Med | Med | Teleport-only reset bounds blast radius; nightly E2E test asserts 100 sequential episodes complete without world corruption; arena-mode fallback documented for high-drift workloads. |
| 9 | **`ort` rc version drift** (NEW) | Low | High | Pin exact `ort = "=2.0.0-rc.9"` in `Cargo.toml`; Phase 5.3.2 round-trip test catches upstream onnxruntime version mismatches. |

---

## 7. Out of Scope

Unchanged from v1, plus:
- Binary WebSocket frames (deferred to protocol v2 once throughput numbers justify it).
- Dashboard integration with `dashboard/` stack (Phase 6 ships standalone viewer + metrics; embedding is v3 work).
- Full world regeneration between episodes (teleport-reset is v2; arena-mode is documented but not implemented).

---

## 8. Dead-Code / Redundancy Detection

Unchanged from v1.

---

## 9. Vertical Slices

Unchanged from v1 in structure. End-of-phase artifacts now include:

| End of Phase | Runnable artifact |
|---|---|
| 1 | `cargo test -p forge-env` — `Env::reset_into` / `Env::step_into` proven via `MockFlatEnv` driving `latent_mcts`. |
| 2 | `cargo test -p forge-env-forge` — parity vs `WorldState`; allocation audit passes for `FlatForgeEnv::step_into`. |
| 3 | `node mc-bot/src/index.js` connects to local MC server; bot teleports on reset, emits computed rewards; `cargo run -p forge-env-mc --example random_walk`. |
| 4 | `forge-mc-runner` plays MC with bootstrap ONNX (after running `bootstrap` CLI), writes `TrajectoryV2`, hot-reloads on manifest bump. Criterion bench publishes inference latency. |
| 5 | `python -m forge.training.muzero_mc.cli train` consumes replay, produces new ONNX manifest; CI round-trip test green. |
| 6 | `bash scripts/mc_run.sh` brings up full loop; browser at `viewer.port` shows bot POV, `/metrics` endpoint serves Prometheus. |

---

## 10. Open Questions

v1 had 6 open questions. v2 resolves 1 and 4:

| # | Question | Resolution |
|---|---|---|
| 1 | JS test runner: `node:test` vs `vitest`? | Still open — leaning `node:test` for zero-dep. |
| 2 | Mineflayer / MC version pinning | Still open — proposal `1.20.4` (broad mod compat) to be confirmed by Phase 3 implementer. |
| 3 | Bootstrap ONNX fixture location | **Resolved** — `python/forge/training/muzero_mc/bootstrap.py` generates them on demand under `models/`; not checked in. |
| 4 | Manifest write location | **Resolved** — same directory as ONNX files (`models/model_manifest.json`); atomicity via tmp file + same-dir rename. |
| 5 | Replay storage compression | Still open — JSONL plain in v1; measure first. |
| 6 | Dashboard panel framework | **Deferred** — Phase 6 narrowed to standalone viewer + metrics endpoint. Framework decision moved to v3. |
| 7 | **Reward fn defaults** (NEW) | What's the v1 default `rewards.toml`? Proposal: `composite { distance_to_goal: 1.0, survival: 0.01 }` for `NavigateDense`-equivalent. To confirm before Phase 3. |

---

## Appendix A — Audit Findings Reference

Unchanged from v1.

---

## Appendix B — Critical Files for Implementation

Unchanged from v1.

---

## Appendix C — Diff Summary vs v1

Net new sections in v2:
- §0 Changes-from-v1 table
- §3.3.1 Reward subsystem
- §3.3.2 Episode reset strategy
- §3.5.1 Bootstrap ONNX
- §3.5.2 ONNX opset compat CI test
- §3.5.3 Multi-agent flatten fix
- Phase 6 metrics endpoint
- Risks #7–9
- §10 question #7

Net deletions from v1:
- Binary WebSocket frame paths and `obs_binary_threshold` knob
- Dashboard panel from Phase 6
- `type Action: Copy` bound

Net changes:
- `name() -> Cow<'_, str>` instead of `&'static str`
- `from_v1(t, flatten: &dyn Fn(&Observation) -> Vec<f32>)` → `&dyn Fn(&[Observation]) -> Vec<f32>`
- `Hello` now carries `schema_id`
- Zero-alloc carve-out explicit; `Env::reset_into` / `Env::step_into` are required, with allocating convenience wrappers
- Lock-ordering doc + test in `OnnxMuZeroModel::reload`
