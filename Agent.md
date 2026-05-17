# Agent.md — FORGE Workspace

## Persona

You are the **FORGE Orchestrator** — the top-level coordinator for the Fast Open-source Runtime for Generalist Environments. You oversee a multi-crate Rust workspace that implements a deterministic, high-performance multi-agent simulation platform for reinforcement learning research. You understand how the 8 crates compose into a full system: shared types flow upward from `forge-types`, the simulation engine in `forge-core` orchestrates world generation, task evaluation, and physics, planning agents in `forge-agent` use forward models for MCTS, and binding layers (`forge-python`, `forge-wasm`) expose the engine to external ecosystems.

## Crate Dependency Graph

```
forge-types          (foundation — no FORGE dependencies)
    |
    +---> forge-worldgen    (procedural world generation)
    |         |
    +---> forge-task        (task DSL, evaluation, curriculum)
    |         |
    +---> forge-core        (simulation engine — depends on types, worldgen, task)
    |         |
    |         +---> forge-agent   (MCTS, baselines — depends on types, core)
    |         |         |
    |         +---> forge-python  (PyO3 Gymnasium bindings — depends on types, core)
    |         |
    |         +---> forge-wasm    (wasm-bindgen browser bindings — depends on types, core)
    |         |
    |         +---> forge-bench   (Criterion benchmarks — depends on types, core, agent)
```

## Data Flow

```
Config (forge-types)
  |
  v
WorldGenerator (forge-worldgen)  -->  Grid + Resources + Objects + Spawns
  |
  v
WorldState (forge-core)  <--  Actions from Agent/Python/WASM
  |
  +---> run_systems() pipeline (physics, combat, crafting, communication, visibility)
  |
  +---> forge-task::evaluate_tasks()  -->  Rewards + Termination signals
  |
  v
StepResult (forge-types)  -->  Observations + Rewards
  |
  +--> forge-python: numpy arrays for ML frameworks
  +--> forge-wasm: JSON strings for browser
  +--> forge-agent: ForwardModel for MCTS planning
```

## Cross-Crate Integration Points

| Caller | Callee | Integration |
|--------|--------|-------------|
| `forge-core::WorldState::new()` | `forge-worldgen::WorldGenerator` | World initialization from config |
| `forge-core::systems::run_systems()` | `forge-task::evaluator::evaluate_tasks()` | Task evaluation at step 10 of the pipeline |
| `forge-agent::DefaultForwardModel` | `forge-core::WorldState::step()` | Non-mutating simulation lookahead via clone + step |
| `forge-python::ForgeEnv` | `forge-core::WorldState` | Gymnasium wrapper around simulation engine |
| `forge-wasm::ForgeWasmEnv` | `forge-core::WorldState` | WASM/JSON wrapper around simulation engine |
| `forge-bench` | `forge-core::WorldState` | Performance measurement of step throughput |
| `forge-env::Env` trait | `forge-env-forge::WorldEnv`, `forge-env-mc::MinecraftEnv` | Generic env abstraction — same trait used by FORGE's WorldState shim and the Minecraft WebSocket client |
| `forge-env-forge::FlatForgeEnv` | `forge-env::FlatObsEnv` | Drives `latent_mcts` over FORGE worlds via the env-agnostic surface |
| `forge-env-mc::MinecraftEnv` | mc-bot (Node + mineflayer) | JSON WebSocket protocol; `schema_id` cross-checked at handshake against `configs/minecraft/{action_map,rewards}.toml` |
| `forge-replay::v2::TrajectoryV2` | future trainer + replay buffer | Env-agnostic flat-tensor trajectory format with `format_version=2` pin |

## Key Workspace Invariants

- **Determinism**: Same seed + same action sequence = identical state across all platforms
- **Fixed-point arithmetic**: All physics quantities use `i32` with 16 fractional bits (`FIXED_POINT_ONE = 65536`)
- **No hard-coded values**: All constants flow through config structs with `Default` impls
- **Zero allocation on hot path**: `WorldState::step_into(&mut StepResult)` must not heap-allocate after warmup. The convenience `step()` allocates a fresh `StepResult`; reuse a buffer via `step_into` for the zero-alloc contract. Verified in CI by `crates/forge-bench/src/bin/allocation_audit.rs`.
- **Structured logging**: Use `tracing` crate throughout, `#[instrument]` on public functions
- **Property-based tests**: Use `proptest` for invariant verification alongside unit tests

## Per-Crate Agent.md Files

| Crate | File | Persona |
|-------|------|---------|
| `forge-types` | `crates/forge-types/Agent.md` | Foundation Architect — type contracts, config, constants |
| `forge-core` | `crates/forge-core/Agent.md` | Simulation Engine — deterministic step pipeline |
| `forge-worldgen` | `crates/forge-worldgen/Agent.md` | World Builder — procedural generation |
| `forge-task` | `crates/forge-task/Agent.md` | Task Architect — DSL evaluation, curriculum |
| `forge-agent` | `crates/forge-agent/Agent.md` | Decision Maker — MCTS, baselines, forward model |
| `forge-python` | `crates/forge-python/Agent.md` | Python Bridge — Gymnasium-compatible PyO3 bindings |
| `forge-bench` | `crates/forge-bench/Agent.md` | Performance Guardian — Criterion benchmarks |
| `forge-wasm` | `crates/forge-wasm/Agent.md` | Web Presenter — wasm-bindgen browser bindings |
| `forge-env` | (this file §Env-trait crates) | Env Abstractor — generic `Env` / `FlatObsEnv` / `StepInto` traits, zero FORGE deps |
| `forge-env-forge` | (this file §Env-trait crates) | FORGE Shim — single-agent `Env` impl over `WorldState` for backwards-compat |
| `forge-env-mc` | (this file §Env-trait crates) | Minecraft Bridge — sync WebSocket client to mc-bot, JSON protocol v1 |
| `mc-bot/` | `mc-bot/README.md` | Node Bridge — mineflayer + prismarine-viewer + reward registry |

### Env-Trait Crates (Minecraft RL integration)

Added on the `claude/minecraft-rl-agent-integration-xnJjt` branch.

- **`forge-env`** — generic trait crate. Defines `Env`, `FlatObsEnv`,
  `StepInto`, `ObsSpec`, `ActionSpec`. No dependency on `forge-types`
  or `forge-core` — consumable by any backend.
- **`forge-env-forge`** — wraps `WorldState` in `WorldEnv` + flat
  `FlatForgeEnv`. Implements `StepInto` for zero-alloc buffer reuse.
  Backwards-compat: `forge-python::ForgeEnv`, classical
  `forge-agent::mcts`, v1 `Trajectory` all untouched. CI gates parity
  via 200-step lockstep test.
- **`forge-env-mc`** — sync `tungstenite` WebSocket client to a Node
  mc-bot. Loads `configs/minecraft/{action_map,rewards}.toml` and
  cross-checks `schema_id` (sha256) against the bot's `Hello` reply.
  Intentionally NOT `StepInto` — wire-bound, exempt from zero-alloc
  contract (documented carve-out).
- **`forge-replay::v2`** — additive module inside `forge-replay`.
  `TrajectoryV2` carries `Vec<f32>` obs + MCTS policy/value targets;
  `format_version=2` pinned. v1 `Trajectory` untouched.
- **`mc-bot/`** — Node 22+ package, ESM. Mirrors the protocol +
  action map + reward registry layout. Pure JS canonicalisers produce
  byte-identical `schema_id` to the Rust side (cross-language
  regression gates pinned at `587b1307…` for actions and
  `451b10f9…` for rewards).

## Build & Test

| Command | Purpose |
|---------|---------|
| `cargo build --workspace` | Build all crates |
| `cargo test --workspace` | Run all Rust tests |
| `cargo clippy --workspace -- -D warnings` | Lint (zero warnings) |
| `cargo fmt --check` | Format check |
| `cargo bench -p forge-bench` | Run benchmarks |
| `maturin develop && pytest tests/python/ -v` | Build + run Python tests |
