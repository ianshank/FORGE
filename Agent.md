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

## Python Training Layer (`python/forge/training/`)

| Module | Role |
|--------|------|
| `loggers.py` | `ForgeLogger` ABC + `MLflowLogger`, `WandbLogger`, `TensorBoardLogger`, `CompositeLogger`, `make_logger` factory |
| `mlflow_config.py` | `MlflowSettings` — env-driven config for MLflow (tracking URI, experiment, run name, tags, credentials, system metrics). All public env-var names exported as module constants. |

`MLflowLogger` is the primary integration point:
- Reads all configuration from `MlflowSettings` (env vars → CLI overrides)
- Idempotent experiment creation with TOCTOU-safe concurrent-start handling
- `log`, `log_artifact`, `log_artifacts`, context-manager support, `run_id` property
- Gracefully degrades to no-op / warning when MLflow is not installed

`scripts/train.py` exposes 8 `--mlflow-*` CLI flags and four helpers
(`_build_mlflow_settings`, `_params_for_run`, `_flatten_for_params`,
`_maybe_make_mlflow_logger`) that thread an optional `MLflowLogger`
through all training entry points.

## Build & Test

| Command | Purpose |
|---------|---------|
| `cargo build --workspace` | Build all crates |
| `cargo test --workspace` | Run all Rust tests |
| `cargo clippy --workspace -- -D warnings` | Lint (zero warnings) |
| `cargo fmt --check` | Format check |
| `cargo bench -p forge-bench` | Run benchmarks |
| `maturin develop && pytest tests/python/ -v` | Build + run Python tests |
