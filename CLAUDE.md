# FORGE Development Conventions

## Build & Test Commands
- `cargo build --workspace` — Build all crates
- `cargo test --workspace` — Run all Rust tests
- `cargo clippy --workspace -- -D warnings` — Lint (must pass with zero warnings)
- `cargo fmt --check` — Format check
- `cargo bench -p forge-bench` — Run benchmarks
- `pytest tests/python/ -v` — Run Python tests (requires `maturin develop` first)

## Architecture
- **Workspace**: Multi-crate Rust workspace under `crates/`
- **forge-types**: Shared types, configs, errors — no heavy dependencies
- **forge-core**: Simulation engine — deterministic step function
- **forge-worldgen**: Procedural world generation (Perlin noise, WFC)
- **forge-task**: Task DSL and curriculum system
- **forge-agent**: MCTS planning and baseline agents
- **forge-python**: PyO3 bindings for Python/Gymnasium API
- **forge-bench**: Criterion benchmarks

## Key Principles
- **No hard-coded values**: All constants flow through config structs with `Default` impls
- **Deterministic**: Same seed + actions = identical state. Use `fixed` crate for physics, `rand_pcg` for RNG
- **Zero allocation on hot path**: `WorldState::step_into(&mut StepResult)` must not heap-allocate after warmup. The convenience `step()` wrapper allocates a fresh `StepResult` per call; reuse a buffer via `step_into` for the zero-alloc contract. Enforced in CI by `crates/forge-bench/src/bin/allocation_audit.rs` + `benchmarks/runner/check_zero_alloc.py`.
- **Structured logging**: Use `tracing` crate throughout. `#[instrument]` on public functions
- **Property-based tests**: Use `proptest` for invariant verification alongside unit tests

## Code Style
- Run `cargo fmt` before committing
- All public items must have doc comments
- Error types use `thiserror` derive macros
- Config structs derive `Clone, Debug, Serialize, Deserialize` and impl `Default`
- Use `tracing::{info, debug, warn, error, trace}` for logging, not `println!`

## MLflow Tracking

The Python training layer ships first-class MLflow experiment tracking via
`python/forge/training/mlflow_config.py` (`MlflowSettings`) and the
`MLflowLogger` in `python/forge/training/loggers.py`.

All configuration is environment-driven — no URIs, experiment names, or
credentials are hard-coded:

| Source | Mechanism |
|--------|-----------|
| Env vars | `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME`, `MLFLOW_RUN_NAME`, `MLFLOW_TRACKING_USERNAME/PASSWORD/TOKEN`, `FORGE_MLFLOW_TAGS`, etc. |
| CLI flags | `--mlflow-enabled`, `--mlflow-experiment`, `--mlflow-run-name`, `--mlflow-tracking-uri`, `--mlflow-artifact-location`, `--mlflow-tags`, `--mlflow-system-metrics`, `--mlflow-config` |
| Programmatic | `MlflowSettings.from_env().merge(...)` |

Key invariants:
- `MLflowLogger` accepts a `MlflowSettings` instance or builds one from env.
- `_resolve_experiment` is TOCTOU-safe: concurrent parallel runs that race on
  experiment creation are handled gracefully.
- `mlflow` is an optional dependency; if not installed, `MLflowLogger`
  construction raises `ImportError`; `_maybe_make_mlflow_logger` in
  `scripts/train.py` swallows this and continues without tracking.
- All env-var name strings are module-level constants in `mlflow_config.py`
  — callers never spell magic strings.
