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
- **Zero allocation on hot path**: `WorldState::step()` must not heap-allocate
- **Structured logging**: Use `tracing` crate throughout. `#[instrument]` on public functions
- **Property-based tests**: Use `proptest` for invariant verification alongside unit tests

## Code Style
- Run `cargo fmt` before committing
- All public items must have doc comments
- Error types use `thiserror` derive macros
- Config structs derive `Clone, Debug, Serialize, Deserialize` and impl `Default`
- Use `tracing::{info, debug, warn, error, trace}` for logging, not `println!`
