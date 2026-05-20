# FORGE Development Conventions

## Build & Test Commands
- `cargo build --workspace` — Build all crates
- `cargo test --workspace` — Run all Rust tests
- `cargo clippy --workspace -- -D warnings` — Lint (must pass with zero warnings)
- `cargo fmt --check` — Format check
- `cargo bench -p forge-bench` — Run benchmarks
- `pytest tests/python/ -v` — Run Python tests (requires `maturin develop` first)
- `cd mc-bot && npm test` — Run Node-side mc-bot tests (no install required for the dep-free modules; `npm install` for mineflayer + Biome)
- `cd mc-bot && npm run lint` — Biome lint + format check on the JS surface
- `cargo run -p forge-mc-runner -- --dry-run --episodes 1` — Smoke-test the runner binary without docker / Minecraft (CI: `forge-mc-runner-bin` job)
- `cargo test -p forge-agent --features onnx` + `cargo test -p forge-mc-runner --features onnx-reload` — Exercise the `OnnxMuZeroModel::reload()` + `into_reload_fn` ONNX hot-reload surface
- `curl http://127.0.0.1:9090/metrics` — Scrape the runner's Prometheus endpoint (disabled by setting `metrics_port = 0` in the runner config; `metrics_bind` defaults to `127.0.0.1`)
- `python -m forge.training.muzero_mc.cli bootstrap --obs-dim N --action-dim M --schema-id <sha> --out models/` — Phase-5 random-init bundle
- `python -m forge.training.muzero_mc.cli validate-manifest <path>` — Validate a model_manifest.json (exit 0 / 3 / 4)
- `python -m forge.training.muzero_mc.cli train --input trajectories/ --out models/ --schema-id <sha> --obs-dim N --action-dim M --iters 100 --export-every 10` — Train loop consuming `.json` / `.json.gz` trajectories, periodic ONNX export + manifest bump that the runner's `HotReloadWatcher` picks up. `--obs-dim`, `--action-dim`, `--schema-id`, `--out`, `--input` are required; `--manifest` defaults to `<out>/model_manifest.json`
- `scripts/mc_run.sh --dry-run` — Print resolved docker compose argv for the stack (no side effects)
- `scripts/mc_run.sh --build` — Bring the Minecraft + mc-bot + runner stack up (foreground; Ctrl-C cleans up)
- `cd mc-bot && npm run typecheck` — `tsc --noEmit` gate over the `.js` source (CI: `mc-bot-test` job runs this between `npm ci` and `npm test`)
- `pytest tests/python/integration/ -m minecraft_e2e -v` — Run the opt-in compose-stack E2E (requires docker; never runs on a default `pytest` invocation)

## Architecture
- **Workspace**: Multi-crate Rust workspace under `crates/`
- **forge-types**: Shared types, configs, errors — no heavy dependencies
- **forge-core**: Simulation engine — deterministic step function
- **forge-worldgen**: Procedural world generation (Perlin noise, WFC)
- **forge-task**: Task DSL and curriculum system
- **forge-agent**: MCTS planning and baseline agents
- **forge-python**: PyO3 bindings for Python/Gymnasium API
- **forge-bench**: Criterion benchmarks
- **forge-env**: Generic `Env` / `FlatObsEnv` trait crate. No FORGE deps; every env implements buffer-filling `reset_into` / `step_into`, with allocating `reset` / `step` wrappers for convenience.
- **forge-env-forge**: Single-agent `Env` impl over `WorldState`. Additive shim — does not replace `forge-python::ForgeEnv` or classical `mcts`.
- **forge-env-mc**: Sync WebSocket client to a Node `mc-bot` exposing a Minecraft env via `Env` + `FlatObsEnv`.
- **forge-mc-runner**: End-to-end episode runner. `RunnerConfig` (TOML + validate; opt-in `trajectory_compression = "gzip"` + `trajectory_gzip_level` + `metrics_bind` + `metrics_histogram_buckets` fields), `ModelManifest` (atomic save, sha256-per-role, monotonic version, pinned `MANIFEST_SCHEMA_VERSION = 1`), `HotReloadWatcher` (between-episode poll-only contract; no downgrade), `TrajectoryWriter` (atomic `TrajectoryV2` save, opt-in `with_compression()` for `.json.gz`), `Runner<E, M>` (the episode loop with optional `MetricsRecorder` via `with_metrics()`), and `metrics.rs` (axum + prometheus server exposing the five v2-plan §3.6 signals). The `forge-mc-runner` binary is `#[tokio::main]` with SIGINT graceful shutdown joining the runner + metrics tasks. ONNX hot-reload integration (`OnnxMuZeroModel::reload()` + `onnx_reload::into_reload_fn`) is feature-gated behind `onnx-reload` so the runner stays buildable without ONNX Runtime.
- **forge-replay::v2**: Env-agnostic flat-tensor trajectory format (additive; v1 untouched).
- **mc-bot/**: Node bridge (mineflayer + reward registry + reset + viewer), out of the Cargo workspace.

## Key Principles
- **No hard-coded values**: All constants flow through config structs with `Default` impls
- **Deterministic**: Same seed + actions = identical state. Use `fixed` crate for physics, `rand_pcg` for RNG
- **Zero allocation on hot path**: `WorldState::step_into(&mut StepResult)` must not heap-allocate after warmup. The convenience `step()` wrapper allocates a fresh `StepResult` per call; reuse a buffer via `step_into` for the zero-alloc contract. Enforced in CI by `crates/forge-bench/src/bin/allocation_audit.rs` + `benchmarks/runner/check_zero_alloc.py`. **Carve-out**: wire-bound envs like `forge-env-mc::MinecraftEnv` implement the same `Env::step_into` method and reuse caller observation buffers, but internal WebSocket I/O and JSON parsing still allocate and are explicitly exempt from the core allocation audit. In-process envs such as `FlatForgeEnv` must honour the full buffer-reuse contract.
- **Structured logging**: Use `tracing` crate throughout. `#[instrument]` on public functions
- **Property-based tests**: Use `proptest` for invariant verification alongside unit tests
- **Cross-language schema_id contracts**: For configs shared between Rust and JS (e.g. `configs/minecraft/{action_map,rewards}.toml`), both sides compute the same canonical sha256 and pin a known-good value in paired tests (Rust `xlang_*_pinned_to_known_good` ↔ Node `xlang ... matches Rust`). Drift on either side fails both tests simultaneously.

## Code Style
- Run `cargo fmt` before committing
- All public items must have doc comments
- Error types use `thiserror` derive macros
- Config structs derive `Clone, Debug, Serialize, Deserialize` and impl `Default`
- Use `tracing::{info, debug, warn, error, trace}` for logging, not `println!`
- JS code: ESM modules, Node 22+, `node:test` for unit tests, no `eval`, no auto-registration via static-import side-effects (avoids ESM TDZ traps — inject factories explicitly)
