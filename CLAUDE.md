# FORGE Development Conventions

> These are the day-to-day operational conventions. For the project's durable
> mission, scope boundaries, and Seven Core Invariants, see
> [`docs/CHARTER.md`](docs/CHARTER.md) — read it before planning work.

## Build & Test Commands
- `cargo build --workspace` — Build all crates
- `cargo test --workspace` — Run all Rust tests
- `cargo clippy --workspace -- -D warnings` — Lint (must pass with zero warnings)
- `cargo fmt --check` — Format check
- `cargo bench -p forge-bench` — Run benchmarks
- `make wasm-check` — Clippy `crates/forge-wasm` for `wasm32-unknown-unknown` with `-D warnings` (CI: the blocking `wasm` job). Part of `make verify`; skips with an actionable message if the target isn't installed
- `make wasm` / `make wasm-test` — Build the browser bundle into `web/pkg/` via `scripts/build_wasm_demo.sh`, and run the crate's `#[wasm_bindgen_test]`s in a real wasm runtime via `scripts/wasm_test_node.sh`. Both need `wasm-pack` (`scripts/install_wasm_pack.sh`); `wasm-test` is deliberately **not** in `make verify` because it requires a network-installed binary
- `make web-e2e` — `node:test` unit coverage for `web/app.js`'s pure helpers (`tests/web-e2e/unit/`) plus Playwright driving the real demo page in Chromium (`tests/web-e2e/e2e/specs/`) against a fresh wasm-pack build. CI: the non-blocking `wasm-e2e` job. Needs `wasm-pack` + a Chromium download, so also **not** in `make verify`
- `python -m pytest tests/python/ -v` — Run Python tests using the local environment (e.g. Python 3.11.9) to bind core dependencies like `onnx`, `torch` and `onnxruntime` (requires `maturin develop` first)
- `cd mc-bot && npm test` — Run Node-side mc-bot unit/integration/security tests in strict TS using `tsx --test`
- `cd mc-bot && npm run lint` — Biome lint + format check on the TypeScript source
- `cargo run -p forge-mc-runner -- --dry-run --episodes 1` — Smoke-test the runner binary without docker / Minecraft (CI: `forge-mc-runner-bin` job)
- `cargo test -p forge-server -p forge-observability` — Server REST/WebSocket + history-endpoint tests (incl. `tests/history_endpoints_integration.rs` driven via `tower::oneshot`) and the shared tracing-init unit/doctests
- `FORGE_SERVER_HISTORY_DIR=/tmp/forge-history cargo run -p forge-server` — Run the visualization server (REST + WebSocket). All knobs are `FORGE_SERVER_*` env vars (bind `127.0.0.1:8080` — loopback by default since the API can reset/step the sim; history dir `forge-history/`, retention 10000, query limit 500, request timeout 30000ms, max body 1 MiB, optional `AUTH_TOKEN` gating the mutating routes); `FORGE_LOG_FORMAT=json` switches structured logging on
- `docker compose -f docker/compose.minecraft.yml --profile monitoring up -d prometheus grafana` — Opt-in Prometheus (`:9091`) + Grafana (`:3001`) stack scraping the runner's `forge_mc_*` metrics (needs `metrics_bind = "0.0.0.0"` in runner.toml; default `up` is unaffected)
- `cargo test -p forge-agent --features onnx` + `cargo test -p forge-mc-runner --features onnx-reload` — Exercise the `OnnxMuZeroModel::reload()` + `into_reload_fn` ONNX hot-reload surface
- `curl http://127.0.0.1:9090/metrics` — Scrape the runner's Prometheus endpoint (disabled by setting `metrics_port = 0` in the runner config; `metrics_bind` defaults to `127.0.0.1`)
- `python -m forge.training.muzero_mc.cli bootstrap --obs-dim N --action-dim M --schema-id <sha> --out models/` — Phase-5 random-init bundle
- `python -m forge.training.muzero_mc.cli validate-manifest <path>` — Validate a model_manifest.json (exit 0 / 3 / 4)
- `python -m forge.training.muzero_mc.cli train --input trajectories/ --out models/ --schema-id <sha> --obs-dim N --action-dim M --iters 100 --export-every 10` — Train loop consuming `.json` / `.json.gz` trajectories, periodic ONNX export + manifest bump that the runner's `HotReloadWatcher` picks up. `--obs-dim`, `--action-dim`, `--schema-id`, `--out`, `--input` are required; `--manifest` defaults to `<out>/model_manifest.json`
- `scripts/mc_run.sh --dry-run` — Print resolved docker compose argv for the stack (no side effects)
- `scripts/mc_run.sh --build` — Bring the Minecraft + mc-bot + runner stack up (foreground; Ctrl-C cleans up)
- `cd mc-bot && npm run typecheck` — `tsc --noEmit` strict typecheck gate over the `.ts` source (CI: `mc-bot-test` job gates on this)
- `pytest tests/python/integration/ -m minecraft_e2e -v` — Run the opt-in compose-stack E2E (requires docker; never runs on a default `pytest` invocation)
- `python -m forge.training.muzero_mc.cli compute-schema-id --action-map configs/minecraft/action_map.toml --rewards configs/minecraft/rewards.toml --quiet` — v0.4: print the canonical 64-hex schema_id to stdout (stderr-bound logs). Used by `mc_self_play.sh` to populate `FORGE_MC_SCHEMA_ID` before compose-up
- `python -m forge.training.muzero_mc.cli train --input trajectories/ --out models/ --schema-id <sha> --obs-dim N --action-dim M --continuous --round-iters 10 --round-poll-sleep 5 --max-trajectories 200 --max-bundle-versions 5 --device cpu` — v0.4 continuous trainer: yields one round summary per loop iteration, polls trajectory dir for new files (cold-start safe), exports an atomic `v{NNNNNNNN}/` bundle subdir + bumps manifest each round. SIGINT-clean shutdown
- `cargo build -p forge-mc-runner --features mc-live` — v0.5 live runner build. NOTE: `mc-live` no longer implies `onnx-reload` (v0.5 split — the random-baseline path doesn't need ORT). For trained-mode runs add `--features onnx-reload`, or use the convenience aggregate `--features mc-live-bundled` which adds `onnx-reload` + `ort/load-dynamic` for the docker image
- `scripts/mc_self_play.sh [--gpu] [--detach]` — v0.4 one-command orchestrator: preflights compose v2, computes schema_id via `trainer-bootstrap` one-shot, exports `FORGE_MC_SCHEMA_ID`, runs `bootstrap` if needed, brings up self-play profile with trained identity (`FORGE_MC_RANDOM_ACTIONS=false`, `FEATURES=mc-live-bundled`). `--baseline-only` leaves shipped `random_actions=true`. `--gpu` layers `compose.minecraft.gpu.yml`
- `scripts/mc_self_play.sh --dry-run` — Print every step's resolved docker-compose argv to STDERR + exit 0 (used by `tests/python/integration/test_mc_self_play_unit.py`)
- `pytest tests/python/integration/test_minecraft_self_improvement_smoke.py -v` — v0.4 self-improvement smoke (PR-CI gate; runs by default, skips if torch+onnx extras missing)
- `python -m forge.training.muzero_mc.cli capture-baseline --variant random|trained --episodes 100 --out baseline_<variant>.json [--metrics-url URL] [--trajectory-dir PATH] [--timeout-secs N] [--dry-run]` — v0.5 first-real-run baseline capture. Drives N episodes against a running stack, dumps snapshot JSON the plotter consumes. `scripts/mc_capture_baseline.py` is a thin shim
- `python scripts/mc_plot_baseline.py --random baseline_random.json --trained baseline_trained.json --out docs/results/v0.5-first-real-run.md [--no-plots]` — v0.5 Markdown report + matplotlib PNGs. Sources per-episode rewards from trajectory JSON (NOT Prometheus aggregates). `--no-plots` runs table-only on hosts without matplotlib
- `cargo run -p forge-mc-runner -- --random-actions --mc-config configs/minecraft/env.toml` — v0.5 random-actions runtime switch. Bypasses MCTS entirely; live.rs skips the ONNX bundle load. Used by `capture-baseline --variant random` via the env-var ladder
- `docker build -f docker/mc-runner.Dockerfile -t forge-mc-runner:dev .` — v0.5 runner image (rust:1.94.1-bookworm builder, 135 MB debian:bookworm-slim runtime, builds with `--features mc-live` — random-baseline-only). For trained mode rebuild with `--features mc-live-bundled` (builder stage installs ONNX Runtime `>=1.23.2` unconditionally; the `onnx-features` CI job builds and tests this feature surface on every push, though it doesn't build this Dockerfile itself — RUST_IMAGE_TAG is kept in sync with rust-toolchain.toml by convention, not by a CI check)
- `docker compose -f docker/compose.minecraft.yml --env-file docker/compose.minecraft.env up -d minecraft mc-bot runner` — v0.5 full stack bring-up. Compose mounts `configs/minecraft/env.docker.toml` over `env.toml` so the bot uses docker DNS hostnames (`bot.host = "minecraft"`, `ws_url = "ws://mc-bot:8765"`) instead of the local-dev `127.0.0.1` defaults
- `python scripts/v05_handshake_probe.py 127.0.0.1 8765` — v0.5 stdlib-only WS handshake probe. Connects to the live mc-bot, reads `Hello`, validates the v0.5 contract end-to-end (`obs_dim=920`, `grid_shape={11,11,1,7,73}`). Returns EXIT_OK / EXIT_GRID_SHAPE_MISSING / EXIT_GRID_SHAPE_MISMATCH for CI gating
- `python scripts/v05_manual_baseline.py --episodes N --max-steps-per-episode M --out PATH` — v0.5 Python-driven random baseline (stand-in for the Rust runner's `--random-actions` path while the trained-mode docker image is being plumbed through). Output JSON schema-compatible with `mc_plot_baseline.py`. Shares the `scripts/_ws_client.py` RFC 6455 frame parser with the handshake probe
- `scripts/mc_evidential_capture.sh [--dry-run] [--episodes N]` — operator trained-vs-random capture. `--dry-run` is the CI surface. Live path exits 3 without Docker and never invents `evidential_episodes >= 3`
- `UPDATE_GOLDEN_REPLAYS=1 cargo test -p forge-replay --test golden_replay` — regenerate CompactReplay v2 goldens after `ForgeConfig` field additions; then append `docs/results/replay_flip_log.md`. Do not combine UPDATE with a snapshot-read in the same cargo invocation (tests run in parallel)

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
- **forge-observability**: Shared tracing/log init (`init_tracing(TracingOptions)` / `try_init_tracing`). The single home for `tracing-subscriber` setup; `forge-server` + `forge-mc-runner` reuse it (no duplicated bootstrap). Output format is env-driven via `FORGE_LOG_FORMAT` (`text` default, `json`), filter via `RUST_LOG` with a per-binary default.
- **forge-server**: Axum HTTP + WebSocket server. REST env API (`/api/env/{reset,step,render}`), live broadcast, and a `history` module (`HistoryStore` trait + append-only `JsonlHistoryStore` + `InMemoryHistoryStore`) backing `POST`+persist on `/api/training-metrics` & `/api/decision-traces` and `GET /api/{training-metrics,decision-traces}/history` + `/api/runs`. Config via `FORGE_SERVER_*` env vars (incl. `HISTORY_DIR`/`_RETENTION`/`_QUERY_LIMIT`); run id resolved from `?runId=` → `X-Forge-Run-Id` → server-session id.
- **forge-civ**: Grid topology abstraction + pathfinding — square/hex `neighbor`/`distance` consumed by `forge-core`.
- **forge-memory**: Persistent agent memory — episodic, semantic, and preference stores with strength decay/eviction.
- **forge-social**: Social interaction primitives — trust + reputation models.
- **forge-cognitive**: LLM-backed cognitive agent — completion-provider abstraction + configs.
- **forge-proposal**: SBIR proposal template system — agency profiles, cost volumes, technical/validation sections, cover pages.
- **forge-mangomas**: MangoMAS agent-training integration — parameter sweeps, swarm, curriculum, adapters, transfer.
- **forge-eval**: Agent-agnostic evaluation harness — `Scorecard` + reproducibility manifest + MLflow / HuggingFace exporters (HTTP exporter behind the `http-mlflow` feature). Success requires attached `scenario_tasks` complete (not mere `terminated`).
- **forge-data**: Training-data loaders, dataset adapters, and expert-demo generation.
- **forge-integration** (package `forge-integration-layer`): Cross-layer integration orchestrator wiring the subsystems together.
- **forge-cloud**: Cloud training pipeline + edge deployment — workers, replay transport, storage backends (GCS via the `gcs` feature), model registry.
- **forge-edge**: Edge deployment runtime — inference + telemetry.
- **forge-wasm**: WebAssembly visualization module for the browser demo.
- **mc-bot/**: Node bridge (mineflayer + reward registry + reset + viewer + `createLogger` + `BotManager` heartbeat), out of the Cargo workspace.

## Key Principles
- **No hard-coded values**: All constants flow through config structs with `Default` impls
- **Deterministic**: Same seed + actions = identical state. Use `fixed` crate for physics, `rand_pcg` for RNG
- **Zero allocation on hot path**: `WorldState::step_into(&mut StepResult)` must not heap-allocate after warmup. The convenience `step()` wrapper allocates a fresh `StepResult` per call; reuse a buffer via `step_into` for the zero-alloc contract. Enforced in CI by `crates/forge-bench/src/bin/allocation_audit.rs` + `benchmarks/runner/check_zero_alloc.py`. **Carve-out**: wire-bound envs like `forge-env-mc::MinecraftEnv` implement the same `Env::step_into` method and reuse caller observation buffers, but internal WebSocket I/O and JSON parsing still allocate and are explicitly exempt from the core allocation audit. In-process envs such as `FlatForgeEnv` must honour the full buffer-reuse contract.
- **Structured logging**: Use `tracing` crate throughout. `#[instrument]` on public functions
- **Property-based tests**: Use `proptest` for invariant verification alongside unit tests
- **Cross-language schema_id contracts**: For configs shared between Rust, JS, and Python (`configs/minecraft/{action_map,rewards,milestone_rewards,crafting_rewards}.toml`), all three sides compute the same canonical sha256. Nested reward **file contents** fold into the rewards hash when loaded from disk; `block_embeddings.toml` is a **separate** obs-layout pin, not today's two-input `schema_id`. Pin a known-good value in paired tests (Rust `xlang_*_pinned_to_known_good` ↔ Node `xlang ... matches Rust` ↔ Python `test_muzero_mc_schema_id.py`). Drift on any side fails all three simultaneously. An advisory PreToolUse hook (`.claude/hooks/guard_schema_id_pins.py`) reminds on `git commit` when those TOML files are staged. High-level `[scenario]` TOML is a **second** xlang pin: Rust `crates/forge-types/src/scenario.rs` ↔ Python `tests/python/test_scenario_compiler_xlang_pin.py` for `orchard_coverage` and `crop_scout`. An advisory hook (`.claude/hooks/guard_golden_replay.py`) reminds to refresh `tests/golden/replays/` when scenario TOML or `ForgeConfig` sources are staged.

## Code Style
- Run `cargo fmt` before committing
- All public items must have doc comments
- Error types use `thiserror` derive macros
- Config structs derive `Clone, Debug, Serialize, Deserialize` and impl `Default`
- Use `tracing::{info, debug, warn, error, trace}` for logging, not `println!`
- JS code: ESM modules, Node 22+, `node:test` for unit tests, no `eval`, no auto-registration via static-import side-effects (avoids ESM TDZ traps — inject factories explicitly)
