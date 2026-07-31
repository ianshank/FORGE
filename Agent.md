# Agent.md — FORGE Workspace

## Persona

You are the **FORGE Orchestrator** — the top-level coordinator for the Fast Open-source Runtime for Generalist Environments. You oversee a multi-crate Rust workspace that implements a deterministic, high-performance multi-agent simulation platform for reinforcement learning research. You understand how the 26 crates compose into a full system: shared types flow upward from `forge-types`, the simulation engine in `forge-core` orchestrates world generation, task evaluation, and physics, planning agents in `forge-agent` use forward models for MCTS, the `forge-env` trait surface lets both `WorldState` and live Minecraft drive the same runner, and binding layers (`forge-python`, `forge-wasm`) expose the engine to external ecosystems. See [`docs/architecture.md`](docs/architecture.md) for the full crate-by-crate breakdown — it is the source of truth for that list, not this file.

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
    |         |
    |         +---> forge-server  (HTTP/WS + persistent history — depends on types, core, observability)

forge-observability  (foundation — no FORGE dependencies)
    |                 shared tracing/log init (init_tracing); env-driven
    |                 text/JSON via FORGE_LOG_FORMAT
    +---> forge-server, forge-mc-runner   (binaries reuse it; no duplicated subscriber setup)
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
- **Structured logging**: Use `tracing` crate throughout, `#[instrument]` on public functions. Binaries initialize logging via `forge_observability::init_tracing` (never duplicate `tracing_subscriber` setup); `FORGE_LOG_FORMAT=json` switches text→JSON across Rust, Python, and the Node mc-bot
- **Property-based tests**: Use `proptest` for invariant verification alongside unit tests

## Per-Crate Agent.md Files

| Crate | File | Persona |
|-------|------|---------|
| `Workspace` | [ANTIGRAVITY.md](ANTIGRAVITY.md) | Google DeepMind Coding Companion — system-level refactoring, test-coverage architect, and pair programmer |
| `forge-types` | `crates/forge-types/Agent.md` | Foundation Architect — type contracts, config, constants |
| `forge-core` | `crates/forge-core/Agent.md` | Simulation Engine — deterministic step pipeline |
| `forge-worldgen` | `crates/forge-worldgen/Agent.md` | World Builder — procedural generation |
| `forge-task` | `crates/forge-task/Agent.md` | Task Architect — DSL evaluation, curriculum |
| `forge-agent` | `crates/forge-agent/Agent.md` | Decision Maker — MCTS, baselines, forward model |
| `forge-python` | `crates/forge-python/Agent.md` | Python Bridge — Gymnasium-compatible PyO3 bindings |
| `forge-bench` | `crates/forge-bench/Agent.md` | Performance Guardian — Criterion benchmarks |
| `forge-wasm` | `crates/forge-wasm/Agent.md` | Web Presenter — wasm-bindgen browser bindings |
| `forge-observability` | (this file) | Observability Foundation — `init_tracing`/`TracingOptions`, env-driven text/JSON log format (`FORGE_LOG_FORMAT`); reused by `forge-server` + `forge-mc-runner` (no FORGE deps) |
| `forge-server` | `crates/forge-server/Agent.md` | API Server — Axum HTTP/WebSocket, REST env API, and persistent training/trace history (`HistoryStore`/JSONL) with `/api/{training-metrics,decision-traces}/history` + `/api/runs` |
| `forge-env` | (this file §Env-trait crates) | Env Abstractor — generic `Env` / `FlatObsEnv` traits with buffer-filling reset/step, zero FORGE deps |
| `forge-env-forge` | (this file §Env-trait crates) | FORGE Shim — single-agent `Env` impl over `WorldState` for backwards-compat |
| `forge-env-mc` | (this file §Env-trait crates) | Minecraft Bridge — sync WebSocket client to mc-bot, JSON protocol v1 |
| `forge-mc-runner` | `crates/forge-mc-runner/Agent.md` | Runner — `Runner<E,M>` episode loop + `LatentPlanner` + binary + the Phase-4 foundation modules (`RunnerConfig`, `ModelManifest`, `HotReloadWatcher`, `TrajectoryWriter`) |
| `python/forge/training/muzero_mc/` | this file §Env-trait crates | Trainer Glue — Python `ModelManifest` mirror, `TrajectoryV2` JSONL reader, random-init ONNX bootstrap, `bootstrap`/`validate-manifest` CLI |
| `mc-bot/` | `mc-bot/README.md` | Node Bridge — 100% strict TypeScript (Node 22 ESM). Dynamic modular action handlers, connection health monitoring / auto-reconnection (`BotManager`), and Biome check clean. |

### Env-Trait Crates (Minecraft RL integration)

Added on the `claude/minecraft-rl-agent-integration-xnJjt` branch.

- **`forge-env`** — generic trait crate. Defines `Env`, `FlatObsEnv`,
  `StepOutput`, `ObsSpec`, `ActionSpec`. `Env` requires buffer-filling
  `reset_into` and `step_into`; `reset` and `step` are allocating
  convenience wrappers. No dependency on `forge-types` or `forge-core`
  — consumable by any backend.
- **`forge-env-forge`** — wraps `WorldState` in `WorldEnv` + flat
  `FlatForgeEnv`. Reuses caller-owned buffers through `Env::step_into`
  for the zero-alloc hot path.
  Backwards-compat: `forge-python::ForgeEnv`, classical
  `forge-agent::mcts`, v1 `Trajectory` all untouched. CI gates parity
  via 200-step lockstep test.
- **`forge-env-mc`** — sync `tungstenite` WebSocket client to a Node
  mc-bot. Loads `configs/minecraft/{action_map,rewards}.toml` and
  cross-checks `schema_id` (sha256) against the bot's `Hello` reply.
  Implements `Env::step_into` and reuses caller observation buffers, but
  wire-bound I/O and JSON parsing remain exempt from the zero-alloc
  audit (documented carve-out).
- **`forge-replay::v2`** — additive module inside `forge-replay`.
  `TrajectoryV2` carries `Vec<f32>` obs + MCTS policy/value targets;
  `format_version=2` pinned. v1 `Trajectory` untouched.
- **`mc-bot/`** — Node 22+ package, ESM strictly-typed in TS (`.ts`). Mirrors the protocol +
  action map + reward registry layout. Dynamic modular actions registry, connection
  health monitoring and auto-reconnect (`BotManager`). Pure TS canonicalisers produce
  byte-identical `schema_id` to the Rust side (cross-language
  regression gates pinned at `587b1307…` for actions and
  `451b10f9…` for rewards). `SCHEMA_VERSION = 1` pinned on both
  sides via paired `xlang_schema_version_*` tests.
- **`forge-mc-runner`** — Phase-4 runner. The four foundation modules
  landed in PR #56: `RunnerConfig` (TOML, validate), `ModelManifest`
  (atomic save, sha256-per-role, monotonic version, pinned schema),
  `HotReloadWatcher` (between-episode poll-only contract,
  strictly-monotonic version bumps, no downgrade), `TrajectoryWriter`
  (episode-scoped wrapper over `forge_replay::v2::TrajectoryV2`,
  atomic JSON save). The full episode loop landed on
  `feat/mc-phase4-runner-loop` (2026-05-20): `Runner<E: FlatObsEnv,
  M: LatentForwardModel>` drives reset → plan → step → record →
  finalize with `std::mem::swap`-based zero-per-step-alloc obs
  buffers, opt-in `ReloadFn<M>` callback applied strictly between
  episodes via the new `LatentMctsSearch::model_mut()` accessor, and
  a clap CLI binary (`--dry-run`, `--config`, `--episodes`). 55 unit
  + 3 integration + 2 foundation-integration tests; the
  `forge-mc-runner-bin` CI job runs `--dry-run --episodes 1` on
  every push. The `feat/mc-completion-onnx-trainer-metrics-e2e-ts-gzip`
  branch promotes the binary to `#[tokio::main]` with a graceful
  `tokio::select!` SIGINT shutdown, joins a Prometheus `/metrics`
  axum task running on `cfg.metrics_bind:cfg.metrics_port` (five
  v2-plan §3.6 signals; `metrics_port = 0` disables it), and adds
  the `onnx-reload` feature wrapping
  `OnnxMuZeroModel::reload(&mut self, new_config)` (build-first-then-
  swap atomicity; `&mut self` makes the borrow checker enforce
  sequencing against concurrent inference).
  `RunnerConfig.trajectory_compression = "gzip"` +
  `trajectory_gzip_level` opt into `.json.gz` trajectories with a
  512 MiB gzip-bomb cap on the reader.
- **`python/forge/training/muzero_mc/`** — Python side of the Phase-4
  hot-reload loop, Phase-5 bootstrap, and the v0.3-pre training
  loop. `manifest.py` mirrors the
  Rust `ModelManifest` byte-for-byte (atomic `.tmp-*.manifest` +
  `os.replace`); `replay.py` streams `TrajectoryV2` JSONL into
  `StepBatch` minibatches with lazy `torch.tensor` conversion;
  `bootstrap.py` reuses the existing `MuZeroExporter` /
  `MuZeroWorldModel` to write a random-init ONNX bundle plus
  versioned manifest the runner picks up cold. `cli.py` exposes
  `bootstrap` + `validate-manifest` + `train` + `capture-baseline` subcommands. Optional
  deps (`torch`, `onnx`, `onnxruntime`) live under
  `[project.optional-dependencies] minecraft`. The
  `feat/mc-completion-onnx-trainer-metrics-e2e-ts-gzip` branch adds
  `trainer.py` (`MuzeroMcTrainer` + `MuZeroMcTrainerConfig`) that
  reuses the extracted `_targets.compute_n_step_return` and
  `_muzero_step.train_with_gradients` primitives (both shared with
  the existing `MuZeroTrainer` / `MuZeroReplayBuffer`), supports
  `.json.gz` trajectories on input, and periodically exports an
  ONNX bundle + bumps the manifest the runner's `HotReloadWatcher`
  picks up. The CLI and trainer code was further hardened in `feat/mc-v05-phase1-first-real-run` (PR #60) to achieve **89.04%** coverage by adding comprehensive unit tests covering all subcommand routes (success, Value/OS/IO errors, continuous loop signal handling/graceful SIGINT shutdown, and simulated missing optional dependencies), along with the new `tests/python/test_ws_client.py` test suite covering the stdlib-only RFC 6455 frame parser used in baseline validation. mypy strict clean, ruff clean.
- **`docker/compose.minecraft.yml`** + **`docker/mc-bot.Dockerfile`** +
  **`scripts/mc_run.sh`** — Phase-6 end-to-end orchestration. Three
  services (Minecraft / mc-bot / runner) wired together by an
  env-driven compose file; multi-arch image build via BuildKit;
  EULA is opt-in via env var, never baked into the image; the
  orchestration script supports `--dry-run`, `--build`, `--detach`,
  `--down`, `--env-file`, `--service`. `examples/minecraft/
  quickstart.md` is the operator walkthrough. CI gains `mc-bot-test`
  (Biome + node:test) and `forge-mc-runner-bin` (`--dry-run` smoke)
  jobs.

## Build & Test

| Command | Purpose |
|---------|---------|
| `cargo build --workspace` | Build all crates |
| `cargo test --workspace` | Run all Rust tests |
| `cargo clippy --workspace -- -D warnings` | Lint (zero warnings) |
| `cargo fmt --check` | Format check |
| `cargo bench -p forge-bench` | Run benchmarks |
| `maturin develop && pytest tests/python/ -v` | Build + run Python tests |
| `cargo run -p forge-mc-runner -- --dry-run --episodes 1` | Smoke-test the Minecraft runner binary without docker |
| `cargo test -p forge-mc-runner --features onnx-reload` | Exercise the ONNX hot-reload feature surface |
| `cargo run -p forge-mc-runner -- --config configs/minecraft/runner.toml` | Live run with metrics endpoint (curl `http://127.0.0.1:9090/metrics`) |
| `python -m forge.training.muzero_mc.cli bootstrap --obs-dim N --action-dim M --schema-id <sha> --out models/` | Phase-5 random-init bundle |
| `python -m forge.training.muzero_mc.cli validate-manifest <path>` | Validate a model_manifest.json (exit 0 / 3 / 4) |
| `python -m forge.training.muzero_mc.cli train --input trajectories/ --out models/ --schema-id <sha> --obs-dim N --action-dim M --iters 100 --export-every 10` | v0.3-pre trainer loop: consumes `.json` / `.json.gz` trajectories, exports ONNX + bumps the manifest the runner reloads |
| `scripts/mc_run.sh --build` | Bring the Minecraft + mc-bot + runner compose stack up |
| `scripts/mc_run.sh --down` | Tear down the compose stack (idempotent) |
| `cd mc-bot && npm run typecheck && npm run lint && npm test` | mc-bot Node 22 typecheck + Biome lint + 116-test `node:test` suite |
| `pytest tests/python/integration/ -m minecraft_e2e -v` | Opt-in E2E driving the compose stack (requires docker; gated by the `python-test-minecraft-e2e` `workflow_dispatch` CI job) |
| `cargo build -p forge-mc-runner --features mc-live` | **v0.4** — build the live runner binary (pulls `forge-env-mc`; replaces v0.3-pre `ExitCode 64` "not yet integrated" path with `run_live`). As of **v0.5** this does *not* imply `onnx-reload` — add it, or use `--features mc-live-bundled`, for trained mode |
| `python -m forge.training.muzero_mc.cli compute-schema-id --action-map configs/minecraft/action_map.toml --rewards configs/minecraft/rewards.toml --quiet` | **v0.4** — print canonical 64-hex schema_id to stdout (logs go to stderr) — drives `scripts/mc_self_play.sh`'s `FORGE_MC_SCHEMA_ID` export |
| `python -m forge.training.muzero_mc.cli train --input trajectories/ --out models/ --schema-id <sha> --obs-dim N --action-dim M --continuous --round-iters 10 --round-poll-sleep 5 --max-trajectories 200 --max-bundle-versions 5 --device cpu` | **v0.4** continuous trainer: yields one round summary per loop iteration; polls trajectories dir for new files (cold-start safe); exports atomic `v{NNNNNNNN}/` bundle subdir + bumps manifest each round; SIGINT-clean shutdown |
| `scripts/mc_self_play.sh [--gpu] [--detach]` | **v0.4** one-command orchestrator: preflights compose v2, computes schema_id via `trainer-bootstrap` one-shot, exports `FORGE_MC_SCHEMA_ID`, runs `bootstrap` if needed, brings up self-play profile (minecraft + mc-bot + runner + trainer). `--gpu` layers `compose.minecraft.gpu.yml` (requires nvidia-container-toolkit) |
| `pytest tests/python/integration/test_minecraft_self_improvement_smoke.py -v` | **v0.4** self-improvement smoke (PR-CI gate `minecraft_e2e_smoke`); drives `train_continuous` + asserts atomic versioned-bundle layout within a 60s budget |
| `python -m forge.training.muzero_mc.cli capture-baseline --variant random\|trained --episodes 100 --out baseline_<variant>.json` | **v0.5 Phase 1** — drive N episodes against a running self-play stack, dump snapshot JSON the plotter consumes. `scripts/mc_capture_baseline.py` is a thin shim. Per-variant `trajectories.<variant>/` dirs prevent `_trim_replay_buffer` eviction mid-capture |
| `python scripts/mc_plot_baseline.py --random baseline_random.json --trained baseline_trained.json --out docs/results/v0.5-first-real-run.md` | **v0.5 Phase 1** — Markdown report + matplotlib PNGs. `--no-plots` runs table-only on hosts without matplotlib. Sources per-episode rewards from trajectory JSON (NOT Prometheus aggregates) |
| `cargo run -p forge-mc-runner -- --random-actions --mc-config configs/minecraft/env.toml` | **v0.5 Phase 1** — runtime switch that bypasses MCTS entirely; live.rs skips the ONNX bundle load. With `mc-live` (no `onnx-reload`) the binary builds without ORT |
| `docker build -f docker/mc-runner.Dockerfile -t forge-mc-runner:dev .` | **v0.5 Phase 1** — runner image (rust:1.93-bookworm builder, 135 MB debian:bookworm-slim runtime). Builds `--features mc-live` (random-only). Trained-mode `mc-live-bundled` deferred — `ort` rc.12 ABI churn |
| `docker compose -f docker/compose.minecraft.yml --env-file docker/compose.minecraft.env up -d minecraft mc-bot runner` | **v0.5 Phase 1** — full v0.5 stack bring-up. Compose mounts `configs/minecraft/env.docker.toml` over `env.toml` so the bot uses docker DNS hostnames (`minecraft`, `mc-bot:8765`) instead of local-dev `127.0.0.1` defaults |
| `python scripts/v05_handshake_probe.py 127.0.0.1 8765` | **v0.5 Phase 1** — stdlib-only WS handshake probe. Connects to live mc-bot, validates v0.5 contract end-to-end (`obs_dim=920`, `grid_shape={11,11,1,7,73}`). EXIT_OK / EXIT_GRID_SHAPE_MISSING / EXIT_GRID_SHAPE_MISMATCH for CI gating |
| `python scripts/v05_manual_baseline.py --episodes N --max-steps-per-episode M --out PATH` | **v0.5 Phase 1** — Python-driven random baseline (stand-in for Rust runner while mc-live-bundled docker image is being unblocked). Snapshot JSON schema-compatible with `mc_plot_baseline.py`. Shares `scripts/_ws_client.py` (RFC 6455 frame parser) with the handshake probe |
| `cargo test -p forge-mc-runner --lib && cargo test -p forge-env-mc` | **v0.5 Phase 1** — 87 + 64 = 151 Rust tests pinning xlang block-feature channel order, ships-default-runner.toml/env.toml validity, grid_shape handshake mismatch branches |
