# Changelog

All notable changes to FORGE will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

#### v0.2 Implementation Hardening

- Added **5 hex-grid integration tests** covering full episode cycle, deterministic replay, multi-agent episodes, square-move rejection, and serialization roundtrip (`tests/rust/integration_tests.rs` — 29 total passing)
- Added **2 hex benchmark groups** (`bench_step_hex_single_agent`, `bench_step_hex_multi_agent`) to `forge-bench` Criterion benchmarks (7 groups total)
- Added **22 new `forge-replay` tests** across compact replay, trajectory, export, and config modules (36 → 58 tests)
- Added **16 new `forge-scenario` tests** across registry, compose, and config modules (43 → 59 tests)
- Added **`tests/python/test_mangomas_smoke.py`** with 21 end-to-end smoke tests covering:
  - TOML config loading for all 8 `configs/mangomas/*.toml` files
  - Config-to-episode wiring for action adapters, observation adapters, and batch collection
  - Pipeline stage integration for BDI, constitutional, RSSM, curiosity, and curriculum controllers
  - Export pipeline integrity with full roundtrip and weight loading verification
  - End-to-end pipeline execution through the BDI stage
- Added **`python-test-fast` CI job** — runs pure-Python tests without maturin/native extension build for faster PR feedback
- Added **Docker GHCR publishing job** with multi-arch support (`linux/amd64` + `linux/arm64`), Docker Buildx, semver tag extraction, and GitHub Actions cache

### Changed

- **Benchmark regression gate**: `critcmp` comparison changed from warning to hard failure — regressions >5% now block PRs
- **Python type compliance**: Fixed all 10 mypy errors across `muzero_buffer.py`, `wrappers.py`, `muzero_mcts.py`, and `pyproject.toml` override configuration — 86 source files pass mypy strict
- **Python formatting**: Applied `ruff format` across 40 Python files; resolved all E501 line-length violations
- **`.gitignore`**: Added entries for stale build artifacts (`clippy_output.txt`, `demo_results.md`)

### Fixed

- Fixed `forge-scenario` registry tests using incorrect data assumptions (OR-logic for tag matching, correct scenario names)
- Fixed `compose_scenarios` test expecting `Result` when function returns `Option`
- Fixed mypy `# type: ignore[return-value]` → `[no-any-return]` for 5 return statements in `muzero_mcts.py`
- Fixed `numpy.signedinteger` → `int` cast in `muzero_buffer.py` for mypy compliance
- Removed stale `# type: ignore[no-any-return]` from `wrappers.py` after numpy stub alignment

---

### Added

#### Hex Grid Topology & Dynamic Action Spaces

- Added **`forge-civ`** as a dedicated topology crate for square and hex grids, including:
  - `GridTopology` / `GridTopologyKind` dispatch for square and odd-r hex worlds
  - shared neighbor lookup, distance, line-of-sight, disk queries, and A* pathfinding
  - topology-focused unit and property coverage for square and hex behaviors
- Added `GridType` to world config plus `MoveHex(HexDirection)` support in `forge-types`
- Added config-aware action encoding/decoding across Rust, Python, WASM, evaluation, and MangoMAS runners so discrete action IDs match the active grid/drone/agri layout
- Added `configs/scenarios/hex_patrol.toml` as a reusable hex-grid scenario fixture

#### MangoMAS Collection & Pipeline Control Plane

- Added `python/forge/mangomas/collector.py` for scenario resolution, batch FORGE rollout collection, and JSON collection-report generation
- Added `python/forge/mangomas/pipeline.py` for stage-based MangoMAS orchestration covering BDI, constitutional, RSSM, curiosity, sweep, curriculum, and export stages
- Added missing MangoMAS config surfaces for pipeline paths/logging/execution and transfer overrides, with TOML parsing support
- Added MangoMAS CLI entry points in `scripts/train.py`:
  - `--agent mangomas-collect` for collection-only workflows
  - `--agent mangomas` for collection plus stage-pipeline execution
  - scenario, collection-policy, pipeline output, and collection-report flags

#### Regression Coverage

- Added focused Python regression coverage for MangoMAS config parsing, collector decoding, pipeline manifests, and training CLI argument parsing
- Added targeted hex-grid action-space and visibility regressions in Rust and Python so square-only assumptions fail fast during review

#### Cloud Training Pipeline & Edge Deployment (`forge-cloud`, `forge-edge`)

- **`forge-cloud` crate** (125 tests): Distributed cloud training infrastructure.
  - Config types: `CloudConfig`, `WorkerConfig`, `CoordinatorConfig`, `StorageConfig`, `ReplayTransportConfig`, `ModelRegistryConfig` with `AggregationStrategy`, `StorageBackend`, `FallbackPolicy` enums.
  - Error types: `CloudError`, `WorkerError`, `TransportError`, `StorageError`, `ModelRegistryError` with `CloudResult<T>` alias.
  - Core traits: `ReplayStore`, `ModelStore`, `WorkerManager` with `WorkerMetadata`, `WorkerInfo`, `WorkerStatus`, `SeedAssignment` data types.
  - `InMemoryWorkerRegistry`: Thread-safe in-memory `WorkerManager` implementation for testing and single-node deployments.
  - `LocalReplayStore`: Filesystem-backed `ReplayStore` for CompactReplay persistence.
  - `LocalModelStore`: Filesystem-backed model store implementing both `forge-cloud::ModelStore` (u32 versions) and `forge-types::transport::ModelStore` (string versions).
  - `TrajectoryReconstructor`: Reconstructs full `Trajectory` objects from `CompactReplay` via deterministic replay, with batch support producing `OfflineDataset`.
  - Replay compression/decompression and `ReplayBatch` for transport.
  - 22 constants with `DEFAULT_*` prefix and validation tests.

- **`forge-edge` crate** (40 tests): Edge deployment runtime.
  - `AdaptiveMctsSearch<M>`: Wraps `LatentMctsSearch` from forge-agent with latency budgeting. Estimates per-simulation cost via EMA, adjusts `num_simulations` per search call, clamps to `[min, max]` from `EdgeConfig`.
  - `LatencyEstimator`: EMA-based per-simulation latency tracker with configurable alpha.
  - `TelemetryCollector`: Bounded store-and-forward buffer for `CompactReplay` with flush via `ReplayTransport` trait.
  - `EdgeAgent<M>`: Composite agent implementing `AgentInterface` — flattens observations, runs adaptive MCTS, falls back to Noop on error. Compatible with `EvalHarness`, `BatchRunner`, and all FORGE infrastructure.
  - `AdaptiveSearchMetrics` and `TelemetrySnapshot` diagnostic types.

- **Foundation types in `forge-types`**:
  - `CloudConfig` and `EdgeConfig` added to `ForgeConfig` with `#[serde(default)]` for full backward compatibility.
  - `CloudError` (7 variants) and `EdgeError` (5 variants) with `#[from]` conversions.
  - `ReplayTransport` and `ModelStore` traits in new `transport` module.
  - 25 `DEFAULT_CLOUD_*` and `DEFAULT_EDGE_*` constants with validation tests.
  - Environment variable overrides for `cloud.*` and `edge.*` config sections.

- **`EdgeReplayLoader` in `forge-data`** (18 tests): Implements `DatasetLoader` for edge telemetry ingestion — scans directory for `.bin` CompactReplay files, reconstructs trajectories, returns `OfflineDataset`.

- **Integration test** (`tests/rust/integration_cloud_edge.rs`, 9 tests): Exercises full cloud-edge data loop including storage roundtrips, reconstruction determinism, EdgeAgent evaluation, telemetry flush, worker lifecycle, backward compatibility, and model version management.

- **Distributed Docker** (`docker/docker-compose.distributed.yml`): Coordinator + scalable worker services with shared volumes and `FORGE_CLOUD_*` / `FORGE_EDGE_*` environment variable configuration.

- **Training config** (`configs/training/distributed.toml`): Complete distributed training configuration with cloud and edge sections enabled.

- **Proposal** (`docs/cloud_edge_proposal.md`): GCP-specific technical proposal with MouseDroidAGI flagship use case, architecture diagrams, cost analysis, and 4-phase implementation roadmap.

#### MuZero Latent-Space Planning & ONNX Integration

- Embedded a full `MuZeroWorldModel` across Python and Rust for evaluating search in latent-space environments.
- Added `forge_agent::latent_mcts` with `LatentMctsSearch`, capable of dynamically routing inference through a generic `LatentForwardModel` trait.
- Added `OnnxMuZeroModel` backend utilizing `ort` (ONNX Runtime v2) with safe mutex session management for E2E MCTS evaluations.
- Implemented `MuZeroExporter` to natively convert the multi-head PyTorch MuZero network instances into `.onnx` binaries natively.
- Developed an end-to-end integration test (`onnx_integration.rs`) to automatically orchestrate Python ONNX export and Rust latent graph search.

#### MangoMAS Bridge Coverage And Training Surface

- Added targeted Python coverage for the MangoMAS bridge components: constitutional pre-training, adaptive curriculum control, curiosity-weight optimization, and MCTS sweep reporting
- Added a config-driven discrete SAC training preset in `configs/training/sac_default.toml` and expanded `examples/train_sac_cleanrl.py` to honor TOML-backed model and feature-extractor settings
- Added branch-specific regression tests for vectorized env wrappers, feature extractors, and pure-Python `forge_env` import/fallback behavior

#### Python Coverage Expansion

- Added `tests/python/test_device.py` to cover accelerator detection paths in `forge.utils.device`
- Expanded `tests/python/test_mappo.py` with config-factory, auto-device, batched action, and `RandomPolicyNetwork` coverage
- Expanded `tests/python/test_forge_env.py` to exercise `forge_env.__init__`, `forge_env.utils`, wrapper edge cases, and pure-Python fallback branches

### Changed

#### MangoMAS Configuration Hardening

- Consolidated MangoMAS bridge defaults into `python/forge/mangomas/config.py` so curriculum tiers, constitutional constraints, curiosity weights, sweep bounds, and batch collection settings all flow from configuration objects
- Updated the constitutional trainer, curiosity optimizer, curriculum controller, batch collector, and sweep runner to consume shared config defaults instead of duplicating literals in module code
- Expanded `python/forge_env/__init__.py`, `feature_extractors.py`, and `vecenv.py` to better tolerate optional native or ML dependencies while keeping the package importable for pure-Python validation

#### Rust Coverage Hardening

- Converted naive `unwrap()/expect()` calls inside `onnx_model.rs` and `LatentForwardModel` into robust `anyhow::Result` boundaries bubbled up through the `search` pipeline.
- Added targeted Rust coverage for MCTS terminal-search and short-priors fallback behavior in `forge-agent`
- Fixed outdated 2-argument signature definitions wrapped by duplicate `mod proptests` in `action.rs` and `config.rs`.
- Expanded predicate, validation, and terrain edge-case coverage across `forge-task`, `forge-types`, and `forge-worldgen`

#### Python Gap Analysis Cleanup

- Replaced remaining hard-coded Python values with named constants in the Gymnasium env wrapper, MAPPO reward normalization, trainer checkpoint defaults, dashboard client tests, and shared pytest fixtures
- Enforced a Python coverage floor with `pytest --cov-fail-under=85` in `pyproject.toml`
- Standardized Python test fixtures and assertions around exported wrapper constants instead of duplicated literals

#### Docker Multi-Service Deployment (`docker/`)

Production-ready Docker Compose stack with three independently deployed services:

**Simulation Service** (`docker/Dockerfile`)

- Upgraded Rust base image to `1.85` (required for `fixed` crate edition 2024)
- Builds `forge-server` binary via multi-stage `rust:1.85-bookworm` → `python:3.11-slim-bookworm`
- Builds `forge_env` native Python extension via `maturin build -m crates/forge-python/Cargo.toml`
- `HEALTHCHECK` on `/health` endpoint with 15s interval, 3s timeout, 3 retries

**Dashboard Service** (`docker/Dockerfile.dashboard`, `docker/nginx.conf`)

- Dedicated `node:20 → nginx:1.27-alpine` multi-stage image (~40 MB vs monolithic)
- Nginx serves the React SPA with SPA routing (all paths → `index.html`)
- Reverse proxies `/api/` and `/ws` → `simulation:8080` for same-origin access
- `/healthz` endpoint to satisfy Docker health checks

**Demo UI Service** (`docker/Dockerfile.demo`)

- No changes to the Dockerfile, but fully integrated into the new Compose stack
- Exposed on `http://localhost:8765`

**Orchestration** (`docker/docker-compose.yml`)

- Bridge network `forge-net` for inter-service communication by name
- Health-gated `depends_on`: dashboard + demo wait for `simulation` to be `healthy`
- `restart: unless-stopped` for production resilience
- All ports bound to `127.0.0.1` for security

**Build Context** (`.dockerignore`)

- Excludes `target/`, `node_modules/`, `.git/`, caches, and coverage artifacts

#### Dashboard TypeScript Fixes

- `tsconfig.json`: Added `"types": ["vite/client"]` for `import.meta.env` recognition
- `tsconfig.node.json`: Added `"types": ["node"]` for `process.env` in `vite.config.ts`
- `vite.config.ts`: Added `/// <reference types="vitest" />` triple-slash directive
- `App.tsx`: Prefixed unused `setSelectedAgent` → `_setSelectedAgent` (`noUnusedLocals`)
- `package.json`: Added `@types/node` devDependency

#### Python Test Quality

- Fixed `# noqa: PLC0415` directives across test files (ruff RUF100 auto-fix)
- Fixed `TC003` in `test_gymnasium_env.py`: moved `Generator` import into `TYPE_CHECKING` block
- Improved type annotations from `object` → specific env types (`ForgeGymnasiumEnv`, `ForgeParallelEnv`)

### Deployment URLs

After `docker compose -f docker/docker-compose.yml up -d`:

| Service | URL | Health |
|---------|-----|--------|
| Simulation (Rust/Axum) | `http://localhost:8080` | `GET /health` |
| Dashboard (React/nginx) | `http://localhost:3000` | nginx |
| Demo UI (FastAPI) | `http://localhost:8765` | `GET /health` |

#### Interactive Demo UI (`demo_ui/`)

A full-stack interactive web application that streams the FORGE demo live in the browser.

**Backend** (`demo_ui/backend/`)

- `main.py` — FastAPI application with Server-Sent Events (SSE) endpoints:
  - `GET /` — serves the single-page frontend
  - `GET /health` — liveness check
  - `GET /api/sections` — list all 8 demo sections (key, name, index)
  - `POST /api/run/{section}` — stream a single section's output via SSE
  - `POST /api/run-all` — stream all 8 sections sequentially via SSE
  - `GET /api/results` — return parsed `demo_results.md` as JSON
- `forge_runner.py` — async subprocess wrapper around `forge_demo.py`:
  - `run_section()` async generator — streams stdout line by line
  - `run_all()` — emits `__SECTION_START__`/`__SECTION_END__` sentinel tokens
  - `parse_results_md()` — regex-based parser for baseline results markdown
  - `SECTIONS` dict mapping slug → display name for all 8 demo sections

**Frontend** (`demo_ui/frontend/`)

- `index.html` — single-page app with header, sidebar nav, live terminal, stats panel, footer controls
- `styles.css` — premium dark-mode design system: glassmorphism cards, color tokens (`--cyan`, `--pass`, `--fail`), `fade-in` micro-animations, responsive grid layout
- `app.js` — modular vanilla JS:
  - `ForgeTerminal` — ANSI-aware live terminal with syntax highlighting (PASS/FAIL colors, grid colorization)
  - `WorldRenderer` — ASCII grid → canvas renderer using `TERRAIN_COLORS` palette
  - `SectionNav` — sidebar state machine (idle → running → pass/fail) with mini results panel
  - `Runner` — SSE orchestrator: manages `AbortController`, labeled-loop sentinel parsing, progress bar, and timer

**Tests** (`demo_ui/tests/`)

- `test_backend.py` — 15 unit + integration tests (all passing):
  - `parse_results_md()` — structure, values, 8 sections, performance metrics, missing file
  - API endpoints — `/health`, `/api/sections` (count/keys/schema), `/api/results`, 404 for unknown section
  - SSE streams — `/api/run/worldgen` and `/api/run-all` return `text/event-stream`
  - `SECTIONS` constant — is dict with 8 expected keys
- `test_sections.py` — functional tests for output presence and keyword validation per section
- `conftest.py` + `pytest.ini` — asyncio-auto mode, `anyio` backend

#### Launcher

- `run_demo.ps1` — PowerShell one-click launcher:
  - Installs `demo_ui/backend/requirements.txt`
  - Starts `uvicorn` on `http://127.0.0.1:8765`
  - Opens browser automatically

#### Root Fixes

- `conftest.py` (repo root) — injects FORGE root into `sys.path` so `demo_ui` is importable from any CWD
- `.gitignore` — added `demo_ui` artifact exclusions, `.pytest_cache/`, `.claude/`

### Performance (from baseline run)

- 8/8 demo sections pass in `Quick` mode (~3.4 seconds total)
- 136,419 steps/second, 7.33 μs/step, zero console errors

---

## [0.1.0] — 2026-02-26

### Added

- Initial FORGE platform release
  - `forge-core` — deterministic simulation engine (13-phase pipeline, zero-alloc hot path)
  - `forge-worldgen` — Perlin noise procedural world generation (7 biomes)
  - `forge-task` — composable task DSL (7 operators, 10 predicates, 6 tiers, adaptive curriculum)
  - `forge-agent` — MCTS planner (PUCT selection, pluggable policy/value, forward model)
  - `forge-python` — PyO3 bindings with NumPy observations and GIL release during `step()`
  - `forge-wasm` — wasm-bindgen bindings with JSON string I/O for browser environments
  - `forge-types` — shared types, configs, error definitions
  - `forge-bench` — Criterion benchmarks for step throughput and world creation
  - Python wrappers: Gymnasium, PettingZoo Parallel, JAX-vectorized, Flatten/Normalize/TimeLimit
  - 389 tests (unit + property-based via `proptest` + integration)
  - 130,000+ steps/second from Python, <8 μs/step including PyO3 overhead
  - Deterministic: same seed + actions = byte-identical results
  - WebAssembly support via `forge-wasm` crate
  - `examples/forge_demo.py` — comprehensive 8-section showcase
