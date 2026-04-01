# FORGE Implementation Plan — Next Steps (March 2026)

## Context

FORGE is an 18-crate Rust workspace (simulation sandbox for training AI agents) with ~1,253 Rust tests all passing, clean clippy, and a CI pipeline. The project has progressed through 21 PRs covering core engine hardening (v0.1), extended crate scaffolding (memory, social, cognitive, integration, procgen, server), and newest additions (eval, scenario, replay, mangomas). One PR remains open (#2: Sprint 6-7 features). The existing 7-phase implementation plan (`docs/implementation_plan.md`) has Phases 0-1 complete and Phase 2 partially done. This plan picks up from current state and drives toward v0.2, v0.3, and v1.0.

### Current State Summary
- **All 18 crates compile and pass clippy** with zero warnings
- **~1,253 Rust tests pass**, 0 failures
- **CI pipeline** exists (`.github/workflows/ci.yml`) with fmt, clippy, test, bench, python, demo-ui jobs
- **PR #2 (open)**: Sprint 6-7 features (mypy strict, replay/record, Docker, multi-agent dashboard) — 9,210 additions, 54 files, targeting non-main branch
- **Python ecosystem**: 48 test files, MangoMAS bridge, training scripts, demo UI
- **Key gap**: `forge-integration-layer` has only 23 tests; newest crates (eval: 29, scenario: 41, replay: 30) need deeper coverage

---

## Phase 1: Resolve PR #2 and Establish Clean Baseline

**Goal**: Merge or close the stale open PR and ensure all branches are reconciled.

### 1a. Triage PR #2 (Sprint 6-7 Features)
- **File**: PR #2 targets `claude/plan-forge-environment-htAoK`, not `main` — this is likely stale
- Review the 4 epics (E17 mypy, E11 replay, E16 Docker, E13 dashboard) for overlap with already-merged PRs (#11 Docker, #16 implementation plan, #20 MangoMAS coverage)
- Cherry-pick any unmerged work (replay CLI, multi-agent dashboard SVG components) into a fresh branch off `main`
- Close PR #2 after extracting value; open targeted follow-up PRs if needed

### 1b. Sync Working Branch
- Ensure `claude/create-implementation-plan-Unk9Q` is rebased on latest `main`
- Confirm green baseline: `cargo test --workspace`, `cargo clippy --workspace -- -D warnings`, `cargo fmt --check`

---

## Phase 2: Test Coverage to 85%+ Per Crate

**Goal**: Every crate meets the 85% coverage floor with proptest invariant verification.

### 2a. `forge-integration-layer` (23 tests — lowest coverage)
**Files**: `crates/forge-integration/src/{orchestrator.rs, config.rs, metrics.rs, lib.rs}`
- Add orchestrator pipeline tests: memory -> social -> cognitive -> action per tick
- Add config serde roundtrip using `assert_config_serde_roundtrip!` from `forge-types/src/test_util.rs`
- Add metrics counter/timer accuracy tests
- Add integration test: `WorldState` -> orchestrator -> valid action output
- Proptest: random agent count + random tick sequences -> orchestrator never panics
- **Target**: 23 -> 50+ tests

### 2b. `forge-eval` (29 tests)
**Files**: `crates/forge-eval/src/{harness.rs, scorecard.rs, config.rs}`
- Add harness tests: parallel episode execution, timeout handling, seed determinism
- Add scorecard aggregation edge cases: empty results, single episode, all failures
- Add config defaults validation, serde roundtrip
- Proptest: random eval config -> valid scorecard structure
- **Target**: 29 -> 50+ tests

### 2c. `forge-replay` (30 tests)
**Files**: `crates/forge-replay/src/{compact.rs, trajectory.rs, export.rs, config.rs}`
- Add compact replay determinism: encode -> decode -> re-simulate matches
- Add trajectory storage boundary tests: max capacity, empty episodes
- Add CSV/JSON Lines export format validation
- Proptest: random action sequences -> valid replay roundtrip
- **Target**: 30 -> 50+ tests

### 2d. `forge-scenario` (41 tests)
**Files**: `crates/forge-scenario/src/{config.rs, registry.rs, compose.rs}`
- Add registry search edge cases: no matches, all tiers, empty tags
- Add compose merge conflict resolution tests
- Add TOML loading with malformed/missing fields
- Proptest: random scenario configs -> compose always produces valid config
- **Target**: 41 -> 55+ tests

### 2e. Coverage Gate in CI
**File**: `.github/workflows/ci.yml`
- Add `cargo-tarpaulin` job with `--workspace` and `--fail-under 85` threshold
- Cache tarpaulin binary for faster runs
- Output per-crate coverage report as CI artifact

### Reusable Utilities
- `forge-types/src/test_util.rs`: `assert_config_serde_roundtrip!`, `assert_config_defaults_valid!`
- `ForgeConfig::default()` for baseline config in all test constructors
- `WorldState::new()` from `forge-core` for standard test state
- `rand_pcg::Pcg64Mcg` with fixed seeds for deterministic test RNG

---

## Phase 3: CI/CD Hardening (v0.2 Gate)

**Goal**: Production-grade CI that catches regressions before merge.

### 3a. Coverage Reporting
**File**: `.github/workflows/ci.yml`
- Add tarpaulin with per-crate breakdown, fail-under=85
- Upload coverage to Codecov or as GitHub Actions artifact
- Add badge to README

### 3b. Python CI Split
**File**: `.github/workflows/ci.yml`
- Split Python job into fast (pure-Python import/config/utility tests) and slow (maturin develop + native integration tests)
- Add `--cov-fail-under=85` for Python coverage gate (already present, verify it runs)

### 3c. Benchmark Regression Detection
**File**: `.github/workflows/ci.yml` + `crates/forge-bench/`
- Store Criterion JSON baseline in CI artifacts
- Compare PR benchmarks against baseline, flag >5% regressions as warnings
- Use `critcmp` or custom script for comparison

### 3d. Docker Image Publishing
**Files**: `docker/`, `.github/workflows/ci.yml`
- Add multi-arch build (`linux/amd64` + `linux/arm64`) via `docker buildx`
- Push to GHCR on release tags
- Health check validation in CI

---

## Phase 4: WASM Live Demo + Replay (v0.3)

**Goal**: Public-facing demo and replay infrastructure.

### 4a. WASM Demo on GitHub Pages
**Files**: `crates/forge-wasm/`, `.github/workflows/gh-pages.yml` (create)
- Compile `forge-wasm` via `wasm-pack build --target web`
- Create minimal HTML/JS page that imports WASM module and runs simulation in-browser
- CI workflow: build WASM -> deploy to `gh-pages` branch on push to main
- All configuration via JSON passed to WASM (no hardcoded values in JS)

### 4b. Replay/Record Mode
**Files**: `python/forge/replay.py`, `python/forge_env/wrappers.py`
- `RecordEpisodeWrapper`: capture seed + actions to `.forge` JSON v1 format
- `load_replay` / `play_replay` functions with configurable playback speed
- `export_gif` from world canvas frames
- CLI entry point: `forge-replay record|play|export`
- Config-driven: replay format version, compression, max episode length all via config

### 4c. Playwright E2E Browser Tests
**File**: `demo_ui/tests/test_e2e_browser.py` (create)
- Section badges: IDLE -> RUNNING -> PASS transitions
- Terminal streaming without errors
- Progress bar advancement
- World canvas renders visible pixel data
- All URLs via config (no hardcoded `localhost:8765`)

---

## Phase 5: REST API + Dashboards (v1.0)

**Goal**: FORGE as a production service with rich visualization.

### 5a. REST API (forge-server)
**Files**: `crates/forge-server/src/{api.rs, config.rs}`
- `POST /api/env/reset` — config + seed -> new environment
- `POST /api/env/step` — action -> observation + reward + done
- `GET /api/env/render` — ASCII + grid JSON
- OpenAPI spec generation via `utoipa` crate
- Rate limiting via config (`ApiConfig { rate_limit_per_sec: u32, auth_enabled: bool }`)
- Auth token support (optional, config-gated)
- All ports/hosts from config structs, never hardcoded

### 5b. Multi-Agent Dashboard
**Files**: `demo_ui/frontend/`, `crates/forge-server/src/ws_handler.rs`
- Per-agent color coding on world canvas (Oklab WCAG-AA palette from config)
- Communication token visualization (badge overlay)
- Real-time reward curves per agent via WebSocket streaming
- Agent selection/filtering panel

### 5c. Task Curriculum Visualizer
**Files**: `demo_ui/frontend/`, new component
- DSL tree rendering (recursive task visualization using existing `forge-task` evaluator)
- Tier distribution histogram
- Live success/failure rates streamed from training loop

### 5d. Training Integration Hardening
**Files**: `python/forge/training/`, `scripts/`
- SB3 + CleanRL example scripts reading config from TOML files
- W&B / MLflow logging hooks from env wrappers (config-driven logger selection)
- Pre-trained model checkpoint serving via demo UI

---

## Phase 6: Technical Debt Reduction

| Item | Priority | File(s) | Action |
|------|----------|---------|--------|
| conftest.py sys.path hack | High | `tests/python/conftest.py` | Convert to `pyproject.toml` package install |
| demo_ui as installable package | High | `demo_ui/pyproject.toml` | `pip install -e demo_ui/` |
| Cross-platform launcher | Medium | `demo_ui/run_demo.sh` | Test on Linux/macOS, unify with .ps1 |
| MangoMAS E2E smoke test | High | `tests/python/` | Wire mangomas components through minimal FORGE episode loop |
| Predicate terrain_id magic numbers | Low | `crates/forge-task/src/predicate.rs` | Replace with `TerrainType::from_id()` |
| ObjectState string matching | Low | `crates/forge-types/` | Replace manual match with `FromStr`/`TryFrom` impl |

---

## Key Principles (Enforced Throughout)

- **No hard-coded values**: All constants via config structs with `Default` impls and `#[serde(default)]`
- **Backwards compatible**: New fields use `#[serde(default)]`, new enum variants non-breaking
- **Modular/Reusable**: Shared test macros, shared `test_util.rs`, workspace dependencies
- **Dynamic**: Config-driven behavior at runtime, no compile-time feature gates for runtime behavior
- **Deterministic**: `fixed` crate for physics, `rand_pcg` for RNG, seed flows through all layers
- **85%+ coverage**: Every phase includes specific test plans; proptest for invariant verification
- **Zero allocation on hot path**: `PhysicsScratch` pattern for pre-allocated buffers
- **Structured logging**: `tracing` with `#[instrument]` on public functions

---

## Verification Plan (After Each Phase)

```bash
# Rust
cargo build --workspace                          # zero errors
cargo test --workspace                           # all tests pass
cargo clippy --workspace -- -D warnings          # zero warnings
cargo fmt --check                                # formatted
cargo tarpaulin --workspace --fail-under 85      # >=85% per crate

# Python (after maturin develop)
pytest tests/python/ -v --cov=forge --cov=forge_env --cov-fail-under=85

# Demo UI
pytest demo_ui/tests/ -v --tb=short

# Benchmarks
cargo bench -p forge-bench                       # no regressions >5%
```

---

## Critical Files to Modify

**Phase 2 (Coverage)**:
- `crates/forge-integration/src/lib.rs` — add `#[cfg(test)] mod tests` with 30+ tests
- `crates/forge-eval/src/{harness.rs, scorecard.rs}` — add edge case + proptest coverage
- `crates/forge-replay/src/{compact.rs, trajectory.rs}` — add roundtrip + proptest
- `crates/forge-scenario/src/{registry.rs, compose.rs}` — add search + merge tests

**Phase 3 (CI)**:
- `.github/workflows/ci.yml` — tarpaulin, benchmark regression, Docker publish

**Phase 4 (WASM/Replay)**:
- `crates/forge-wasm/` — wasm-pack build target
- `.github/workflows/gh-pages.yml` (new) — WASM deploy
- `python/forge/replay.py` — replay CLI and viewer
- `demo_ui/tests/test_e2e_browser.py` (new) — Playwright tests

**Phase 5 (REST API)**:
- `crates/forge-server/src/api.rs` — REST endpoints
- `crates/forge-server/Cargo.toml` — add `utoipa` for OpenAPI
- `demo_ui/frontend/` — dashboard components

**Existing Utilities to Reuse**:
- `forge-types/src/test_util.rs`: `assert_config_serde_roundtrip!`, `assert_config_defaults_valid!`
- `forge-core::WorldState::new()` — standard test state
- `ForgeConfig::default()` — baseline config for all tests
- `forge-types/src/constants.rs` — shared named constants
- `forge-types/src/agent_interface.rs` — `AgentInterface` trait for agent-agnostic eval
