# FORGE Implementation Plan — Next Steps

## Context

FORGE is a 14-crate Rust workspace (simulation sandbox for training AI agents) at **v0.1.0** with an unreleased demo UI. The 8 core crates are mature (91.13% overall coverage, 889 Rust + 450+ Python tests). Six extended crates (forge-procgen, forge-server, forge-memory, forge-social, forge-cognitive, forge-integration) were added post-v0.1 (~7,308 lines) with framework/skeleton implementations that need test coverage and deeper integration. Two open PRs (#11 Docker deployment, #2 Sprint6-7 features) need to be resolved. The roadmap targets v0.2 (CI/Docker/E2E), v0.3 (WASM live demo, benchmarks, replay), and v1.0 (REST API, dashboards, training integration). Six files are below the 85% coverage target.

---

## Phase 0: Establish Green Baseline

Before any new development, fix the 2 pre-existing test failures identified in PR #2 to ensure a clean starting point. The verification plan for every subsequent phase requires all tests to pass -- this is only meaningful if the baseline is already green.

- Investigate and fix the 2 failing tests on the default branch
- Confirm `cargo test --workspace` passes with zero failures
- Confirm `pytest tests/python/ -v` passes with zero failures (skips are acceptable)
- Document root causes and fixes in commit messages

---

## Phase 1: Close Coverage Gaps (Target: All Files >= 85%)

### 1a. `forge-task/predicate.rs` (78.9% -> 90%+)
**File:** `crates/forge-task/src/predicate.rs`
- Missing coverage: `AgentOnTerrain` with agent outside grid bounds, `ObjectAt` with no objects context, wildcard `_` branch in `evaluate_predicate`
- Add tests:
  - Agent position outside grid dimensions (grid.get returns None)
  - `ObjectAt` with objects=None context
  - `ObjectInState` with all lowercase variants
  - Proptest: random predicate + random EvalContext -> never panics

### 1b. `forge-agent/mcts/search.rs` (82.7% -> 90%+)
**File:** `crates/forge-agent/src/mcts/search.rs`
- Missing coverage: deep tree traversal (max_depth hit), terminal state detection, agent_idx >= num_agents branch
- Add tests:
  - Search with `max_depth: 1` to hit depth limit
  - State that `is_terminal()` immediately -> value = 0.0
  - `agent_idx` > num_agents (out of bounds guard at line 76)
  - Multi-agent search (2+ agents, searching for agent 1)

### 1c. `forge-types/validation.rs` (81.5% -> 90%+)
**File:** `crates/forge-types/src/validation.rs`
- Missing coverage: drone validation branches (starting_battery > max_battery, aerial_drain_rate < 0, ascend_cost < 0, recharge_rate < 0)
- Add tests:
  - `starting_battery > max_battery` (line 149-156)
  - Negative `aerial_drain_rate` (line 157-164)
  - Negative `ascend_cost` (line 165-172)
  - Negative `recharge_rate` (line 173-180)
  - Zero `starting_battery` (line 141-148)
  - Proptest: drone config with enabled=true, random field combinations

### 1d. `forge-worldgen/terrain.rs` (84.6% -> 90%+)
**File:** `crates/forge-worldgen/src/terrain.rs`
- Add edge-case tests for terrain generation boundaries
- Proptest: various seed + dimension combos -> valid terrain grid

### 1e. `forge-agent/baselines.rs` (85.7% -> 90%+)
**File:** `crates/forge-agent/src/baselines.rs`
- Test uncovered baseline agent paths (greedy, rule-based policy selections)
- Add tests for edge cases: empty inventory, no valid targets

### 1f. `forge-task/generator.rs` (85.2% -> 90%+)
**File:** `crates/forge-task/src/generator.rs`
- Test task generation edge cases: min/max difficulty bounds, empty agent list
- Proptest: random config -> valid task tree

---

## Phase 2: Extended Crate Test Coverage (Target: 85%+ Each)

### 2a. `forge-memory` (1,346 lines, ~7 source files)
**Files:** `crates/forge-memory/src/{semantic,episodic,preference,store,config,error}.rs`
- Tests needed:
  - Semantic memory: add/query/decay cycle, duplicate facts, capacity limits
  - Episodic memory: record/replay/decay, temporal ordering, bounded storage
  - Preference memory: update/query/reinforce, tendency convergence
  - InMemoryStore: cross-memory queries, persistence roundtrip
  - Config: serde roundtrip, defaults validation (reuse `assert_config_serde_roundtrip!` macro)
  - Error: Display/From impls
  - Proptest: random operations never panic, storage stays within bounds

### 2b. `forge-social` (927 lines, ~6 source files)
**Files:** `crates/forge-social/src/{trust,reputation,alliance,social_reward,config}.rs`
- Tests needed:
  - Trust matrix: symmetric updates, bounds [0.0, 1.0], unknown agents
  - Reputation: increment/decrement, public vs private, decay
  - Alliance: formation/dissolution, member limits, conflict detection
  - Social reward: cooperation signal calculation, reward scaling
  - Config: serde roundtrip, defaults
  - Proptest: random agent interactions -> trust always in valid range

### 2c. `forge-cognitive` (1,073 lines, ~6 source files)
**Files:** `crates/forge-cognitive/src/{agent,provider,prompt,reasoning,config}.rs`
- Tests needed:
  - Provider trait: mock provider returning deterministic responses
  - Prompt builder: observation + memory + social context -> valid prompt string
  - Reasoning traces: chain-of-thought recording, trace truncation
  - Agent: full decide cycle with mock provider
  - Config: serde roundtrip, defaults

### 2d. `forge-integration` (434 lines, ~4 source files)
**Files:** `crates/forge-integration/src/{orchestrator,config,metrics}.rs`
- Tests needed:
  - Orchestrator: per-agent-per-tick coordination (memory -> social -> cognitive)
  - Config: unified config merging, serde roundtrip
  - Metrics: counter increments, timer accuracy
  - Integration test: full pipeline from WorldState -> orchestrator -> action

### 2e. `forge-procgen` (1,599 lines, ~7 source files)
**Files:** `crates/forge-procgen/src/{map_generator,objective_generator,curriculum,team_composer,grammar,seed}.rs`
- Tests needed:
  - Map generator: cluster placement, terrain distribution, seed determinism
  - Objective generator: grammar expansion, valid task tree output
  - Curriculum: difficulty progression, tier transitions
  - Team composer: balanced team generation, constraint satisfaction
  - Seed: deterministic derivation, uniqueness
  - Proptest: random seed -> valid map + objectives

### 2f. `forge-server` (1,919 lines, ~7 source files)
**Files:** `crates/forge-server/src/{api,ws_handler,state,metrics,config,main}.rs`
- Tests needed:
  - API: endpoint routing, request/response serialization, error handling
  - WebSocket: connection lifecycle, message broadcasting, disconnection
  - State: concurrent access, simulation state updates
  - Metrics: counter/gauge correctness
  - Config: serde roundtrip, defaults
  - Integration test: start server -> HTTP request -> valid response

---

## Phase 3: Resolve Open PRs

### 3a. PR #11 -- Docker Multi-Service Deployment
**Branch:** `feat/docker-deployment`
- Review for conflicts with main branch
- Validate Docker Compose 3-service stack with unified port configuration
- Unify all service ports via env-var config (`FORGE_PORT`, `FORGE_DASHBOARD_PORT`, `FORGE_SIM_PORT`) -- no hardcoded port numbers across docker-compose.yml, E2E tests, or launcher scripts
- Ensure health checks pass, TypeScript/lint fixes are clean
- Merge or rebase onto current main

### 3b. PR #2 -- Sprint 6-7 Features
**Branch:** `feat/beta-release-planning`
- Contains: mypy strict (E17), replay/record (E11), Docker (E16), multi-agent dashboard (E13)
- Check overlap with PR #11 (Docker features may conflict)
- Validate 128 passing tests, address 2 pre-existing failures
- Evaluate whether to cherry-pick individual epics or merge as-is

---

## Phase 4: v0.2 Release -- CI/Docker/E2E

### 4a. GitHub Actions CI
**File:** `.github/workflows/ci.yml` (create/extend)
- Cargo build, test, clippy, fmt check
- Python: maturin develop -> pytest
- Demo UI: pip install -> pytest demo_ui/tests/
- Coverage: cargo-tarpaulin with 85% threshold gate
- Cache: cargo registry, pip, node_modules

### 4b. Playwright E2E Browser Tests
**File:** `demo_ui/tests/test_e2e_browser.py` (create)
- Section badges: IDLE -> RUNNING -> PASS transitions
- Terminal streaming without errors
- Progress bar advancement
- World canvas renders visible pixel data
- Uses config-driven URLs (no hardcoded localhost)

### 4c. Docker Finalization
**Files:** `docker/`, `docker-compose.yml`
- Multi-stage builds: rust:1.82-bookworm -> python:3.11-slim
- Health check endpoints
- Non-root user, localhost-only binding
- Docker Hub multi-arch image publishing (CI step)

---

## Phase 5: v0.3 -- WASM Live Demo + Benchmarks + Replay

### 5a. GitHub Pages WASM Demo
- Compile forge-wasm -> wasm-pack
- Static HTML/JS demo page using WASM directly (no server)
- CI: build WASM -> deploy to gh-pages branch

### 5b. Benchmark Regression Tracking
**File:** `crates/forge-bench/` + CI integration
- Store Criterion JSON artifacts in CI
- Compare against baseline, flag >5% regressions
- Display throughput trends in demo UI stats panel

### 5c. Replay/Record Mode
**Files:** `python/forge/replay.py`, wrapper classes
- RecordEpisodeWrapper: capture actions + seed to JSON
- Replay viewer: configurable playback speed
- GIF export from world canvas

---

## Phase 6: v1.0 -- REST API + Dashboards + Training

### 6a. REST API (forge-server)
- `POST /api/env/reset` -- config + seed -> new environment
- `POST /api/env/step` -- action -> observation + reward
- `GET /api/env/render` -- ASCII + grid JSON
- OpenAPI spec generation, rate limiting, auth tokens

### 6b. Multi-Agent Dashboard
- Per-agent color coding on world canvas
- Communication token visualization
- Reward curves per agent (real-time streaming via WebSocket)

### 6c. Task Curriculum Visualizer
- DSL tree rendering (recursive task visualization)
- Tier distribution histogram
- Live success/failure rates

### 6d. Training Integration Hardening
- SB3 + CleanRL example scripts with config files
- W&B / MLflow logging hooks from env wrappers
- Pre-trained model checkpoint serving via demo UI

---

## Phase 7: Technical Debt

| Item | Priority | Action |
|------|----------|--------|
| Mypy strict for demo_ui/tests | Medium | Add type annotations, enable strict mode |
| conftest.py sys.path hack | Medium | Convert to pyproject.toml package install |
| demo_ui as installable package | Medium | Add pyproject.toml, `pip install -e demo_ui/` |
| Cross-platform launcher | Medium | `run_demo.sh` for Linux/macOS alongside .ps1 |
| Predicate terrain_id magic numbers | Low | Replace with TerrainType::from_id() method |
| ObjectState string matching | Low | Replace manual match with FromStr/TryFrom impl |

---

## Key Principles (Enforced Throughout)

- **No hard-coded values**: All constants via config structs with `Default` impls
- **Backwards compatible**: New fields use `#[serde(default)]`, new enum variants non-breaking
- **Modular/Reusable**: Shared test macros (`assert_config_serde_roundtrip!`), shared test utilities in `forge-types/src/test_util.rs`
- **Dynamic**: Config-driven behavior, no compile-time feature gates for runtime behavior
- **Deterministic**: `fixed` crate for physics, `rand_pcg` for RNG, seed flows through all layers
- **85%+ coverage**: Every phase includes specific test plans; proptest for invariant verification
- **Zero allocation on hot path**: `PhysicsScratch` pattern for pre-allocated buffers
- **Structured logging**: `tracing` with `#[instrument]` on key public functions that perform significant work (skip trivial getters to avoid log noise and overhead)

---

## Verification Plan

After each phase:
1. `cargo build --workspace` -- zero errors
2. `cargo test --workspace` -- all tests pass
3. `cargo clippy --workspace -- -D warnings` -- zero warnings
4. `cargo fmt --check` -- formatted
5. `cargo tarpaulin --workspace` -- >=85% per crate
6. `pytest tests/python/ -v` -- Python tests pass (after `maturin develop`)
7. Extended crates: run crate-specific tests (`cargo test -p forge-memory`, etc.)

---

## Files to Create/Modify (Summary)

**New test files:**
- `crates/forge-memory/src/lib.rs` (add `#[cfg(test)] mod tests`)
- `crates/forge-social/src/lib.rs` (add `#[cfg(test)] mod tests`)
- `crates/forge-cognitive/src/lib.rs` (add `#[cfg(test)] mod tests`)
- `crates/forge-integration/src/lib.rs` (add `#[cfg(test)] mod tests`)
- `crates/forge-procgen/src/lib.rs` (add `#[cfg(test)] mod tests`)
- `crates/forge-server/src/lib.rs` (add `#[cfg(test)] mod tests`)
- `demo_ui/tests/test_e2e_browser.py`
- `.github/workflows/ci.yml`

**Modified files (coverage gaps):**
- `crates/forge-task/src/predicate.rs` -- add ~8 tests
- `crates/forge-agent/src/mcts/search.rs` -- add ~5 tests
- `crates/forge-types/src/validation.rs` -- add ~6 tests
- `crates/forge-worldgen/src/terrain.rs` -- add ~4 tests
- `crates/forge-agent/src/baselines.rs` -- add ~4 tests
- `crates/forge-task/src/generator.rs` -- add ~4 tests

**Existing utilities to reuse:**
- `assert_config_serde_roundtrip!` macro (shared config test patterns)
- `assert_config_defaults_valid!` macro
- `forge_types::test_util` -- shared test helpers (make_agent, make_config, etc.)
- `forge_core::WorldState::new()` -- standard test state construction
- `ForgeConfig::default()` -- baseline for all config-driven tests
