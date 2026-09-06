# FORGE Codebase Optimization & Enterprise Hardening Master Plan

**Date:** 2026-09-06  
**Status:** Approved Engineering Blueprint  
**Authors:** Senior Engineering & Technical Product Management Working Group  
**Cross-References:** [`docs/next_steps.md`](../next_steps.md), [`docs/architecture.md`](../architecture.md), [`docs/hardcoded-values-audit.md`](../hardcoded-values-audit.md), [`docs/CHARTER.md`](../CHARTER.md)

---

## 1. Executive Summary & Team Charter

This master plan synthesizes findings from a comprehensive engineering audit across the complete FORGE platform:
- 26 Rust crates in the workspace (`crates/*`)
- Python package (`python/forge`, `python/forge_env`)
- Node.js / TypeScript bot service (`mc-bot/`)
- React 18 / Vite operator dashboard (`dashboard/`)
- FastAPI interactive demo backend (`demo_ui/`)
- WebAssembly in-browser static simulation demo (`web/`, `crates/forge-wasm`)

Our mandate as a joint engineering and product team is to elevate FORGE from a high-velocity simulation and RL research codebase into a hardened, enterprise-grade platform. Every proposed architecture change strictly respects FORGE's foundational invariants:
1. **Bit-identical determinism** (Invariant 6): identical seeds produce bit-for-bit identical trajectory outputs across ticks.
2. **Zero-allocation hot-path execution** on `WorldState::step_into` (Invariant 7): 0 bytes allocated during simulation execution.
3. **Configuration-driven behavior with zero hardcoded runtime magic numbers** (Invariant 5).
4. **Strict schema-versioned cross-language contracts** across Rust, Python, and TypeScript.

---

## 2. Platform Audit & Current Health Matrix

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              FORGE Monorepo Health Matrix                              │
├───────────────────┬──────────────┬──────────────────┬─────────────────┬────────────────┤
│ Subsystem         │ Build / CI   │ Code Coverage    │ "God File" Risk │ Security / Pin │
├───────────────────┼──────────────┼──────────────────┼─────────────────┼────────────────┤
│ Rust Workspace    │ ✅ Green     │ 85% tarpaulin    │ In-file test    │ cargo-deny     │
│ (26 Crates)       │              │ enforced         │ bloat (50-65%)  │ clean; blocking│
├───────────────────┼──────────────┼──────────────────┼─────────────────┼────────────────┤
│ Python Package    │ ✅ Green     │ 85% pytest-cov   │ High            │ pip-audit      │
│ (forge / env)     │              │ enforced         │ (collector.py)  │ clean; blocking│
├───────────────────┼──────────────┼──────────────────┼─────────────────┼────────────────┤
│ mc-bot            │ ✅ Green     │ 85% lines /      │ Moderate        │ npm-audit      │
│ (TypeScript)      │ (237 tests)  │ 75% branches c8  │ (index.ts)      │ advisory       │
├───────────────────┼──────────────┼──────────────────┼─────────────────┼────────────────┤
│ dashboard         │ ✅ Green     │ 98.8% lines /    │ Clean           │ npm-audit      │
│ (React 18 / Vite) │ (Vitest)     │ 90.3% branches   │ (<275 loc/file) │ advisory       │
├───────────────────┼──────────────┼──────────────────┼─────────────────┼────────────────┤
│ demo_ui           │ ✅ Green     │ Report-only      │ Clean           │ Needs 70%      │
│ (FastAPI)         │ (Playwright) │ (no floor)       │                 │ coverage floor │
├───────────────────┼──────────────┼──────────────────┼─────────────────┼────────────────┤
│ web (WASM)        │ ✅ Green     │ E2E Playwright   │ Clean           │ Needs wasm-pack│
│                   │ (wasm-check) │ integration      │                 │ for local run  │
└───────────────────┴──────────────┴──────────────────┴─────────────────┴────────────────┘
```

---

## 3. Phase 1: CI/CD Green State & Immediate Hardening (Landed)

The following critical remediations landed on `claude/plan-forge-environment-htAoK`:

### 3.1 Docker ABI Alignment (`docker/Dockerfile`)
- **Problem**: Multi-stage `docker/Dockerfile` built wheels using `rust:1.94.1-bookworm` (Python 3.11), producing wheels with `cp311` ABI tags. The runtime stage used `python:3.14-slim-bookworm`, causing `pip install /tmp/*.whl` to fail with `not a supported wheel on this platform`.
- **Resolution**: Aligned runtime to `python:3.11-slim-bookworm`.
- **Hygiene**: Removed unused `FORGE_CONFIG_PATH` env var and `COPY forge.toml /app/` (the `forge-server` binary uses `ForgeConfig::default()` in code and does not read `forge.toml`).

### 3.2 Configuration Drift Prevention (`crates/forge-types/src/config.rs`)
- **Problem**: The root `forge.toml` contained Python-targeted sections (`[hardware]`, `[simulation]`, `[training]`), whereas the Rust `ForgeConfig` schema defines `[world]`, `[physics]`, `[agents]`, `[task]`, etc. Without strict validation, `serde` silently dropped all mismatched sections.
- **Resolution**: Decorated `ForgeConfig` and all 11 nested child configuration structs with `#[serde(deny_unknown_fields)]`. Unknown or misspelled fields now cause an explicit deserialization error. Added documentation clarifying the schema divide.

### 3.3 Dead Code Elimination
- Cleaned unused export `listBuiltins()` in `mc-bot/src/reward/index.ts`.
- Removed stale `#[allow(dead_code)]` on `MlflowHttpClient.http` in `crates/forge-eval/src/exporters/mlflow_http.rs` (field actively used across 7 REST endpoints).
- Clarified unused parameter `target_entropy` in `examples/train_sac_cleanrl.py` by renaming to `_target_entropy` with explicit explanatory comments.

### 3.4 Security Scanning Promotion
- Promoted `pip-audit` from advisory (`|| true`) to a blocking CI gate in `.github/workflows/security.yml`. Verified clean with zero known CVE vulnerabilities.

---

## 4. Phase 2: God File Reduction & Structural Decoupling

### 4.1 The In-File Test Bloat Pattern in Rust

A core architectural finding of this peer review is that large files in the Rust workspace are primarily **in-file test suites (`#[cfg(test)] mod tests`) rather than oversized production modules**:

| File Path | Total Lines | Production Lines | Test Lines | % Test Code |
|---|---|---|---|---|
| `crates/forge-core/src/world.rs` | 2,122 | 776 | 1,346 | **63.4%** |
| `crates/forge-core/src/physics.rs` | 1,663 | 577 | 1,086 | **65.3%** |
| `crates/forge-types/src/config.rs` | 1,605 | 796 | 809 | **50.4%** |
| `crates/forge-types/src/action.rs` | 1,372 | 666 | 706 | **51.5%** |
| `crates/forge-task/src/predicate.rs` | 1,155 | 455 | 700 | **60.6%** |

#### Step 2.1: Test Extraction Strategy
Instead of risking disruptions to private field accesses or allocator ergonomics, unit and proptest suites should be moved into dedicated test files:
- `crates/forge-core/src/world/tests.rs`
- `crates/forge-core/src/physics/tests.rs`
- `crates/forge-types/src/config/tests.rs`
- `crates/forge-types/src/action/tests.rs`

**Result**: Immediately drops file lengths below the 800-line threshold without modifying production logic or zero-alloc contracts.

### 4.2 Modularizing `crates/forge-core/src/world.rs` (Production: 776 lines)

Decompose `world.rs` into a clean submodule structure:
```
crates/forge-core/src/world/
├── mod.rs           # Re-exports WorldState and public API surface
├── state.rs         # WorldState struct definition and scratch buffer layout
├── step.rs          # step(), step_into(), make_step_result(), fill_step_result()
├── reset.rs         # reset(), reset_into(), world generation orchestration
├── observation.rs   # observe(), fill_observation(), observation flat vector layouts
├── serialize.rs     # bincode state serialization, golden hash calculation
├── debug.rs         # ASCII grid debug renderers
└── tests.rs         # Extracted 1,346 lines of unit and proptests
```
**Zero-Allocation Invariant**: All scratch buffers (`physics_scratch`, `agri_scratch`, `step_actions`, `validated_actions`, `near_station`, `crafting_object_map`, `comm_messages`, `push_scratch`) remain fields on `WorldState`. `step_into` reuses them exactly as before, ensuring 100% compliance with `benchmarks/runner/check_zero_alloc.py`.

### 4.3 Decomposing `python/forge/mangomas/collector.py` (1,476 lines)

Unlike the Rust files, `collector.py` is 100% production code combining scenario resolution, action decoding, multi-worker rollout, and trajectory writing.
Decompose into `python/forge/mangomas/collector/`:
- `types.py`: `ResolvedForgeScenario`, `ScenarioRolloutSummary`, `CollectionJobSpec`
- `scenario.py`: Scenario file resolution, tier mapping, metadata extraction
- `action_decoder.py`: Discrete action mapping across base, drone, agriculture, and hex actions
- `sync_rollout.py`: Synchronous rollout loop (`collect_scenario_episodes_sync`)
- `async_rollout.py`: Asynchronous concurrent rollout engine (`collect_scenario_episodes_async`)
- `writer.py`: Trajectory record formatting and filesystem persistence
- `__init__.py`: Clean facade preserving backward-compatible exports

### 4.4 Decomposing `mc-bot/src/index.ts` (616 lines)

Decompose `mc-bot/src/index.ts` into:
- `auth.ts`: `extractAuthToken`, `timingSafeEquals`, `authorizeRequest`, `createVerifyClient`
- `connection.ts`: `createConnectionHandler`, message framing, step/reset lifecycle
- `server.ts`: `startProtocolServer`, WebSocket configuration, listener management
- `index.ts`: Process guards, CLI argument bootstrap, entry point (<80 lines)

---

## 5. Phase 3: Coverage Gates & Security Hardening

### 5.1 Enforcing Coverage on `demo_ui`
- `demo_ui/.coveragerc` and `demo_ui/pytest.ini` currently omit `--cov-fail-under` (report-only).
- **Target**: Run baseline measurement on `demo_ui/backend`, set `--cov-fail-under=70` in `demo_ui/pytest.ini`, and document the ratcheting plan to 85% in `docs/next_steps.md`.

### 5.2 Ratcheting Advisory Security Scanners
1. **`cargo-machete`**:
   - Currently advisory (`|| true`) due to false positives on derive-only dependencies (`serde`, `thiserror`, `fixed`).
   - Fix: Configure `[package.metadata.cargo-machete] ignored = [...]` in affected crates, then make `cargo machete` a blocking check in `ci.yml`.
2. **`npm-audit`**:
   - Run audit across `mc-bot` and `dashboard`. Resolve or pin advisory exceptions, then promote to blocking.
3. **E2E Playwright Promotion**:
   - Once `dashboard-e2e` and `wasm-e2e` maintain a 3-release zero-flake record, promote from advisory to required branch protection checks.

---

## 6. Phase 4: Configuration Hardening & Isolation

### 6.1 Audit of Remaining Magic Numbers

| Parameter | Location | Issue | Hardening Remediation |
|---|---|---|---|
| Metrics Fallback URL | `python/forge/training/muzero_mc/capture_baseline.py:53` | Hardcoded `http://127.0.0.1:9090/metrics` | Support `FORGE_MC_METRICS_URL` environment variable override |
| MLflow Plotly CDN | `crates/forge-eval/src/exporters/mlflow_payload.rs:699` | Hardcoded `https://cdn.plot.ly/plotly-2.35.2.min.js` | Support `FORGE_PLOTLY_JS_URL` for air-gapped / offline deployments |
| mc-bot Configuration | `mc-bot/src/config.ts` | TypeScript defaults manually mirror `env.toml` | Build-time validation via `scripts/check_pinned_config_consistency.py` |

---

## 7. Phase 5: Enterprise Organization & Architecture Governance

### 7.1 Formal Workspace Layering (Tiers L0 to L5)

To guarantee unidirectional dependency flow and prevent circular coupling, the 26 workspace crates are stratified into 6 architectural tiers:

```
┌────────────────────────────────────────────────────────────────────────┐
│ Tier 5: Distributed Cloud                                              │
│   forge-cloud                                                          │
├────────────────────────────────────────────────────────────────────────┤
│ Tier 4: Applications, Runners & Tooling                                │
│   forge-mc-runner, forge-mangomas, forge-eval, forge-data,             │
│   forge-edge, forge-bench                                              │
├────────────────────────────────────────────────────────────────────────┤
│ Tier 3: Agents, Adapters & Protocol Servers                            │
│   forge-agent, forge-replay, forge-server, forge-python, forge-wasm,   │
│   forge-env-forge, forge-integration-tests                             │
├────────────────────────────────────────────────────────────────────────┤
│ Tier 2: Simulation Engine & Cognitive Models                           │
│   forge-core, forge-cognitive                                          │
├────────────────────────────────────────────────────────────────────────┤
│ Tier 1: Domain Primitives & Environments                               │
│   forge-civ, forge-worldgen, forge-task, forge-memory, forge-social,   │
│   forge-env-mc                                                         │
├────────────────────────────────────────────────────────────────────────┤
│ Tier 0: Foundation Layer (Zero Workspace Dependencies)                 │
│   forge-types, forge-env, forge-observability                          │
└────────────────────────────────────────────────────────────────────────┘
```

#### Layering Rules:
- Crates in Tier $N$ may only depend on crates in Tier $< N$.
- Tier 0 crates must never depend on any other workspace crate.
- `forge-core` must never depend on `forge-agent`, `forge-eval`, or `forge-server`.
- Enforce these rules via `deny.toml` `[bans]` configurations in CI.

### 7.2 Research Stack Lifecycle & `forge-env-forge`
- `forge-env-forge` currently has 0 downstream Cargo dependencies.
- **Action**: Plumb `forge-env-forge` into `forge-bench` for hot-path allocation audits alongside `forge-env-mc`, proving that generic `Env` implementers adhere to zero-allocation guarantees.
- Annotate experimental crates (`forge-memory`, `forge-social`, `forge-cognitive`, `forge-cloud`) with maturity badges (`[Experimental]`, `[Research]`, `[Production]`).

### 7.3 Per-Crate Documentation & Config Catalog
- **Per-Crate READMEs**: Introduce a standard `README.md` in each of the 26 crates detailing tier, public API, feature flags, and examples.
- **Config Catalog (`docs/config-catalog.md`)**: Catalog all 25+ TOML configuration files in `configs/`, mapping them to their consuming crate, schema type, and environment overrides.

### 7.4 Bincode 2.x Migration Roadmap
- `bincode 1.3.3` is unmaintained (tracked as an exception in `deny.toml`).
- `WorldState` relies on `bincode` for checkpoints and determinism hashing.
- Roadmap:
  1. Build a regression test suite validating binary compatibility on saved snapshots.
  2. Implement bincode 2.x under a `bincode-v2` feature flag with explicit wire-format testing.
  3. Validate zero-alloc deserialization performance before cutting over the default.

---

## 8. Verification & Release Checklist

Every PR executing against this master plan must verify:

```bash
# 1. Rust workspace integrity
cargo fmt --all --check
cargo clippy --workspace --all-targets --features forge-cloud/gcs -- -D warnings
cargo test --workspace --features forge-cloud/gcs

# 2. Invariants & security
make alloc-audit          # Zero allocation on hot path
make mutants              # Mutation testing on mc-runner & forge-server
make deny                 # Cargo dependency and license audit
make gitleaks             # Secret scan across repository
make pin-check            # Rust/ONNX/wasm-pack pin consistency
make text-check           # CRLF and line encoding integrity
make ci-parity            # Local vs CI target parity

# 3. TypeScript & Python suites
cd mc-bot && npm run typecheck && npm run lint && npm run test:coverage
cd dashboard && npm run build && npm run lint && npm run test:coverage
python3 -m unittest discover -s .claude/hooks -p 'test_*.py'
```
