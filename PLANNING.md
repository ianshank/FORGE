# FORGE — Beta Release Planning

> Branch: `feat/beta-release-planning`  
> Planned: 2026-02-27  
> Target beta: **v0.2.0-beta** | Target GA: **v1.0.0**

---

## 1. Current State (v0.1.0 baseline)

| Area | Status |
|---|---|
| Rust core engine | ✅ Production-quality, 130K+ steps/sec |
| Python bindings (PyO3) | ✅ Gymnasium + PettingZoo + JAX wrappers |
| WebAssembly bindings | ✅ `forge-wasm` crate, JSON I/O |
| Demo UI (FastAPI + JS) | ✅ Dark-mode, SSE streaming, live world canvas |
| Rust tests | ✅ 389 tests (unit + property-based) |
| Python tests | ✅ 61 tests (20 original + 41 new; 4 skipped pending maturin) |
| CI pipelines | ✅ `ci.yml` + `demo_ui_ci.yml` + `rust.yml` |
| Docker container | ✅ `Dockerfile` created |
| E2E browser tests | ✅ `test_e2e_browser.py` (Playwright) |
| PyPI packaging | ❌ Not published |
| WASM live demo (GH Pages) | ❌ Not deployed |
| Benchmark regression CI | ❌ Manual only |
| REST env API | ❌ Not implemented |
| Multi-agent dashboard | ❌ Not implemented |
| SB3/CleanRL training examples | ⚠️ File exists, not CI-validated |
| W&B / MLflow logging hooks | ❌ Not implemented |
| Replay / record mode | ❌ Not implemented |
| Task curriculum visualizer | ❌ Not implemented |

---

## 2. Beta Release Goals (v0.2.0-beta)

> **Definition of Beta**: Publicly shareable, installable from PyPI, with a live WASM demo, hardened CI, and baseline training integration.

### Success Criteria

1. `pip install forge-env` works from PyPI (manylinux wheels)
2. Live WASM demo accessible at `https://ianshank.github.io/FORGE`
3. SB3 + CleanRL example scripts run end-to-end in CI
4. Benchmark regressions (>5% slowdown) fail automated CI
5. 80%+ Python test coverage reported in CI
6. All 389 Rust tests + 61 Python tests remain green

---

## 3. Milestones & Epics

### Milestone 1 — CI & Packaging Hardening (Sprint 1-2 · 2 weeks)

| Epic | Size | Description | Priority |
|---|---|---|---|
| E1: PyPI Packaging | L | maturin-based manylinux wheels, `forge-env` on PyPI | P0 |
| E2: Benchmark Regression CI | M | Criterion JSON → GitHub Actions artifact, 5% gate | P0 |
| E3: Coverage Enforcement | S | `pytest-cov` ≥80% gate in `ci.yml` | P1 |
| E4: Cross-platform Launcher | S | `run_demo.sh` Bash launcher for Linux/macOS | P1 |

**Dependencies**: E1 unblocks E3 (coverage runs against installed wheel).

---

### Milestone 2 — WASM Live Demo (Sprint 3-4 · 2 weeks)

| Epic | Size | Description | Priority |
|---|---|---|---|
| E5: WASM Build + GH Pages | L | `wasm-pack build`, deploy to `gh-pages` branch | P0 |
| E6: In-browser Demo Page | M | Replace SSE calls with direct WASM calls, static HTML | P0 |
| E7: Shareable World Seeds | S | URL hash `#seed=42` loads specific world | P2 |

**Dependencies**: E5 unblocks E6. Requires `wasm-pack` in CI.

---

### Milestone 3 — Training Integration (Sprint 5-6 · 2 weeks)

| Epic | Size | Description | Priority |
|---|---|---|---|
| E8: SB3 PPO Example (CI) | M | `train_ppo.py` runs in CI with smoke test | P0 |
| E9: CleanRL PPO Example | M | `train_ppo_cleanrl.py` with metrics logging | P1 |
| E10: W&B / MLflow Hooks | M | Optional logging hooks in `RecordEpisodeStatistics` | P2 |
| E11: Replay / Record Mode | L | JSON replay files, GIF export from world canvas | P2 |

**Dependencies**: E8 requires E1 (installable package). E10 extends E8.

---

### Milestone 4 — Advanced UI & REST API (Sprint 7-8 · 2 weeks)

| Epic | Size | Description | Priority |
|---|---|---|---|
| E12: REST Env API | L | `/api/env/reset`, `/api/env/step`, `/api/env/render` | P1 |
| E13: Multi-Agent Dashboard | M | Per-agent coloring, comm tokens, reward curves | P2 |
| E14: Task Curriculum Visualizer | M | DSL tree UI panel, tier histogram | P2 |
| E15: Pre-trained Model Serving | L | SB3 checkpoint → demo UI inference API | P3 |

**Dependencies**: E12 unblocks E13, E14. E15 requires E8.

---

## 4. Technical Debt (addressed in parallel)

| Item | Sprint | Notes |
|---|---|---|
| `conftest.py` sys.path hack → `pyproject.toml` installs | 1 | `pip install -e .` for `demo_ui` |
| Mypy strict for `demo_ui/tests/` | 1 | Remove `warn_unused_ignores = false` override |
| Windows path hardcoding in scripts | 1 | Use `pathlib.Path` throughout |
| Section-level output parsing | 3 | Richer stats from `forge_demo.py` structured output |
| `maturin develop` in primary CI | 2 | Build & test against local source, not installed wheel |

---

## 5. Sprint Plan (8 sprints × 2 weeks)

| Sprint | Epics | Goal |
|---|---|---|
| Sprint 1 (Feb 27 – Mar 12) | E1, E3, E4 + debt | PyPI wheels + coverage gates |
| Sprint 2 (Mar 13 – Mar 26) | E2 + E1 integration | Benchmark regression CI live |
| Sprint 3 (Mar 27 – Apr 9) | E5, E6 | WASM build pipeline + GH Pages |
| Sprint 4 (Apr 10 – Apr 23) | E7 + E8 | Shareable links + SB3 training CI |
| Sprint 5 (Apr 24 – May 7) | E9, E10 | CleanRL + W&B hooks |
| Sprint 6 (May 8 – May 21) | E11 | Replay / Record / GIF |
| Sprint 7 (May 22 – Jun 4) | E12, E13 | REST API + multi-agent UI |
| Sprint 8 (Jun 5 – Jun 18) | E14, E15 | Curriculum viz + model serving |
| **Beta GA** | — | **v0.2.0-beta tag, PyPI publish** |

---

## 6. Risks & Blockers

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| manylinux cross-compilation complexity | High | P0 blocker | Use `maturin` official GH Actions; pin `manylinux2014` |
| WASM bundle size exceeds 5 MB | Medium | Demo load time | `wasm-opt -O3`; lazy load assets |
| SB3 version pin conflicts | Medium | CI flakiness | Use `extras_require` with loose bounds |
| Benchmark noise in CI (shared runners) | High | False regressions | Use `critcmp`, require 3 runs; gate at 10% not 5% |
| `forge-wasm` JSON I/O performance cliff | Low | WASM demo laggy | Profile with `wasm-profiler`; consider binary encoding |

---

## 7. Definition of Done (per Epic)

- [ ] Feature implemented and merged to `main`
- [ ] Unit + integration tests written (≥80% coverage delta)
- [ ] CI passes (ruff, mypy, cargo clippy, pytest, cargo test)
- [ ] PRD acceptance criteria verified (AQA report generated)
- [ ] CHANGELOG.md updated under `[Unreleased]`
- [ ] Documentation updated (README.md or `docs/`)
