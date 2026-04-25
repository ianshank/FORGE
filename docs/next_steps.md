# FORGE — Next Steps & Roadmap

Post-demo-UI priorities, roughly in order of impact.

---

## Immediate (v0.2)

### 0. ✅ Hex Grid Rollout Hardening — COMPLETED

5 hex integration tests added (`test_hex_episode_full_cycle`, `test_hex_deterministic_replay`, `test_hex_multi_agent_episode`, `test_hex_grid_ignores_square_move`, `test_hex_serialization_roundtrip`), 2 Criterion benchmark groups (`step_hex_single_agent`, `step_hex_multi_agent`), and topology dispatch validated for performance regression.

~~This branch introduces a reusable topology layer (`forge-civ`) plus hex-grid movement, visibility, and scenario support. The next hardening pass should focus on parity and integration, not more feature sprawl.~~

### 1. ✅ MangoMAS Bridge End-to-End Smoke Paths — COMPLETED

21 smoke tests in `tests/python/test_mangomas_smoke.py` exercise TOML config loading, curriculum controller tier progression, constitutional trainer shaping, curiosity optimizer search, batch collector episode loops, and sweep runner parameter iteration. All tests run without the native Rust extension.

~~The branch now has stronger unit coverage for MangoMAS config resolution, curriculum control, constitutional shaping, curiosity optimization, and MCTS sweep reporting, but the next gap is end-to-end execution.~~

### 2. ✅ Python Validation Split In CI — COMPLETED

CI now has three separate Python jobs: `python-lint` (ruff + mypy, no build), `python-test-fast` (pytest without native extension), and `python-test` (full maturin build + coverage-gated pytest at 85%).

~~The Python package surface is now coverage-gated and more resilient to missing optional dependencies.~~

### 3. ✅ Playwright E2E Browser Tests — COMPLETED

`demo_ui/tests/test_e2e_browser.py` now ships four browser-driven tests
on top of the existing page-load checks: `TestSectionStateMachine`
(IDLE → RUNNING → PASS via real DOM click), `TestTerminalStream`
(span-count threshold + no `.c-fail` spans + no console errors),
`TestProgressBar` (`runAll`-driven progress advance + Stop), and
`TestWorldCanvas` (non-zero pixels via `getImageData`). Function-scoped
`fresh_page` isolates per-test DOM. Selector / timeout / span-threshold
constants are extracted at module level for tuning.

### 4. ✅ GitHub Actions CI for Demo UI — COMPLETED

The existing `demo-ui` job now installs `demo_ui[dev]`, runs
`playwright install --with-deps chromium`, and executes the full E2E
suite (previously skipped via `pytest.importorskip`). `actions/setup-python@v5`
gained `cache: pip` keyed off `demo_ui/backend/requirements.txt` and
`demo_ui/pyproject.toml`. The httpx health-smoke step is unchanged and
still runs after the test step. `playwright==1.48.*` is pinned in
`demo_ui/pyproject.toml` for reproducibility.

### 5. ✅ Docker Container for Demo UI — COMPLETED

The full three-service Docker Compose stack is now deployed:

- `docker/Dockerfile.dashboard` — React SPA served via nginx:alpine
- `docker/Dockerfile` — Rust simulation server + forge_env native extension
- `docker/Dockerfile.demo` — FastAPI demo UI
- `docker/docker-compose.yml` — Orchestration with health-gated `depends_on`, bridge network, restart policies
- `docker/nginx.conf` — SPA routing + reverse proxy for `/api/` and `/ws`
- `.dockerignore` — Optimized build contexts

**New Docker next steps:**

- Publish images to Docker Hub (`ianshank/forge-simulation`, `forge-dashboard`, `forge-demo`)
- Add multi-arch builds (`linux/amd64` + `linux/arm64`) via `docker buildx`
- Tag images on GitHub release with semantic versions

---

## Near-term (v0.3)

### 6. GitHub Pages / WASM Live Demo

Compile `forge-wasm` and serve a **fully-static** demo directly from `gh-pages`:

- No server required — all simulation runs in the browser via WASM
- Replace the SSE backend with in-browser WASM calls
- Enables public shareable demo link

### 7. ✅ Benchmark Regression Tracking — COMPLETED

CI `bench` job uses `critcmp` with a 5% regression threshold. Baselines are cached per-OS and updated on default-branch merges. Criterion results saved as artifacts.

~~Integrate `forge-bench` Criterion results into the demo UI's stats panel:~~

### 8. Replay / Record Mode

Allow the demo UI to:

- **Record** a run to a JSON replay file (actions + RNG seed)
- **Replay** a recorded session at configurable speed
- Export the world canvas as an animated GIF

---

## Longer-term (v1.0)

### 9. REST API for External Integrations

Expose FORGE as a proper REST service so external tools (notebooks, ML frameworks) can drive it:

```http
POST /api/env/reset          {"seed": 42, "config": {...}}
POST /api/env/step           {"action": 1}
GET  /api/env/render         → returns ASCII + grid JSON
```

### 10. Multi-Agent Dashboard

Extend the demo UI world canvas to show:

- Per-agent color coding
- Communication token visualization
- Reward curves per agent

### 11. Task Curriculum Visualizer

Add a dedicated UI panel for the task system:

- Current task tree visualization (recursive DSL rendering)
- Curriculum tier distribution histogram
- Live success/failure rate as the agent trains

### 12. SB3 / Cleanrl Training Integration

Add example scripts and CI integration for:

- `train_ppo.py` with Stable Baselines 3
- `train_ppo_cleanrl.py` with CleanRL
- `train_sac_cleanrl.py` with TOML-backed SAC defaults and feature-extractor configuration
- W&B / MLflow logging hooks from FORGE env wrappers
- Pre-trained model checkpoint serving via the demo UI

---

## Technical Debt

| Item | Priority | Notes |
|---|---|---|
| Mypy strict mode for `demo_ui/tests/` | Low | Test files have unannotated optional args |
| `conftest.py` root sys.path approach | Medium | Consider `pyproject.toml` package install instead |
| `demo_ui` as installable package | Medium | `pip install -e demo_ui/` makes imports cleaner |
| Section-level output parsing | Low | Parse structured data from `forge_demo.py` for richer stats |
| `run_demo.ps1` → `run_demo.sh` cross-platform | Medium | Add Bash launcher for Linux/macOS users |
| Python coverage gate maintenance | Medium | Keep new modules above the 85% floor as Python surface area grows |
| Native vs pure-Python test split | ✅ Done | Split into `python-test-fast` and `python-test` CI jobs |
| MangoMAS integration smoke tests | ✅ Done | 21 smoke tests in `test_mangomas_smoke.py` |
| Allocation audit (`step_into` zero-alloc invariant) | ✅ Done | `crates/forge-bench/src/bin/allocation_audit.rs` + `benchmarks/runner/check_zero_alloc.py` wired into CI as the `alloc-audit` job; baseline snapshot in `benchmarks/baselines/reference_a/alloc_audit.json` |
| Multi-agent allocation audit | ✅ Done | `--agents <list>` flag added (default sweep `1,8,16,32,64,128`, also via `FORGE_BENCH_AGENT_COUNTS`); rows labelled `<base>@n=<count>` with typed `num_agents` field. Surfaced and fixed a real per-step heap allocation in `forge-core::physics::process_movements_with_scratch` (snapshot moved into `PhysicsScratch::agents_snapshot`). Reference baseline regenerated; all 72 rows zero-alloc |
| `reference_b` hardware profile | Scaffolded | Directory committed with `.gitkeep`; `benchmarks/baselines/README.md` documents the regeneration command (including `--agents` flag and `FORGE_BENCH_AGENT_COUNTS`). User runs locally on workstation hardware and commits the JSON in a follow-up PR |
| Replace `panic!` on enum variants in non-step crates | Medium | ~20 sites in `crates/forge-proposal`, `crates/forge-server/src/ws_handler.rs`, `crates/forge-mangomas/src/curriculum/task_mapping.rs`, and `crates/forge-types/src/{task,action}.rs` should become `Result` returns or `unreachable!` with safety proof. Out of scope for the alloc-fix branch |
| Replace `.unwrap()` on TOML parsing in `crates/forge-scenario/src/config.rs` | Medium | ~15 sites; move to `?` and propagate `Result` |
| Workspace `dev-dependencies` consolidation | Low | `proptest` + `tracing-subscriber` declared per-crate in 15 crates; promote to `workspace.dev-dependencies` |
| Commented-out `println!` in `forge-data` | Low | Either delete or convert to `tracing::info!` in `generator.rs`, `lib.rs`, `minari.rs`, `maze.rs`, `edge_replay.rs` |
