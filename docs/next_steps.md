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

### 5b. ✅ LM Studio Offline Teacher Pipeline — COMPLETED

Behavioural-cloning data pipeline driven by a locally-served Gemma 4 e4b
(default) or Qwen 2.5 14B Instruct, or any other OpenAI-compatible
endpoint. One LLM call per env step amortises across four trainers (BC,
BDI, Constitutional, future RSSM).

- `LMStudioProvider` + async `acomplete` path in `forge.cognitive.providers`
- `PromptBuilder` + JSON Schema for deterministic structured decisions
- `StructuredLLMAgentConfig` / `LLMAgent.aact` JSON-mode parser with
  legacy free-text fallback
- `TeacherDecisionTrace` + `TeacherTraceWriter` for sharded JSONL/gzip
  outputs (composes `forge.traces.trace_logger.TraceLogger`)
- `BCTrainer` (NumPy + optional torch path) + `BCStage` prepended to
  `MangoMASPipeline.run`
- `TeacherConfig` TOML section + `FORGE_TEACHER_*` env overrides + CLI
  flags (`--collection-policy llm` etc.)
- Episode-level concurrency via `asyncio.Semaphore`; on-disk shards
  written in `episode_index` order so traces are byte-identical for the
  same `base_seed` regardless of `concurrency`.

See `docs/architecture.md` §3.9, `configs/cognitive/gemma_e4b_teacher.toml`
(default), and `configs/cognitive/qwen14b_teacher.toml` (alternative).

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

### 6. GitHub Pages / WASM Live Demo  `[STATUS: not-started]`

Compile `forge-wasm` and serve a **fully-static** demo directly from `gh-pages`:

- No server required — all simulation runs in the browser via WASM
- Replace the SSE backend with in-browser WASM calls
- Enables public shareable demo link

The `crates/forge-wasm/` crate exists and builds (verified during PR #45
post-merge validation, 2026-05-16); the gap is the
`.github/workflows/gh-pages.yml` workflow, the wasm-pack pipeline, and the
demo glue code.

### 7. ✅ Benchmark Regression Tracking — COMPLETED

CI `bench` job uses `critcmp` with a 5% regression threshold. Baselines are cached per-OS and updated on default-branch merges. Criterion results saved as artifacts.

~~Integrate `forge-bench` Criterion results into the demo UI's stats panel:~~

### 8. Replay / Record Mode  `[STATUS: partial]`

The `crates/forge-replay/` crate ships record/replay primitives (extended in
PR #45). Remaining work is the demo UI integration:

- **Record** a run to a JSON replay file (actions + RNG seed)
- **Replay** a recorded session at configurable speed
- Export the world canvas as an animated GIF

---

## Longer-term (v1.0)

### 9. REST API for External Integrations  `[STATUS: not-started]`

Expose FORGE as a proper REST service so external tools (notebooks, ML frameworks) can drive it:

```http
POST /api/env/reset          {"seed": 42, "config": {...}}
POST /api/env/step           {"action": 1}
GET  /api/env/render         → returns ASCII + grid JSON
```

### 10. Multi-Agent Dashboard  `[STATUS: not-started]`

Extend the demo UI world canvas to show:

- Per-agent color coding
- Communication token visualization
- Reward curves per agent

### 11. Task Curriculum Visualizer  `[STATUS: not-started]`

Add a dedicated UI panel for the task system:

- Current task tree visualization (recursive DSL rendering)
- Curriculum tier distribution histogram
- Live success/failure rate as the agent trains

### 12. SB3 / Cleanrl Training Integration  `[STATUS: not-started]`

Add example scripts and CI integration for:

- `train_ppo.py` with Stable Baselines 3
- `train_ppo_cleanrl.py` with CleanRL
- `train_sac_cleanrl.py` with TOML-backed SAC defaults and feature-extractor configuration
- W&B / MLflow logging hooks from FORGE env wrappers
- Pre-trained model checkpoint serving via the demo UI

---

## Sequencing (post-PR #45, 2026-05-16)

Three-tier priority anchored on the validation run captured in
`target/test-reports/summary.md` (PR #45 = `feat(eval): phase 1 — scenario
suite + on-disk artefact persistence`, merge SHA `d5a162b`). PR #45 itself
introduced zero regressions; the items below are pre-existing or newly
surfaced during triage.

### P0 — Next sprint (v0.2 close-out)

- Publish Docker images to Docker Hub (multi-arch + semver tags) — Section 5
  open bullets.
- Opt-in LM Studio CI smoke (`pytest -m lmstudio` against a containerised
  endpoint) — Tech Debt row "Real LM Studio integration smoke test".
- `BCTrainer._train_torch` branch coverage (KL-only + value-loss-only) —
  Tech Debt row "Torch path coverage for `BCTrainer._train_torch`". One of
  the two branches (value-loss) is now covered by
  `test_torch_path_uses_value_loss_when_value_hats_supplied` (2026-05-16);
  KL-only branch remains.
- Fix `forge-server::config::tests::test_from_env_defaults` flake (new TD
  row below) — small, high-signal, blocks `cargo test --workspace` green CI.

### P1 — v0.3 window

- Section 6: `forge-wasm` + wasm-pack + `.github/workflows/gh-pages.yml`.
- Section 8: replay/record demo-UI glue on top of `crates/forge-replay/`.
- `reference_b` hardware baseline JSON populated and committed.
- `dev-dependencies` consolidation across the 15 affected crates.

### P2 — v1.0 + research

- Section 9: `forge-server` REST surface (`/api/env/{reset,step,render}`)
  with `utoipa` OpenAPI generation (utoipa not yet in any workspace
  `Cargo.toml`).
- Section 10: multi-agent dashboard (Oklab palette, comm overlay,
  per-agent reward streams).
- Section 11: task curriculum visualizer.
- Section 12: SB3 / CleanRL / SAC integrations with
  `LoggingConfig.backend = wandb | mlflow`.
- DPO trainer in `python/forge/mangomas/dpo_trainer.py` consuming
  `TeacherDecisionTrace.top_k_probs` / `value_hat` (struct + fields
  confirmed at `python/forge/mangomas/teacher_trace.py:35-56`).
- Vectorised step-level teacher concurrency (gated on sub-100ms quantised
  inference being available).

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
| Real LM Studio integration smoke test | Low | Today CI exercises only the mocked provider path. Add an opt-in `pytest -m lmstudio` job that spins up the `mlc-llm/qwen` Docker image and runs a 1-episode hex_patrol collection end-to-end. |
| Torch path coverage for `BCTrainer._train_torch` | ✅ Done | KL-only branch (lines 336-339) covered by `test_torch_path_kl_only_branch` (parametrised over `DEFAULT_BC_KL_WEIGHT` active vs. `0.0`, deterministic via `DEFAULT_BC_SEED`); value-loss branch covered by `test_torch_path_uses_value_loss_when_value_hats_supplied` (2026-05-16). Inline `_ToyActorCritic` consolidated into module-scoped `toy_actor_critic_factory` fixture (no duplication across the three torch tests). Local coverage on `bc_trainer.py` rose to 97.45% with all torch-path branches reached. |
| DAgger / DPO follow-on for the teacher pipeline | Medium | Out of scope for the BC PR but a natural next step. The teacher trace schema (`TeacherDecisionTrace`) already records `top_k_probs` and `value_hat`, which DPO would consume directly. Belongs in a separate `forge.mangomas.dpo_trainer` module. |
| Vectorised step-level teacher concurrency | Low | Today concurrency is at the episode level (one LLM call per step, parallelised across episodes). Step-level batching would require a real vec-env under the teacher and is not justified at Qwen 14B latencies. Revisit when sub-100ms quantised inference is available. |
| Workspace `dev-dependencies` consolidation | Low | `proptest` + `tracing-subscriber` declared per-crate in 15 crates; promote to `workspace.dev-dependencies` |
| Commented-out `println!` in `forge-data` | Low | Either delete or convert to `tracing::info!` in `generator.rs`, `lib.rs`, `minari.rs`, `maze.rs`, `edge_replay.rs` |
| `forge-server::config::tests::test_from_env_defaults` env-pollution flake | ✅ Done | Fixed by wrapping `test_from_env_defaults` and `test_from_env_defaults_when_no_env_vars` with the existing `ENV_LOCK` + `EnvScope::new(ALL_KEYS)` machinery in `crates/forge-server/src/config.rs` (the helpers already existed in `mod tests` for the other override tests). Added `eprintln!` diagnostic inside `EnvScope::new` for forensic visibility under `--nocapture`. Reproducer (20 iterations with ambient `FORGE_SERVER_PORT=7777`): pre-fix 20/20 fail, post-fix 0/20 fail. |
| `forge_demo.py demo_day_night` uses out-of-range `default_vision_radius=5` | Medium | The `AgentConfig.default_vision_radius` range was tightened to `[0, 4]` at commit `c3741fe`; `examples/forge_demo.py:492 demo_day_night()` still passes `5`, raising `ValueError: configuration error: ...value 5 out of range [0, 4]` and breaking the demo's day/night section. Either widen the range or update the demo's config. Caught by `demo_ui/tests/test_sections.py::test_section_contains_keywords[daynight-keywords5]`. |
| `torch.jit.trace` deprecation in MuZero export | Low | `python/forge/models/muzero_export.py:145` calls `torch.jit.trace`, which torch is deprecating in favour of `torch.compile` / `torch.export`. Pre-empt removal: switch the rep/dyn/pred traces to `torch.export`. Surfaced as 4 failures in the `numpy-deprec` sweep (`pytest -W error::DeprecationWarning`). |
| Dashboard `npm ci` does not install `eslint` into `node_modules/.bin/` | Medium | `dashboard/package.json` lists `eslint ^8.57.0` in devDependencies but a fresh `npm ci` produces no `node_modules/.bin/eslint`. Likely lock-file drift or a missing peer-dep (`@typescript-eslint/parser` + `@typescript-eslint/eslint-plugin` are not in devDependencies either, yet `lint` script targets `.ts,.tsx`). Fix the dep chain or convert `lint` to a Vite/Biome-based runner. |
| Local `mypy python` walks into installed `torch` and fails on Python-3.10 `match` syntax | Low | CI's lint job installs `mypy<2.0` + `numpy<2.0` only (no torch), so it never hits the issue. Local devs with torch installed see `torch/fx/experimental/symbolic_shapes.py:6069: Pattern matching is only supported in Python 3.10 and greater`. Fix: either raise `[tool.mypy] python_version` to `"3.11"` (matches the supported runtime) or add `[[tool.mypy.overrides]] module = "torch.*"` with `ignore_errors = true`. |
