# FORGE — Next Steps & Roadmap

Post-demo-UI priorities, roughly in order of impact.

---

## Minecraft RL Integration — Remaining Phases (PR #53)

Branch `claude/minecraft-rl-agent-integration-xnJjt` landed the
env-trait foundation (Phases 1, 2, 3 Rust, 3 Node, 5 Rust v2 replay)
in PR #53. Phase-4 foundation (config / manifest / hot-reload watcher /
trajectory writer) followed in `claude/minecraft-phase3-wireup-runner-foundation`.

**Status as of branch `feat/mc-phase4-runner-loop` (2026-05-20):**

| Phase | Status | Where |
|---|---|---|
| 1: `forge-env` generic trait | ✅ landed | PR #53 |
| 2: `forge-env-forge` shim | ✅ landed | PR #53 |
| 3a: Rust WS client (`forge-env-mc`) | ✅ landed | PR #53 |
| 3b: Node `mc-bot/` bridge | ✅ landed | PR #53 |
| 4 foundation: config / manifest / watcher / writer | ✅ landed | PR #56 |
| **4 loop: `Runner<E,M>` + `LatentPlanner` + binary** | ✅ **landed** | **commit `b1cc7f8`** |
| **5: Python `muzero_mc` (manifest / replay / bootstrap / CLI)** | ✅ **landed** | **commit `4c31a7c`** |
| **6: Docker compose + mc-bot CI + Biome lint + quickstart** | ✅ **landed** | **commit `e872987`** |
| **4: `OnnxMuZeroModel::reload()` impl** | ✅ **landed** | branch `feat/mc-completion-onnx-trainer-metrics-e2e-ts-gzip` |
| **5: Full MuZero trainer loop** | ✅ **landed** | branch `feat/mc-completion-onnx-trainer-metrics-e2e-ts-gzip` |
| **6: Prometheus `/metrics` endpoint** | ✅ **landed** | branch `feat/mc-completion-onnx-trainer-metrics-e2e-ts-gzip` |
| **6: E2E pytest integration test** | ✅ **landed** (opt-in `workflow_dispatch`) | branch `feat/mc-completion-onnx-trainer-metrics-e2e-ts-gzip` |
| **6: `mc-bot/` TypeScript toolchain** | ✅ **landed** (tsconfig + `tsc --noEmit` gate) | branch `feat/mc-completion-onnx-trainer-metrics-e2e-ts-gzip` |
| **6: Replay storage compression** | ✅ **landed** (opt-in gzip + bomb cap) | branch `feat/mc-completion-onnx-trainer-metrics-e2e-ts-gzip` |
| **6: `mc-bot/` `.js → .ts` file rewrite** | ✅ **landed** (100% strict TypeScript migration) | branch `feat/mc-v05-phase2-agent-play` |

The sections below document what landed on the `feat/mc-phase4-runner-loop`
branch and what specifically remains.

### Phase 3 mineflayer wire-up (Node entry point)

- `mc-bot/src/index.js` — spin up the mineflayer bot, attach
  prismarine-viewer on `viewer.port`, plumb `applyReset` +
  `RewardConfig` + `ActionMap` into the WS event loop.
- Needs a real Minecraft server (Paper/vanilla) to exercise; CI gate
  can use `flying-squid` mock server but real-bot work happens off-CI.
- Action execution mapping (`ActionKind` → mineflayer commands) —
  implementation guide already documented in `mc-bot/src/action_map.js`
  validators and `configs/minecraft/action_map.toml` comments.

### Phase 4 — `forge-mc-runner` + ONNX hot-reload + bench

**Foundation landed (PR: this branch):**

- `crates/forge-mc-runner/` crate scaffolded with the pieces the
  full runner will assemble:
  - `config::RunnerConfig` — episode loop knobs, `#[serde(default)]`,
    `validate()` invariants.
  - `manifest::{ModelManifest, ModelManifestFiles}` — the
    `model_manifest.json` swap signal (schema version, monotonic
    `version`, sha256-per-role). Atomic save (tmp + rename); pinned
    `MANIFEST_SCHEMA_VERSION = 1`.
  - `hot_reload::HotReloadWatcher` — polls for strictly-monotonic
    version bumps; missing manifest = `Ok(None)`; lower versions
    ignored (no downgrade); doc-contract "poll only between
    episodes".
  - `trajectory::TrajectoryWriter` — episode-scoped wrapper over
    `forge_replay::v2::TrajectoryV2`. Directory created on first
    save; wrong-order calls return `RunnerError::WriterState`.
  - 42 unit tests covering happy, error, validation, atomic-cleanup,
    missing-manifest paths.
- `crates/forge-bench/benches/latent_mcts_inference.rs` Criterion
  bench at 1 / 8 / 25 / 50 / 100 / 200 sim budgets using
  `StubLatentModel` (no ONNX dep). Env-tunable via
  `FORGE_BENCH_MCTS_{SIMS,OBS_DIM,ACTIONS,LATENT_DIM}`. **Closes the
  bench gap from the original audit.**

**All three items in this section LANDED** on
`feat/mc-completion-onnx-trainer-metrics-e2e-ts-gzip` (see PR #58):

- `Runner<E: FlatObsEnv, M: LatentForwardModel>` shipped in commit
  `b1cc7f8`.
- `LatentPlanner<M>` shipped in the same commit (folded into
  `Runner::run_episode`).
- `OnnxMuZeroModel::reload(&mut self, new_config)` shipped in
  commit `11240fc`. The design switched from "fixed mutex
  acquisition order" to "build-first-then-swap with `&mut self`"
  — the borrow checker now enforces sequencing against concurrent
  inference, which is strictly stronger than runtime lock-order
  discipline.

### Phase 5 — Python MuZero trainer (`muzero_mc/`)

- `python/forge/training/muzero_mc/` — `trainer.py`, `replay.py`,
  `exporter.py`, `manifest.py`, `bootstrap.py`, `cli.py`.
- Consumes `TrajectoryV2` jsonl, trains representation / dynamics /
  prediction nets in PyTorch, exports ONNX, writes manifest atomic
  (tmp-file + rename) with monotonic version.
- Needs `torch>=2.3` + `onnx>=1.16` + `onnxruntime>=1.18` (will be
  declared under `[project.optional-dependencies] minecraft`).
- Cross-runtime CI test (`tests/python/training/test_onnx_compat.py`)
  exports a tiny bundle from PyTorch and round-trips it through
  Rust's `OnnxMuZeroModel::load` to catch opset / `ort` version drift.

### Phase 6 — End-to-end glue

- `scripts/mc_run.sh` — orchestrates docker-compose + runner + trainer.
- `docker/mc-bot.Dockerfile` + `docker/compose.minecraft.yml` ship a
  Paper server + the Node bot + the Rust runner together.
- `examples/minecraft/quickstart.md` 10-minute getting-started runbook.
- `tests/e2e_full_loop.py` (nightly, not PR-gated) drives a 3-episode
  loop ending with a `model_manifest.version == 1` reload.
- Prometheus `/metrics` endpoint on `runner.metrics_port` exposes
  `forge_mc_episode_total`, `forge_mc_episode_reward_sum`,
  `forge_mc_planning_latency_ms`, `forge_mc_model_version`,
  `forge_mc_protocol_error_total` for any dashboard the team picks.

### Open decisions (deferred from v2 plan §10)

- JS test runner: `node:test` (current, zero-dep) vs `vitest` (faster).
- Mineflayer / MC version pin (proposal: `1.20.4`).
- Replay compression: JSONL plain vs gzip vs zstd — measure first.
- Dashboard panel framework — defer to the existing dashboard team's
  Vite/React stack vs writing a standalone panel.

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

- ✅ Publish images to Docker Hub — `.github/workflows/ci.yml` `docker` job now
  pushes the existing `docker/Dockerfile` (the only image currently published
  by CI) in parallel to GHCR and `docker.io/${{ vars.DOCKERHUB_NAMESPACE }}/forge`.
  Docker Hub leg is opt-in: gated by `if: vars.DOCKERHUB_NAMESPACE != ''` so
  the job stays green until the user provisions:
  - Repo variable `DOCKERHUB_NAMESPACE` (e.g. `ianshank`)
  - Repo variable `DOCKERHUB_USERNAME` (e.g. `ianshank`)
  - Repo secret `DOCKERHUB_TOKEN` (PAT with `Read & Write` on the `forge`
    repo on Docker Hub — least-privilege; `Delete` is NOT required for the
    publish flow and should be withheld unless image-deletion is also wired
    into CI. Create the Docker Hub repo manually before generating the PAT.)
  Follow-up images (`forge-dashboard`, `forge-demo`) wait for the dashboard
  and demo Dockerfiles to be added to the existing CI build (they exist on
  disk under `docker/` but aren't built by the workflow today).
- ✅ Multi-arch (`linux/amd64` + `linux/arm64`) via `docker buildx` — already
  present in the existing job, now applied to both registries.
- ✅ Semver tags on GitHub release — already present via
  `docker/metadata-action@v5` `type=semver,pattern={{version}}` /
  `{{major}}.{{minor}}`, now applied to both registries.
- ✅ Post-push smoke probe: pulls the first GHCR-emitted tag (NOT
  `github.sha`, which is the full 40-char SHA that metadata-action's
  `type=sha,prefix=` never emits) and runs `/health` against
  `127.0.0.1:8080` with a 30 s retry budget. Container `logs` + `inspect`
  dumped on failure for debuggability.

---

## Near-term (v0.4 landed; v0.5 candidates)

### Minecraft RL — v0.4 LANDED (branch `feat/mc-v04-self-improving-loop`)

The v0.4 milestone shipped the **self-improving training loop**:
live runner online + continuous trainer + atomic versioned bundles +
one-command orchestrator. 9 tracks, single PR (#59). Closes the
v0.3-pre `ExitCode 64` BLOCKER ("live runner wiring not yet
integrated").

| Status | Track | Commit |
|---|---|---|
| ✅ | T1 — `compute-schema-id` CLI + Python schema_id twin | `1a31222` |
| ✅ | T2 — `MuZeroMcTrainerConfig.device` + GPU plumbing | `95ea03d` |
| ✅ | T3 — Live runner wiring (BLOCKER) + `FORGE_MC_SCHEMA_ID` ladder | `1725d43` |
| ✅ | T4 — `train --continuous` + cold-start guard + replay-buffer hygiene | `60241d2` |
| ✅ | T4a — Atomic per-version ONNX bundle export (BLOCKER) | `4013118` |
| ✅ | T5 — Compose trainer service + GPU overlay | `a5ffd8e` |
| ✅ | T6 — `scripts/mc_self_play.sh` orchestrator | `0e990f0` |
| ✅ | T7 — Self-improvement smoke (PR-CI) + opt-in milestone | `cab1280` |
| ✅ | T8 — Docs sweep | (this commit) |

### Minecraft RL — v0.5 Phase 1 LANDED (branch `feat/mc-v05-phase1-first-real-run`, PR #60)

The v0.5 Phase 1 milestone closes the v0.4 → real-run gap. Every
test in PR #59 passed against **stubs and mocks**; no one had
actually brought the stack up against a real `itzg/minecraft-server`.
v0.5 Phase 1 fixes the hidden contract violation (mc-bot emitted
31 floats, MuZeroConfig required 920) and ships the operator-facing
tooling for a calibrated random-vs-trained baseline.

| Status | Track | Commit |
|---|---|---|
| ✅ | T1 — mc-bot block-grid encoder (`observation_grid.js`) + `Hello.grid_shape` xlang pin | `fd5785f` |
| ✅ | T2 — `expected_dim=920` flipped across env.toml / mc_self_play.sh / compose env | `fc9b9d7` |
| ✅ | T3 — `--random-actions` runtime switch + `RandomLatentModel` adapter | `2c7cff6` |
| ✅ | T4 — `forge.training.muzero_mc.cli capture-baseline` subcommand + metrics hoist | `540536c` |
| ✅ | T5 — `scripts/mc_plot_baseline.py` Markdown report + matplotlib PNGs | `2958310` |
| ✅ | T6 — opt-in `python-test-minecraft-real-run` CI job | `2958310` |
| ✅ | T7 — cross-cutting logging audit + Rust-side `BLOCK_FEATURE_CHANNELS` pin | `2958310` |
| ✅ | T8 — `docs/results/v0.5-first-real-run.md` skeleton + final report | `2958310` |
| ✅ | T9 — docs sweep (CHANGELOG / README / CLAUDE.md / Agent.md / architecture / next_steps) | `2958310` + this commit |
| ✅ | Hardening pass 1 — 16 peer-review findings folded in | `7d144a6` |
| ✅ | Lint pass — cargo fmt / clippy / ruff sweep | `1797d71` |
| ✅ | Docker infra — `mc-runner.Dockerfile`, `env.docker.toml`, `v05_handshake_probe.py`, `v05_manual_baseline.py`, ort rc.12 forward-port, `live.rs` feature-gate refactor | `cf06bbf` |
| ✅ | Hardening pass 2 — 6 more peer-review findings (env.docker overlay, `_ws_client.py`, tracing events, ships-default-runner.toml test, snapshot schema-compat) | `5acb374` |

**End-to-end validation against a real `itzg/minecraft-server`:**
v0.5 grid_shape handshake verified (`obs_dim=920`,
`grid_shape={11,11,1,7,73}`, `schema_id` byte-matches Rust + JS xlang
constants). First-real-episode rollouts captured (3 full 400-step
episodes + 1 hardened 11-step episode; rewards -7..+40 with random
actions). Full report: [`docs/results/v0.5-first-real-run.md`](results/v0.5-first-real-run.md).

### Minecraft RL — v0.5 Phase 2

All Phase 2 production-stability and code-hardening goals are **COMPLETED**:

- **Mineflayer auto-reconnect on MC-side tick timeout**  `[STATUS: COMPLETED]`
  Implemented connection health monitoring and tick-age checks in `mc-bot/src/bot_manager.ts`. The `BotManager` automatically tears down and rebuilds the mineflayer instance on stale connection detection using exponential backoff, keeping the WebSocket layer continuously alive.
- **`ort 2.0.0-rc.12` forward-port or rc.9 downgrade**  `[STATUS: COMPLETED]`
  Pinned exact version `2.0.0-rc.9` in `crates/forge-agent/Cargo.toml` and reverted `build_session_from_path` to use the robust `commit_from_file` API, fully resolving the FFI error hazards and the `download-binaries` TLS conflict for trained-mode Docker builds.
  (1) `download-binaries` pulls ureq 3.x whose `tls` is
  feature-gated; (2) `load-dynamic` + onnxruntime 1.22.0 hits an
  ABI mismatch (`unknown field CreateEnvWithCustomLoggerAndGlobal
  ThreadPools`); (3) workspace deps require rustc 1.93 (already
  bumped in the v0.5 Dockerfile); (4) workspace Cargo.lock pins
  versions that conflict with rc.12's build-script
  tracing-subscriber expectations. Fix is to either downgrade
  `ort` to a stable rc.9 line or forward-port
  `forge-agent::onnx_model.rs` to whatever rc.12 ABI accepts.
  Random-baseline `mc-live` already works.

Phase-2 RL-specific candidates (deferred from Phase 1's "Out of
scope" + the first-real-run report's next-steps):

- **Bigger random baseline.** With bot auto-reconnect in place, run
  the full 100-ep × 6000-tick baseline overnight (the v0.5 Phase 1
  evidence is 3-4 real rollouts; enough to validate the contract
  but not statistically meaningful as a trained-vs-random
  comparison).  `[STATUS: not-started]`
- **Block-ID embeddings** (replace the `block_type_hash` mod-N
  with a learned lookup table). Eliminates hash collisions and
  gives the CNN a richer per-tile signal.  `[STATUS: not-started]`
- **Resource-acquisition + milestone reward shapers** ("first
  wood", "first stone tool", etc.) so the planner sees a useful
  gradient beyond the current survival/inventory/distance/health
  composite.  `[STATUS: not-started]`
- **HuggingFace pretrained-checkpoint loader** for warm-starts so
  the trained variant doesn't start from random init.  `[STATUS: not-started]`
- **DPO trainer** consuming `top_k_probs` / `value_hat` from the
  trajectory traces (the schema already carries these; the trainer
  doesn't consume them yet).  `[STATUS: not-started]`
- **3D block-grid variant** (`grid_height_radius > 0`) once the
  Conv3d branch is wired on the trainer side. The v0.5 encoder
  defaults to single-Y-layer (`depth=1`) matching MuZero's 2D
  CNN.  `[STATUS: not-started]`

### Minecraft RL — v0.5 Completed Sweeps

- **mc-bot `.js → .ts` file rewrite.**  `[STATUS: COMPLETED]`
  The entire 16 source files and 15 tests have been migrated to strict ESM TypeScript, with all types, interfaces, and registry mappings fully functional.
- **mc-bot `tsconfig` strictness ratcheting** — `[STATUS: COMPLETED]`
  Enabled `"strict": true` and `"noImplicitAny": true` in `tsconfig.json`, passing `npm run typecheck` with 0 errors.
- **Biome formatter and linter sweep** — `[STATUS: COMPLETED]`
  Biome 1.9.4 checks pass cleanly with 0 errors.
- **Multi-threaded `OnnxMuZeroModel` sharing.** Today's `reload(&mut self)` is borrow-checker safe for the single-owner runner. A follow-up `ArcSwap<Sessions>` refactor would let an `Arc<OnnxMuZeroModel>` be shared across inference threads without cross-generation session leakage; only needed when a real multi-threaded inference caller appears. `[STATUS: not-started]`
- **DPO / preference-trainer consuming teacher decision traces.** The trace schema already carries `top_k_probs` / `value_hat`. A DPO-style trainer would consume them as an alternative to the current value+policy distillation loss. `[STATUS: not-started]`
- **Replay-compression level tuning sweep.** Completed. Trajectory compression sweep benchmarks are documented in `docs/results/replay-compression-sweep.md`. `[STATUS: COMPLETED]`

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
