# Changelog

All notable changes to FORGE will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

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
