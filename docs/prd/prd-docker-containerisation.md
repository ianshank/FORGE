# PRD — E16: Docker Containerisation & Compose Stack

**Epic Slug**: `docker-containerisation`
**Priority**: P1
**Sprint**: 6 (parallel with E11)
**Size**: M
**Depends on**: E12 (REST Env API ✅), Demo UI (`demo_ui/`)

---

## User Story

> **As a** developer or evaluator wanting to try FORGE,
> **I want to** run `docker compose up` and immediately get a fully functional demo UI and REST API,
> **So that** I can evaluate FORGE without installing Rust, maturin, or any Python packages locally.

---

## Problem Statement

The existing `Dockerfile` serves only the demo UI backend. It hard-codes `demo_ui/backend/requirements.txt`, does not include the REST Env API (`forge_env/api.py`), and has no `docker-compose.yml`. Evaluators face a multi-step setup (Rust toolchain, maturin build, Python env) which creates friction before they see value.

---

## Acceptance Criteria

| # | Given | When | Then |
|---|---|---|---|
| AC1 | Repo is cloned on Linux/macOS/Windows with Docker installed | `docker compose up` | All services start; browser navigates to `http://localhost:8080` and sees demo UI |
| AC2 | Services are running | `curl http://localhost:8765/health` | Returns `{"status": "ok"}` within 2s |
| AC3 | Services are running | `curl -X POST http://localhost:8765/envs -H "Content-Type: application/json" -d "{}"` | Returns `201` with session JSON |
| AC4 | Forge REST API container starts | N/A | Startup log shows "FORGE Env API ready" within 10s |
| AC5 | `docker compose down -v` is run | N/A | All containers stop; no orphan volumes |
| AC6 | `.env` file defines `FORGE_MAX_SESSIONS=8` | Container starts | REST API honours that limit (returns 429 after 8 sessions) |
| AC7 | CI runs on push to `main` | N/A | `docker_ci.yml` workflow builds images and runs AC1–AC4 smoke tests |

---

## Out of Scope

- Kubernetes / Helm chart (post-beta)
- GPU-accelerated container (post-beta)
- Multi-arch image (arm64) — linux/amd64 only for beta

---

## Success Metrics

- `docker compose up` → working demo in ≤ 60s on first run (cached layers)
- Images published to `ghcr.io/ianshank/forge-demo-ui` and `ghcr.io/ianshank/forge-env-api`
- Total image size ≤ 800 MB (slim base + wheel cache)

---

## Open Questions

1. Should the native `forge_env` wheel be pre-built in the Docker image (preferred) or `maturin develop` at container start?
2. Use `python:3.11-slim` (current) or `python:3.11-bookworm` distroless? Recommend slim for size.
3. Should `forge-env-api` and `forge-demo-ui` be the same image (simpler) or separate (better separation)?

---

## Implementation Notes

### Files to create / modify

| File | Action | Description |
|---|---|---|
| `Dockerfile` | Modify | Split into `demo-ui` stage; add `forge-env-api` multi-stage target |
| `docker-compose.yml` | Create | `demo-ui` + `forge-env-api` services; health checks; shared network |
| `.env.example` | Create | Document all env vars: `FORGE_MAX_SESSIONS`, `FORGE_SESSION_TTL`, `FORGE_CORS_ORIGINS`, `DEMO_UI_PORT`, `API_PORT` |
| `.dockerignore` | Create / update | Exclude `target/`, `.git/`, `*.pyd`, `__pycache__/` |
| `.github/workflows/docker_ci.yml` | Create | Build + smoke-test on push; push images on `main` |

### Multi-stage `Dockerfile`

```
Stage 1 (builder): install maturin, build forge-env wheel
Stage 2 (demo-ui): copy wheel, install requirements, serve demo_ui
Stage 3 (forge-env-api): copy wheel, install fastapi/uvicorn, serve forge_env.api
```

### `docker-compose.yml` sketch

```yaml
services:
  demo-ui:
    build: { target: demo-ui }
    ports: ["${DEMO_UI_PORT:-8080}:8080"]
    environment: [FORGE_API_URL=http://forge-env-api:8765]
    depends_on: [forge-env-api]
    healthcheck: { test: ["CMD", "curl", "-f", "http://localhost:8080/health"] }

  forge-env-api:
    build: { target: forge-env-api }
    ports: ["${API_PORT:-8765}:8765"]
    env_file: [.env]
    healthcheck: { test: ["CMD", "curl", "-f", "http://localhost:8765/health"] }
```

### Tests

- `tests/docker/test_docker_smoke.py`: pytest + httpx AC1–AC6 smoke tests (runs inside CI after `docker compose up -d`)
- `.github/workflows/docker_ci.yml`: build → compose up → smoke → push (on main)
