# FORGE — Docker Compose Stack
# Multi-stage build: builder (maturin wheel) → demo-ui → forge-env-api
#
# Usage:
#   docker compose build          # build both images
#   docker compose up -d          # start services
#   docker compose --profile dev up  # include dev tools
#   docker compose down -v        # stop + remove volumes
#
# Environment variables (see .env.example):
#   DEMO_UI_PORT    Host port for the demo UI     (default: 8080)
#   API_PORT        Host port for the env API     (default: 8765)
#   FORGE_MAX_SESSIONS, FORGE_SESSION_TTL, FORGE_CORS_ORIGINS

# ── Stage 1: Rust + maturin wheel builder ───────────────────────────────────
FROM rust:1.75-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev libssl-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /forge

# Cache cargo dependencies
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/

RUN pip install --no-cache-dir maturin==1.4.*

# Copy Python sources needed for the build
COPY python/ python/
COPY pyproject.toml ./

RUN maturin build --release --out /dist

# ── Stage 2: Demo UI (FastAPI + SSE streaming) ──────────────────────────────
FROM python:3.11-slim AS demo-ui

WORKDIR /forge

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/forge"

# Install wheel + demo_ui deps
COPY --from=builder /dist/*.whl /tmp/wheels/
COPY demo_ui/backend/requirements.txt /tmp/demo_requirements.txt

RUN pip install --no-cache-dir /tmp/wheels/*.whl \
    && pip install --no-cache-dir -r /tmp/demo_requirements.txt \
    && rm -rf /tmp/wheels /tmp/demo_requirements.txt

COPY demo_ui/ demo_ui/
COPY examples/ examples/
COPY python/ python/
COPY conftest.py ./
COPY demo_results.md ./

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

CMD ["python", "-m", "uvicorn", "demo_ui.backend.main:app", \
     "--host", "0.0.0.0", "--port", "8080"]

# ── Stage 3: FORGE Env REST API ─────────────────────────────────────────────
FROM python:3.11-slim AS forge-env-api

WORKDIR /forge

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/forge"

COPY --from=builder /dist/*.whl /tmp/wheels/

RUN pip install --no-cache-dir /tmp/wheels/*.whl \
    && pip install --no-cache-dir "fastapi>=0.110" "uvicorn[standard]>=0.29" "pydantic>=2.0" \
    && rm -rf /tmp/wheels

COPY python/ python/

EXPOSE 8765

HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8765/health')"

CMD ["python", "-m", "uvicorn", "forge_env.api:app", \
     "--host", "0.0.0.0", "--port", "8765"]
