# FORGE — Simulation backend (Rust server + Python forge-env)
# Multi-stage build for minimal final image size.
# Stage 1: Rust builder — compiles forge-server binary
# Stage 2: Wheel builder — compiles forge-env native Python extension
# Stage 3: Runtime    — slim image with only what's needed at runtime

# ---------------------------------------------------------------------------
# Stage 1: Rust builder
# ---------------------------------------------------------------------------
# Shared Rust toolchain image tag for all builder stages. Must be >= 1.85 so
# cargo can parse edition-2024 dependency manifests (e.g. `fixed` >= 1.30, which
# declares edition = "2024"); older cargo fails with "failed to parse manifest".
# Kept in sync with docker/mc-runner.Dockerfile AND the repo-root
# rust-toolchain.toml pin -- every bare `cargo` CI gate (fmt/clippy/test/
# coverage/onnx-features/etc.) now builds on that pinned version, so an
# un-synced Docker image would be the one artifact CI's toolchain pin
# doesn't actually cover. Override at build time with
# `--build-arg RUST_IMAGE_TAG=...`.
ARG RUST_IMAGE_TAG=1.94.1-bookworm
FROM rust:${RUST_IMAGE_TAG} AS rust-builder
ARG FEATURES=""
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
# Note: tests/ is intentionally excluded — not needed for cargo build --release
#
# When FEATURES is set (e.g. "gcs"), build forge-cloud with that feature
# enabled so feature-gated code is compiled, then build forge-server
# separately.  Avoid building the whole workspace because unrelated crates
# (e.g. forge-python) may require extra tooling only installed in Stage 2.
RUN if [ -n "$FEATURES" ]; then \
      cargo build --release -p forge-cloud --features "$FEATURES" && \
      cargo build --release -p forge-server; \
    else \
      cargo build --release -p forge-server; \
    fi

# ---------------------------------------------------------------------------
# Stage 2: Python wheel builder (separate from runtime to avoid Rust toolchain
# and build-essential ending up in the final image — Gemini review: critical)
# ---------------------------------------------------------------------------
FROM rust:${RUST_IMAGE_TAG} AS wheel-builder
WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-dev python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --break-system-packages maturin

COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
COPY python/ /app/python/
COPY pyproject.toml /app/
RUN maturin build --release -m crates/forge-python/Cargo.toml --out /app/dist

# ---------------------------------------------------------------------------
# Stage 3: Runtime — slim image, no compilers, no Rust toolchain
# ---------------------------------------------------------------------------
FROM python:3.14-slim-bookworm AS runtime
WORKDIR /app

# Runtime-only system deps: libssl3 for TLS, curl for HEALTHCHECK
RUN apt-get update && \
    apt-get install -y --no-install-recommends libssl3 curl && \
    rm -rf /var/lib/apt/lists/*

# Copy compiled Rust server binary from Stage 1
COPY --from=rust-builder /build/target/release/forge-server /usr/local/bin/

# Copy and install the wheel from Stage 2 (no Rust toolchain needed at runtime)
COPY --from=wheel-builder /app/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm -rf /tmp/

# Install remaining Python runtime dependencies
COPY forge.toml /app/
COPY configs/ /app/configs/
RUN pip install --no-cache-dir numpy tomli

# Run as non-root user
RUN useradd --create-home forge
USER forge

ENV PYTHONPATH=/app/python:$PYTHONPATH
ENV FORGE_CONFIG_PATH=/app/forge.toml

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -sf http://localhost:8080/health || exit 1

CMD ["forge-server"]
