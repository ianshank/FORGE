# FORGE — Simulation backend (Rust server + Python forge-env)
# Multi-stage build for minimal final image size.
# Stage 1: Rust builder — compiles forge-server binary
# Stage 2: Wheel builder — compiles forge-env native Python extension
# Stage 3: Runtime    — slim image with only what's needed at runtime

# ---------------------------------------------------------------------------
# Stage 1: Rust builder
# ---------------------------------------------------------------------------
FROM rust:1.85-bookworm AS rust-builder
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
# Note: tests/ is intentionally excluded — not needed for cargo build --release
RUN cargo build --release --workspace

# ---------------------------------------------------------------------------
# Stage 2: Python wheel builder (separate from runtime to avoid Rust toolchain
# and build-essential ending up in the final image — Gemini review: critical)
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS wheel-builder
WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl build-essential && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"

COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
COPY python/ /app/python/
COPY pyproject.toml /app/
RUN pip install --no-cache-dir maturin && \
    maturin build --release -m crates/forge-python/Cargo.toml --out /app/dist

# ---------------------------------------------------------------------------
# Stage 3: Runtime — slim image, no compilers, no Rust toolchain
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime
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
