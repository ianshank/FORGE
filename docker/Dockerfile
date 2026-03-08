# FORGE — Simulation backend (Rust server + Python forge-env)
# Multi-stage build for minimal final image size.

# Stage 1: Rust builder
FROM rust:1.85-bookworm AS rust-builder
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
COPY tests/ tests/
RUN cargo build --release --workspace

# Stage 2: Python environment with native extension
FROM python:3.11-slim-bookworm AS runtime
WORKDIR /app

# Install system dependencies + Rust toolchain for maturin build
RUN apt-get update && \
    apt-get install -y --no-install-recommends libssl3 curl build-essential && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"

# Copy Rust binaries
COPY --from=rust-builder /build/target/release/forge-server /usr/local/bin/

# Build and install the forge-env Python native extension via maturin
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
COPY python/ /app/python/
COPY pyproject.toml /app/
RUN pip install --no-cache-dir maturin && \
    maturin build --release -m crates/forge-python/Cargo.toml --out /app/dist && \
    pip install --no-cache-dir /app/dist/*.whl && \
    rm -rf /app/dist

# Install remaining Python dependencies
COPY forge.toml /app/
COPY configs/ /app/configs/
RUN pip install --no-cache-dir numpy tomli

# Create non-root user and switch
RUN useradd --create-home forge
USER forge

ENV PYTHONPATH=/app/python:$PYTHONPATH
ENV FORGE_CONFIG_PATH=/app/forge.toml

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -sf http://localhost:8080/health || exit 1

CMD ["forge-server"]
