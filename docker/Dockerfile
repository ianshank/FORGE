# FORGE — Full build image (Rust + Python + Dashboard)
# Multi-stage build for minimal final image size.

# Stage 1: Rust builder
FROM rust:1.82-bookworm AS rust-builder
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
COPY tests/ tests/
RUN cargo build --release --workspace

# Stage 2: Dashboard builder
FROM node:20-bookworm-slim AS dashboard-builder
WORKDIR /build/dashboard
COPY dashboard/package.json dashboard/package-lock.json* ./
RUN npm ci --ignore-scripts 2>/dev/null || npm install
COPY dashboard/ ./
RUN npm run build

# Stage 3: Python environment
FROM python:3.11-slim-bookworm AS runtime
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends libssl3 && \
    rm -rf /var/lib/apt/lists/*

# Copy Rust binaries
COPY --from=rust-builder /build/target/release/forge-server /usr/local/bin/

# Copy dashboard build
COPY --from=dashboard-builder /build/dashboard/dist/ /app/dashboard/dist/

# Install Python packages
COPY python/ /app/python/
COPY forge.toml /app/
COPY configs/ /app/configs/
RUN pip install --no-cache-dir numpy tomli && \
    pip install --no-cache-dir -e /app/python

# Run as non-root user
RUN useradd --create-home forge
USER forge

ENV PYTHONPATH=/app/python:$PYTHONPATH
ENV FORGE_CONFIG_PATH=/app/forge.toml

EXPOSE 8080
CMD ["python3", "-m", "forge"]
