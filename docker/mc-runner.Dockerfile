# syntax=docker/dockerfile:1.6
#
# Container image for `crates/forge-mc-runner` — the Rust binary that
# drives the latent-MCTS episode loop against the mc-bot WebSocket
# bridge and writes TrajectoryV2 trajectories.
#
# Builds with the configured Cargo features so the resulting
# image can be built for either standard random baseline or trained mode.
#
# Build (from repo root):
#   docker build \
#     -f docker/mc-runner.Dockerfile \
#     --build-arg FEATURES=mc-live-bundled \
#     -t forge-mc-runner:dev .
#
# Run (via docker compose, see docker/compose.minecraft.yml):
#   docker compose -f docker/compose.minecraft.yml up runner

ARG RUST_IMAGE_TAG=1.93-bookworm
ARG ONNXRUNTIME_VERSION=1.18.0
ARG FEATURES=mc-live

# --- builder stage -----------------------------------------------------
FROM rust:${RUST_IMAGE_TAG} AS builder
ARG ONNXRUNTIME_VERSION
ARG FEATURES
WORKDIR /build

# Pre-install build-essential bits and curl.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        pkg-config libssl-dev ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# Install ONNX Runtime shared library
RUN curl -fsSL -o /tmp/onnxruntime.tgz \
        "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz" && \
    tar -xzf /tmp/onnxruntime.tgz -C /opt && \
    mv "/opt/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}" /opt/onnxruntime && \
    rm /tmp/onnxruntime.tgz

# Manifests first so Cargo's dep-resolution layer caches separately
COPY Cargo.toml ./
COPY crates/ crates/

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    cargo build --release \
        -p forge-mc-runner \
        --features ${FEATURES} && \
    cp /build/target/release/forge-mc-runner /usr/local/bin/forge-mc-runner

# --- runtime stage -----------------------------------------------------
FROM debian:bookworm-slim AS runtime

LABEL org.opencontainers.image.title="forge-mc-runner" \
      org.opencontainers.image.description="Rust latent-MCTS episode runner for FORGE Minecraft RL" \
      org.opencontainers.image.source="https://github.com/ianshank/FORGE"

# libssl3 + libgomp1 cover the typical native-lib dependencies the
# ONNX Runtime + tokio TLS stack pull in. ca-certificates is needed if
# the runner ever GETs a remote model bundle.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libssl3 libgomp1 ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    useradd --create-home --shell /bin/bash forge

# Copy ONNX Runtime shared library
COPY --from=builder /opt/onnxruntime/lib/ /usr/local/lib/onnxruntime/
ENV ORT_DYLIB_PATH=/usr/local/lib/onnxruntime/libonnxruntime.so
ENV LD_LIBRARY_PATH=/usr/local/lib/onnxruntime:${LD_LIBRARY_PATH}

USER forge
WORKDIR /home/forge

COPY --from=builder --chown=forge:forge \
    /usr/local/bin/forge-mc-runner /usr/local/bin/forge-mc-runner

# Default metrics port — overridable per the runner's TOML.
EXPOSE 9090

ENTRYPOINT ["/usr/local/bin/forge-mc-runner"]
