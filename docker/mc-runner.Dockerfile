# syntax=docker/dockerfile:1.6
#
# Container image for `crates/forge-mc-runner` — the Rust binary that
# drives the latent-MCTS episode loop against the mc-bot WebSocket
# bridge and writes TrajectoryV2 trajectories.
#
# Builds with the `mc-live-bundled` Cargo feature so the resulting
# image needs no host-side ONNX Runtime installation (ort's
# `download-binaries` fetches the right shared lib at build time).
#
# Build (from repo root):
#   docker build \
#     -f docker/mc-runner.Dockerfile \
#     -t forge-mc-runner:dev .
#
# Run (via docker compose, see docker/compose.minecraft.yml):
#   docker compose -f docker/compose.minecraft.yml up runner

ARG RUST_IMAGE_TAG=1.93-bookworm

ARG ONNXRUNTIME_VERSION=1.22.0

# --- builder stage -----------------------------------------------------
FROM rust:${RUST_IMAGE_TAG} AS builder
ARG ONNXRUNTIME_VERSION
WORKDIR /build

# Pre-install build-essential bits.  Trained-mode (`--features
# mc-live-bundled`) needs `curl` + the ONNX Runtime tarball install
# below; the v0.5-Phase-1 random-baseline build does NOT.  Until the
# Dockerfile is parameterised by build-arg, the trained-mode block
# is commented out below for the operator to re-enable.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        pkg-config libssl-dev ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Trained-mode (`--features mc-live-bundled`) only — re-enable this
# block + the runtime-stage `COPY --from=builder /opt/onnxruntime/`
# + the `ORT_DYLIB_PATH` env var below when building the ONNX runner.
# RUN apt-get update && apt-get install -y --no-install-recommends curl && \
#     curl -fsSL -o /tmp/onnxruntime.tgz \
#         "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz" && \
#     tar -xzf /tmp/onnxruntime.tgz -C /opt && \
#     mv "/opt/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}" /opt/onnxruntime && \
#     rm /tmp/onnxruntime.tgz

# Manifests first so Cargo's dep-resolution layer caches separately
# from source-code changes.  Lock is intentionally NOT copied — the
# workspace lock pins versions that conflict with ort-sys
# 2.0.0-rc.12's build-script tracing-subscriber expectations; letting
# Cargo resolve fresh inside the image picks a compatible set.  For
# the v0.5 first-real-run capture this is acceptable; CI-side
# image builds should re-pin once the workspace upgrades.
COPY Cargo.toml ./
COPY crates/ crates/

# v0.5 Phase 1: build with just `mc-live` (no ONNX). The trained-
# mode binary requires `--features mc-live-bundled` instead, which
# pulls in ort + the onnxruntime shared lib bundle. For the random-
# baseline capture image, the lighter `mc-live` build skips the
# ort/onnxruntime cascade entirely (saves ~5 min build time + dodges
# the ort rc.12 transitive-dep churn).
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    cargo build --release \
        -p forge-mc-runner \
        --features mc-live && \
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

# Trained-mode-only (`--features mc-live-bundled`): re-enable both
# the COPY and the two ENV lines below + uncomment the build-stage
# tarball install above to ship the ONNX Runtime shared lib.
# COPY --from=builder /opt/onnxruntime/lib/ /usr/local/lib/onnxruntime/
# ENV ORT_DYLIB_PATH=/usr/local/lib/onnxruntime/libonnxruntime.so
# ENV LD_LIBRARY_PATH=/usr/local/lib/onnxruntime:${LD_LIBRARY_PATH}

USER forge
WORKDIR /home/forge

COPY --from=builder --chown=forge:forge \
    /usr/local/bin/forge-mc-runner /usr/local/bin/forge-mc-runner

# Default metrics port — overridable per the runner's TOML.
EXPOSE 9090

ENTRYPOINT ["/usr/local/bin/forge-mc-runner"]
