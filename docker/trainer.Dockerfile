# Dockerfile for the v0.4 self-improving trainer service.
#
# Builds a Python image with the FORGE `[minecraft]` extras (torch,
# onnx, onnxruntime) so the trainer can consume runner-emitted
# trajectories, export ONNX bundles, and bump the manifest the
# runner's `HotReloadWatcher` polls.
#
# Image variants via `--build-arg TORCH_VARIANT={cpu,cu121}`:
#   - `cpu` (default) — CPU-only torch wheel; ~700 MB image.
#   - `cu121` — CUDA 12.1 torch wheel; requires nvidia-container-toolkit
#     on the host AND the GPU compose overlay (compose.minecraft.gpu.yml).
#
# numpy is pinned `<2.0` to match the workspace's CI gate (the savez
# stub regression PR #58 documented hasn't shipped a numpy-2.x fix yet).
#
# Single-stage build: pip install + copy source. Multi-stage would
# trim ~200 MB but adds complexity for a trainer image that's
# already cache-friendly via `--mount=type=cache`.

ARG TORCH_VARIANT=cpu
FROM python:3.11-slim AS base

# ---- runtime tooling ------------------------------------------------
RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- python deps ----------------------------------------------------
# Copy the bare minimum needed for `pip install -e .[minecraft]`:
# the pyproject + the source tree under `python/`.
COPY pyproject.toml /app/pyproject.toml
COPY python /app/python

# Install the right torch wheel via index-url based on TORCH_VARIANT.
# Default is CPU-only; pass `--build-arg TORCH_VARIANT=cu121` to get
# CUDA-enabled wheels. Numpy pinned <2.0 per the CI gate.
# Pinned minor versions so a rebuild on a different day pulls the
# SAME wheels. Wide-open `>=2.0,<3.0` ranges produce drift across
# operator hosts; pinning a single minor track gives reproducibility
# without sacrificing security patches (still picks the latest
# 2.4.x). Bump together with the workspace's `numpy<2.0` gate.
ARG TORCH_VARIANT
ARG TORCH_VERSION=2.4.1
ARG TORCHVISION_VERSION=0.19.1
ARG ONNX_VERSION=1.17.0
ARG ONNXRUNTIME_VERSION=1.20.0
RUN pip install --upgrade pip \
    && if [ "${TORCH_VARIANT}" = "cu121" ]; then \
         pip install --no-cache-dir \
           --index-url https://download.pytorch.org/whl/cu121 \
           "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"; \
       else \
         pip install --no-cache-dir \
           --index-url https://download.pytorch.org/whl/cpu \
           "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"; \
       fi \
    && pip install --no-cache-dir \
         "onnx==${ONNX_VERSION}" \
         "onnxruntime==${ONNXRUNTIME_VERSION}" \
         "numpy>=1.26,<2.0" \
         "tomli; python_version < '3.11'"

# Install the FORGE package itself in editable mode so changes to
# `python/forge/training/muzero_mc/` flow into the container without
# a rebuild (compose `develop` watch can be wired up by operators).
RUN pip install --no-cache-dir -e .

# ---- entrypoint -----------------------------------------------------
# Default command runs the continuous trainer. `mc_self_play.sh`
# invokes the bootstrap one-shot via `docker compose run --rm
# trainer-bootstrap bootstrap ...` which overrides this default.
#
# Every CLI flag has a corresponding env var (TRAINER_*) read from
# the compose-mounted .env file; no values are baked in.
CMD ["python", "-u", "-m", "forge.training.muzero_mc.cli", "train", \
     "--continuous", \
     "--input", "/app/trajectories", \
     "--out", "/app/models"]
