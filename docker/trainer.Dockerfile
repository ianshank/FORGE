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
# Python minor is pinned to 3.11 (the CI-proven interpreter) and must stay
# inside torch==${TORCH_VERSION}'s wheel matrix: torch 2.4.x ships
# cp38-cp312 wheels only, and numpy<2.0 has no cp313+ wheels, so a
# Dependabot bump to python:3.14 makes the pip step below fail for both
# variants. Bump the interpreter and TORCH_VERSION together.
FROM python:3.11-slim-bookworm AS base

# ---- runtime tooling ------------------------------------------------
RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- python deps ----------------------------------------------------
# Only the pure-Python `forge` package is needed: the trainer never
# imports the native `forge_env` extension. `pip install -e .` cannot be
# used here -- pyproject's build backend is maturin, which needs the Rust
# workspace + toolchain this image deliberately doesn't carry -- so the
# package is put on PYTHONPATH instead and the `[minecraft]` extras are
# installed explicitly below.
COPY python /app/python
ENV PYTHONPATH=/app/python

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
# Mirrors the `huggingface-hub` floor in pyproject's `[minecraft]` extra
# (needed by `bootstrap --from-hf`).
ARG HF_HUB_SPEC=">=1.28.0"
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
         "huggingface-hub${HF_HUB_SPEC}"

# Fail the build (not the first training round) if the package or the
# requested torch variant is broken.
RUN python -c "import forge.training.muzero_mc.cli, torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# ---- non-root runtime user ------------------------------------------
# This image previously had no USER directive, so the trainer ran as root
# while holding READ-WRITE bind mounts of the host's `models/` and
# `trajectories/` directories (its `_trim_replay_buffer` unlinks files
# there). A container escape — or simply a path bug — got root on host
# paths. Run as an ordinary user instead.
#
# The uid/gid are build args rather than literals because the trainer WRITES
# to host bind mounts: a container uid that doesn't match the owner of the
# host `models/` + `trajectories/` directories cannot write to them. Match
# your host with:
#
#   docker build -f docker/trainer.Dockerfile \
#     --build-arg APP_UID="$(id -u)" --build-arg APP_GID="$(id -g)" .
#
# 1000:1000 is the default because it is the first non-system uid/gid on
# Debian/Ubuntu hosts, i.e. the usual single-user developer account.
ARG APP_UID=1000
ARG APP_GID=1000
ARG APP_USER=forge
# `getent` guards make this idempotent and rebuild-safe when the requested
# uid/gid already exists in the base image, without `|| true` swallowing a
# genuine failure (`set -e` is on via the default shell's `-c` + the &&
# chain, so anything unexpected still fails the build).
RUN if ! getent group "${APP_GID}" >/dev/null; then \
        groupadd --gid "${APP_GID}" "${APP_USER}"; \
    fi \
    && if ! getent passwd "${APP_UID}" >/dev/null; then \
        useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home \
            --shell /usr/sbin/nologin "${APP_USER}"; \
    fi \
    # The runtime user has to read the source tree under /app. Bind mounts (/app/models, /app/trajectories)
    # are attached at RUN time and keep their host ownership — see the note
    # above about matching APP_UID to the host.
    && chown -R "${APP_UID}:${APP_GID}" /app

USER ${APP_UID}:${APP_GID}

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
