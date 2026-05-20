# syntax=docker/dockerfile:1.6
#
# Container image for `mc-bot/` — the Node.js bridge that joins a
# Minecraft server via mineflayer and exposes a WebSocket the Rust
# `forge-env-mc` client consumes.
#
# Image conventions:
# - Multi-arch via BuildKit: builds on amd64 + arm64 with `--platform`.
# - Rootless: drops to a non-root `node` user.
# - Reproducible: locks Node via the official slim image, installs
#   from package.json + (if present) package-lock.json with
#   `npm ci`. No global tools, no curl-pipe-bash.
#
# Build (host):
#   docker buildx build \
#     --platform linux/amd64,linux/arm64 \
#     -f docker/mc-bot.Dockerfile \
#     -t ghcr.io/ianshank/forge-mc-bot:dev mc-bot/
#
# Run:
#   docker run --rm -e MC_HOST=... -e WS_PORT=8765 ...

ARG NODE_IMAGE_TAG=22-slim
ARG WS_PORT=8765
ARG VIEWER_PORT=3007

# --- builder stage: install prod dependencies in isolation -------------
FROM node:${NODE_IMAGE_TAG} AS builder

# Pin and harden APT (canvas / prismarine-viewer pull native deps; if
# the upstream package list ever grows, do it here, not in the runtime
# stage). Currently no native deps required for the WS-only bot path.
WORKDIR /app

# Copy manifests first so dependency installation caches separately
# from source-code changes.
COPY package.json package-lock.json* ./

# `npm ci` if a lockfile is present, otherwise `npm install --omit=dev`
# (mc-bot currently ships no lockfile; this works either way).
RUN if [ -f package-lock.json ]; then \
        npm ci --omit=dev; \
    else \
        npm install --omit=dev --no-audit --no-fund; \
    fi

# --- runtime stage -----------------------------------------------------
FROM node:${NODE_IMAGE_TAG}

ARG WS_PORT
ARG VIEWER_PORT

LABEL org.opencontainers.image.title="forge-mc-bot" \
      org.opencontainers.image.description="Mineflayer bridge for FORGE Minecraft RL" \
      org.opencontainers.image.source="https://github.com/ianshank/FORGE"

WORKDIR /app

# Bring the installed node_modules from the builder.
COPY --from=builder /app/node_modules ./node_modules

# Copy the source and the in-tree configs.
COPY src ./src
COPY README.md ./README.md
COPY package.json ./package.json

# Run as non-root by default (the `node` user is created in the
# upstream image).
RUN chown -R node:node /app
USER node

# Network: WebSocket port for forge-env-mc, optional viewer port for
# prismarine-viewer. Both come from build args so callers can rebuild
# with different defaults without editing the file.
EXPOSE ${WS_PORT}
EXPOSE ${VIEWER_PORT}

# Sensible runtime defaults; all overridable via env vars at runtime.
ENV MC_BOT_WS_PORT=${WS_PORT} \
    MC_BOT_VIEWER_PORT=${VIEWER_PORT} \
    NODE_OPTIONS="--enable-source-maps"

ENTRYPOINT ["node", "src/index.js"]
