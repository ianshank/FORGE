# syntax=docker/dockerfile:1.6
#
# Container image for `mc-bot/` — the Node.js bridge that joins a
# Minecraft server via mineflayer and exposes a WebSocket the Rust
# `forge-env-mc` client consumes.
#
# Image conventions:
# - Multi-arch via BuildKit: builds on amd64 + arm64 with `--platform`.
# - Rootless: drops to a non-root `node` user.
# - Reproducible: locks Node via the official slim image and installs
#   from the checked-in package-lock.json with `npm ci`. No global
#   tools, no curl-pipe-bash.
# - Compiled: `mc-bot/src` is TypeScript only. The builder stage runs
#   `npm run build` (tsc -p tsconfig.build.json) and the runtime stage
#   executes the emitted JavaScript, so `tsx` — a devDependency that
#   `npm ci --omit=dev` strips — is never needed at runtime.
#
# The build context is `mc-bot/` (matching docker/compose.minecraft.yml).
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

# --- deps stage: production-only dependency tree ------------------------
# Kept separate from the compile stage so the runtime image never carries
# typescript/tsx/biome, and so BuildKit can resolve both trees in parallel.
FROM node:${NODE_IMAGE_TAG} AS deps

WORKDIR /app

# Copy manifests first so dependency installation caches separately
# from source-code changes.
COPY package.json package-lock.json ./

# mc-bot ships a lockfile, so `npm ci` is always the right call: it is
# reproducible and fails loudly if package.json and the lock drift apart.
RUN npm ci --omit=dev

# --- builder stage: compile TypeScript -> dist/ -------------------------
FROM node:${NODE_IMAGE_TAG} AS builder

WORKDIR /app

COPY package.json package-lock.json ./

# Full install (devDependencies included) — the compiler lives there.
RUN npm ci

COPY tsconfig.json tsconfig.build.json ./
COPY src ./src

# Emits ./dist/index.js and friends. tsconfig.build.json pins rootDir to
# ./src so the entry point lands at /app/dist/index.js: mc-bot derives its
# DEFAULT_CONFIG_DIR as `<module dir>/../../configs/minecraft`, which must
# resolve to /configs/minecraft — where compose mounts the config tree.
RUN npm run build

# --- runtime stage -----------------------------------------------------
FROM node:${NODE_IMAGE_TAG}

ARG WS_PORT
ARG VIEWER_PORT

LABEL org.opencontainers.image.title="forge-mc-bot" \
      org.opencontainers.image.description="Mineflayer bridge for FORGE Minecraft RL" \
      org.opencontainers.image.source="https://github.com/ianshank/FORGE"

WORKDIR /app

# Bring the production dependency tree and the compiled output across.
# `--chown=node:node` on each COPY avoids a separate `chown -R` layer
# that would walk every file in node_modules at build time (slow once
# the dep tree grows). Matches BuildKit's incremental-layer model.
COPY --from=deps --chown=node:node /app/node_modules ./node_modules
COPY --from=builder --chown=node:node /app/dist ./dist

COPY --chown=node:node README.md ./README.md
COPY --chown=node:node package.json ./package.json

# Run as non-root by default (the `node` user is created in the
# upstream image). No separate `chown -R` needed because every COPY
# above already wrote the files as `node:node`.
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

# Liveness: the control WebSocket must be accepting TCP connections.
# `net.connect` is asynchronous, so the exit is bound explicitly to the
# 'connect'/'error'/timeout callbacks rather than falling off the end of
# the script (which would race the connect callback). The port is read
# from MC_BOT_WS_PORT above — no literal duplicated here.
# docker/compose.minecraft.yml declares an equivalent healthcheck and
# overrides this one; both must stay semantically identical because the
# `runner` service gates on `mc-bot: service_healthy`.
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD ["node", "-e", "var p=Number(process.env.MC_BOT_WS_PORT);if(!Number.isInteger(p)||p<=0){process.exit(1);}var s=require('node:net').connect(p,'127.0.0.1');s.on('connect',function(){s.destroy();process.exit(0);});s.on('error',function(){process.exit(1);});s.setTimeout(3000,function(){s.destroy();process.exit(1);});"]

ENTRYPOINT ["node", "dist/index.js"]
