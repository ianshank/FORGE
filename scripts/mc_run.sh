#!/usr/bin/env bash
# scripts/mc_run.sh — orchestration for the Minecraft RL stack.
#
# Brings up the Minecraft server + mc-bot + forge-mc-runner via the
# docker/compose.minecraft.yml stack. Idempotent: re-running while the
# stack is up will only refresh services whose images/configs changed.
#
# Usage:
#   scripts/mc_run.sh [--dry-run] [--build] [--detach] [--down] \
#                      [--env-file PATH] [--service NAME] \
#                      [--profile NAME] [--gpu]
#
# Flags:
#   --dry-run          Print the docker compose command(s) that would
#                      run and exit 0. Nothing else touches the host.
#   --build            Force a rebuild of the local images before up.
#   --detach           Run `up` in detached mode (default: foreground
#                      so logs stream to the terminal).
#   --down             Tear the stack down (`docker compose down`).
#   --env-file PATH    Override the env file
#                      (default: docker/compose.minecraft.env, falling
#                      back to compose.minecraft.env.example if the
#                      first does not exist).
#   --service NAME     Limit the action to a single service.
#   --profile NAME     Activate a compose profile (e.g. `self-play`
#                      brings up the v0.4 trainer + trainer-bootstrap
#                      services in addition to the base stack).
#   --gpu              Layer the GPU overlay file
#                      (docker/compose.minecraft.gpu.yml) on top of the
#                      base compose file. Requires nvidia-container-
#                      toolkit on the host and `docker compose` v2.20+.
#
# Every behaviour is driven by env vars or flags — no values are
# hard-coded inside this script.

set -euo pipefail

# ---------------------------------------------------------------------
# Defaults (all overridable via env or flags)
# ---------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-${REPO_ROOT}/docker/compose.minecraft.yml}"
GPU_OVERLAY_FILE="${GPU_OVERLAY_FILE:-${REPO_ROOT}/docker/compose.minecraft.gpu.yml}"
DEFAULT_ENV_FILE="${REPO_ROOT}/docker/compose.minecraft.env"
EXAMPLE_ENV_FILE="${REPO_ROOT}/docker/compose.minecraft.env.example"

DRY_RUN=0
BUILD=0
DETACH=0
DOWN=0
ENV_FILE=""
SERVICE=""
PROFILE=""
USE_GPU=0

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

log()  { printf '%s [mc_run] %s\n' "$(date -u +%FT%TZ)" "$*" >&2; }
die()  { log "ERROR: $*"; exit 1; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 \
    || die "required command not found: $1"
}

usage() {
  sed -n '2,30p' "$0"
}

# ---------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------

while (( $# > 0 )); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --build)   BUILD=1; shift ;;
    --detach)  DETACH=1; shift ;;
    --down)    DOWN=1; shift ;;
    --env-file)
      [[ -n "${2-}" ]] || die "--env-file requires a path"
      ENV_FILE="$2"; shift 2 ;;
    --service)
      [[ -n "${2-}" ]] || die "--service requires a name"
      SERVICE="$2"; shift 2 ;;
    --profile)
      [[ -n "${2-}" ]] || die "--profile requires a name"
      PROFILE="$2"; shift 2 ;;
    --gpu) USE_GPU=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown flag: $1" ;;
  esac
done

# ---------------------------------------------------------------------
# Resolve env-file (default → fallback to example with warning).
# ---------------------------------------------------------------------

if [[ -z "${ENV_FILE}" ]]; then
  if [[ -f "${DEFAULT_ENV_FILE}" ]]; then
    ENV_FILE="${DEFAULT_ENV_FILE}"
  elif [[ -f "${EXAMPLE_ENV_FILE}" ]]; then
    log "WARN: ${DEFAULT_ENV_FILE} not found; using example at ${EXAMPLE_ENV_FILE}."
    log "WARN: copy it to compose.minecraft.env and set MC_EULA=TRUE before running for real."
    ENV_FILE="${EXAMPLE_ENV_FILE}"
  else
    die "no env file found (looked at ${DEFAULT_ENV_FILE} and ${EXAMPLE_ENV_FILE})"
  fi
fi

[[ -f "${ENV_FILE}" ]]      || die "env file does not exist: ${ENV_FILE}"
[[ -f "${COMPOSE_FILE}" ]]  || die "compose file does not exist: ${COMPOSE_FILE}"

# ---------------------------------------------------------------------
# Build the docker compose argv
# ---------------------------------------------------------------------

compose_args=(
  "compose"
  "--env-file" "${ENV_FILE}"
  "-f" "${COMPOSE_FILE}"
)

# Layer the GPU overlay file when --gpu is set. Overlay must come
# AFTER the base file so its `deploy.resources` block wins.
if (( USE_GPU )); then
  [[ -f "${GPU_OVERLAY_FILE}" ]] || die "GPU overlay file missing: ${GPU_OVERLAY_FILE}"
  compose_args+=("-f" "${GPU_OVERLAY_FILE}")
fi

# Activate the named compose profile if --profile was passed.
# Required when bringing up the v0.4 self-play services (trainer +
# trainer-bootstrap live under `profiles: ["self-play"]`).
if [[ -n "${PROFILE}" ]]; then
  compose_args+=("--profile" "${PROFILE}")
fi

if (( DOWN )); then
  cmd=("docker" "${compose_args[@]}" "down" "--remove-orphans")
else
  action_args=()
  (( BUILD ))  && action_args+=("--build")
  (( DETACH )) && action_args+=("-d")
  [[ -n "${SERVICE}" ]] && action_args+=("${SERVICE}")
  cmd=("docker" "${compose_args[@]}" "up" "${action_args[@]}")
fi

# ---------------------------------------------------------------------
# Execute (or dry-run)
# ---------------------------------------------------------------------

if (( DRY_RUN )); then
  printf 'DRY-RUN: '
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

require_cmd docker

# Trap so Ctrl-C in foreground mode brings the stack down cleanly.
if ! (( DETACH )) && ! (( DOWN )); then
  trap 'log "SIGINT received; running compose down"; docker "${compose_args[@]}" down --remove-orphans' INT TERM
fi

log "running: ${cmd[*]}"
"${cmd[@]}"
