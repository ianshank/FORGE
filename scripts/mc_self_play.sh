#!/usr/bin/env bash
# scripts/mc_self_play.sh — v0.4 self-improving Minecraft RL stack.
#
# One-command orchestration of the full self-play loop:
#
#   1. Preflight: verify `docker compose` >= v2.20 (GPU overlay needs it).
#   2. Compute the canonical `schema_id` from the shipped action_map +
#      rewards configs via the `trainer-bootstrap` one-shot container
#      (operator host does NOT need torch installed).
#   3. Export `FORGE_MC_SCHEMA_ID` so the runner picks it up via the
#      env-var ladder added in T3 (RunnerConfig::with_env_var_overrides).
#   4. If `${MODELS_DIR}/model_manifest.json` is missing, run the
#      bootstrap one-shot to seed `models/v00000001/`.
#   5. Bring the full stack up via `mc_run.sh --profile self-play
#      [--gpu] [--detach]`.
#   6. SIGINT / EXIT trap → `mc_run.sh --down`.
#
# Usage:
#   scripts/mc_self_play.sh [--dry-run] [--gpu] [--detach] [--down] \
#                            [--env-file PATH]
#
# Flags:
#   --dry-run          Print every step's command + exit 0; nothing
#                      touches the host. Used by the test in
#                      tests/python/integration/test_mc_self_play_unit.py.
#   --gpu              Layer compose.minecraft.gpu.yml on top of the
#                      base file. Trainer picks `cuda`.
#   --detach           Run the final `compose up` in detached mode.
#                      Default is foreground (logs stream to TTY,
#                      Ctrl-C tears down).
#   --down             Tear the stack down only; skip steps 1-5.
#   --env-file PATH    Override the env file. Forwarded to mc_run.sh +
#                      the compose-run-bootstrap invocations.
#
# Every behaviour is driven by env vars or flags — no values are
# hard-coded inside this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-${REPO_ROOT}/docker/compose.minecraft.yml}"
GPU_OVERLAY_FILE="${GPU_OVERLAY_FILE:-${REPO_ROOT}/docker/compose.minecraft.gpu.yml}"
DEFAULT_ENV_FILE="${REPO_ROOT}/docker/compose.minecraft.env"
EXAMPLE_ENV_FILE="${REPO_ROOT}/docker/compose.minecraft.env.example"
MC_RUN_SH="${SCRIPT_DIR}/mc_run.sh"

DRY_RUN=0
USE_GPU=0
DETACH=0
DOWN=0
ENV_FILE=""

# Compose v2 minimum version required for `deploy.resources.reservations.devices`.
COMPOSE_MIN_VERSION="${COMPOSE_MIN_VERSION:-2.20}"

# Action-map + rewards paths inside the trainer-bootstrap container
# (the compose mount maps ../configs/minecraft → /app/configs by
# default). These are paths INSIDE the container, not host paths.
ACTION_MAP_IN_CONTAINER="${ACTION_MAP_IN_CONTAINER:-/app/configs/action_map.toml}"
REWARDS_IN_CONTAINER="${REWARDS_IN_CONTAINER:-/app/configs/rewards.toml}"
MANIFEST_IN_CONTAINER="${MANIFEST_IN_CONTAINER:-/app/models/model_manifest.json}"

# Bootstrap dimensions. Defaults track the shipped env.toml /
# observation.js values from PR #57. Operators override via env.
OBS_DIM="${OBS_DIM:-31}"
ACTION_DIM="${ACTION_DIM:-12}"

log()   { printf '%s [mc_self_play] %s\n' "$(date -u +%FT%TZ)" "$*" >&2; }
die()   { log "ERROR: $*"; exit 1; }
usage() { sed -n '2,40p' "$0"; }

# Run a command or print it in dry-run mode. The dry-run trace goes
# to STDERR so callers capturing stdout (e.g. command substitution
# around `capture_or_echo`) don't accidentally swallow the trace.
run_or_echo() {
  if (( DRY_RUN )); then
    {
      printf 'DRY-RUN: '
      printf '%q ' "$@"
      printf '\n'
    } >&2
  else
    "$@"
  fi
}

# Capture a command's stdout. In dry-run mode the trace goes to
# STDERR and stdout is a synthetic 64-hex placeholder so downstream
# steps see a well-formed schema_id.
capture_or_echo() {
  if (( DRY_RUN )); then
    {
      printf 'DRY-RUN: '
      printf '%q ' "$@"
      printf '\n'
    } >&2
    # Synthetic 64-hex placeholder so subsequent dry-run steps see a
    # well-formed schema_id.
    printf 'deadbeef%s\n' "$(printf '%.0s0' {1..56})"
  else
    "$@"
  fi
}

# ---------- arg parsing ----------
while (( $# > 0 )); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --gpu)     USE_GPU=1; shift ;;
    --detach)  DETACH=1; shift ;;
    --down)    DOWN=1; shift ;;
    --env-file)
      [[ -n "${2-}" ]] || die "--env-file requires a path"
      ENV_FILE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown flag: $1" ;;
  esac
done

# ---------- env file resolution ----------
if [[ -z "${ENV_FILE}" ]]; then
  if [[ -f "${DEFAULT_ENV_FILE}" ]]; then
    ENV_FILE="${DEFAULT_ENV_FILE}"
  elif [[ -f "${EXAMPLE_ENV_FILE}" ]]; then
    log "WARN: ${DEFAULT_ENV_FILE} not found; falling back to ${EXAMPLE_ENV_FILE}"
    log "WARN: copy it to compose.minecraft.env and set MC_EULA=TRUE for real runs"
    ENV_FILE="${EXAMPLE_ENV_FILE}"
  else
    die "no env file found"
  fi
fi

# ---------- down-only short-circuit ----------
if (( DOWN )); then
  log "tearing the stack down"
  down_args=("--down" "--profile" "self-play" "--env-file" "${ENV_FILE}")
  (( USE_GPU )) && down_args+=("--gpu")
  run_or_echo "${MC_RUN_SH}" "${down_args[@]}"
  exit 0
fi

# ---------- step 1: preflight ----------
log "preflight: checking docker compose version >= ${COMPOSE_MIN_VERSION}"
if (( DRY_RUN )); then
  log "DRY-RUN: skipping `docker compose version` check"
else
  command -v docker >/dev/null 2>&1 || die "docker not on PATH"
  docker compose version >/dev/null 2>&1 \
    || die "docker compose v2 not installed (need >= ${COMPOSE_MIN_VERSION})"
fi

# ---------- step 2: compute schema_id ----------
log "computing schema_id via trainer-bootstrap one-shot"
compose_args=(
  "compose"
  "--env-file" "${ENV_FILE}"
  "-f" "${COMPOSE_FILE}"
)
(( USE_GPU )) && compose_args+=("-f" "${GPU_OVERLAY_FILE}")
compose_args+=("--profile" "self-play")

SCHEMA_ID="$(capture_or_echo \
  docker "${compose_args[@]}" run --rm trainer-bootstrap \
    compute-schema-id \
    --action-map "${ACTION_MAP_IN_CONTAINER}" \
    --rewards "${REWARDS_IN_CONTAINER}" \
    --quiet | tr -d '[:space:]')"

# Validate the captured schema_id (skip in dry-run since it's
# synthetic).
if (( ! DRY_RUN )); then
  [[ ${#SCHEMA_ID} -eq 64 ]] \
    || die "compute-schema-id returned non-64-hex output: ${SCHEMA_ID}"
fi
log "schema_id=${SCHEMA_ID}"
export FORGE_MC_SCHEMA_ID="${SCHEMA_ID}"

# ---------- step 3: bootstrap initial manifest if missing ----------
log "checking for existing manifest at ${MANIFEST_IN_CONTAINER}"
# Use a containerised `test -f` so we don't need to map host paths.
if (( DRY_RUN )); then
  log "DRY-RUN: would check ${MANIFEST_IN_CONTAINER}; assuming missing"
  manifest_present=0
else
  if docker "${compose_args[@]}" run --rm --entrypoint /usr/bin/test \
       trainer-bootstrap -f "${MANIFEST_IN_CONTAINER}" >/dev/null 2>&1; then
    manifest_present=1
  else
    manifest_present=0
  fi
fi

if (( manifest_present == 0 )); then
  log "no manifest at ${MANIFEST_IN_CONTAINER}; running bootstrap one-shot"
  run_or_echo docker "${compose_args[@]}" run --rm trainer-bootstrap \
    bootstrap \
    --schema-id "${SCHEMA_ID}" \
    --obs-dim "${OBS_DIM}" \
    --action-dim "${ACTION_DIM}" \
    --out /app/models
else
  log "manifest already present; skipping bootstrap"
fi

# ---------- step 4: install SIGINT trap (foreground only) ----------
# Foreground path: install a trap that runs `mc_run.sh --down
# --profile self-play [--gpu]` on SIGINT / EXIT. This is belt-and-
# braces on top of `mc_run.sh`'s own foreground trap — if the
# operator hits Ctrl-C between `mc_run.sh up` returning and this
# script exiting, we still tear down cleanly.
#
# Detach path: NO trap. `mc_run.sh --detach` returns immediately;
# the operator runs `mc_self_play.sh --down` separately to tear
# down. The earlier docstring promised a trap in the detach path
# too — that was wrong (would tear down immediately after up).
if ! (( DETACH )); then
  down_trap_args=("--down" "--profile" "self-play" "--env-file" "${ENV_FILE}")
  (( USE_GPU )) && down_trap_args+=("--gpu")
  # shellcheck disable=SC2064 — variable expansion at trap-set time
  # is intentional so the trap command captures the resolved args.
  trap "$(printf '%q ' "${MC_RUN_SH}" "${down_trap_args[@]}")" INT TERM
fi

# ---------- step 5: bring the stack up via mc_run.sh ----------
log "starting the self-play stack"
up_args=("--profile" "self-play" "--env-file" "${ENV_FILE}")
(( USE_GPU )) && up_args+=("--gpu")
(( DETACH )) && up_args+=("--detach")
# Forward FORGE_MC_SCHEMA_ID into the child shell so `mc_run.sh`'s
# compose invocation picks it up via env-file interpolation.
run_or_echo env "FORGE_MC_SCHEMA_ID=${SCHEMA_ID}" "${MC_RUN_SH}" "${up_args[@]}"

log "mc_self_play complete"
