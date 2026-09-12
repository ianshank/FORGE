#!/usr/bin/env bash
# scripts/mc_evidential_capture.sh — Docker-host trained-vs-random capture.
#
# This environment cannot produce live Paper/Minecraft episodes. The
# script is the operator runbook from docs/results/v0.5-loop-survival.md
# in executable form. CI covers `--dry-run` only.
#
# It will NEVER write evidential_episodes >= 3 without a real capture,
# and it will NEVER fill docs/results/v0.5-trained-vs-random.md unless
# scripts/mc_plot_baseline.py accepts both snapshots (floor = 3).
#
# Usage:
#   scripts/mc_evidential_capture.sh --dry-run
#   scripts/mc_evidential_capture.sh --episodes 8
#
# Flags:
#   --dry-run       Print every step and exit 0.
#   --episodes N    Episodes per variant (default 3, the plotter floor).
#   --skip-trained  Random variant only.
#   --random-out PATH
#   --trained-out PATH
#   --report PATH   Markdown report path for mc_plot_baseline.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SELF_PLAY="${SCRIPT_DIR}/mc_self_play.sh"
CAPTURE_MOD=(python -m forge.training.muzero_mc.cli capture-baseline)
PLOTTER="${SCRIPT_DIR}/mc_plot_baseline.py"

DRY_RUN=0
SKIP_TRAINED=0
EPISODES="${EVIDENTIAL_EPISODES:-3}"
RANDOM_OUT="${REPO_ROOT}/docs/results/v0.5-random-evidential.json"
TRAINED_OUT="${REPO_ROOT}/docs/results/v0.5-trained-evidential.json"
REPORT_OUT="${REPO_ROOT}/docs/results/v0.5-trained-vs-random.md"
FLOOR="${MIN_EVIDENTIAL_EPISODES_FOR_COMPARISON:-3}"

log() { printf '%s [mc_evidential_capture] %s\n' "$(date -u +%FT%TZ)" "$*" >&2; }

run_or_echo() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf 'DRY-RUN:' >&2
    printf ' %q' "$@" >&2
    printf '\n' >&2
    return 0
  fi
  "$@"
}

usage() {
  sed -n '2,24p' "$0" | sed 's/^# \?//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --skip-trained) SKIP_TRAINED=1; shift ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --random-out) RANDOM_OUT="$2"; shift 2 ;;
    --trained-out) TRAINED_OUT="$2"; shift 2 ;;
    --report) REPORT_OUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) log "unknown flag: $1"; usage; exit 2 ;;
  esac
done

if [[ "${EPISODES}" -lt "${FLOOR}" ]]; then
  log "episodes=${EPISODES} is below the evidential floor ${FLOOR}"
  exit 2
fi

if [[ "${DRY_RUN}" -eq 0 ]]; then
  if ! command -v docker >/dev/null 2>&1; then
    log "docker is not on PATH. Live capture is host-bound. See docs/results/v0.5-loop-survival.md"
    exit 3
  fi
  if ! docker info >/dev/null 2>&1; then
    log "docker daemon is not reachable. Refusing to invent evidential JSON."
    exit 3
  fi
else
  log "dry-run: skipping docker preflight"
fi

log "step 1/4 random stack (--baseline-only)"
run_or_echo bash "${SELF_PLAY}" --baseline-only --detach
log "step 2/4 capture-baseline random episodes=${EPISODES}"
run_or_echo "${CAPTURE_MOD[@]}" \
  --variant random \
  --episodes "${EPISODES}" \
  --out "${RANDOM_OUT}"

if [[ "${SKIP_TRAINED}" -eq 0 ]]; then
  log "step 3/4 trained stack"
  run_or_echo bash "${SELF_PLAY}" --detach
  log "step 4/4 capture-baseline trained episodes=${EPISODES}"
  run_or_echo "${CAPTURE_MOD[@]}" \
    --variant trained \
    --episodes "${EPISODES}" \
    --out "${TRAINED_OUT}"
  log "plotter (refuses unless both variants have >= ${FLOOR} evidential episodes)"
  run_or_echo python "${PLOTTER}" \
    --random "${RANDOM_OUT}" \
    --trained "${TRAINED_OUT}" \
    --out "${REPORT_OUT}"
else
  log "skipping trained capture and comparison report"
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  log "dry-run complete; no snapshots written"
  exit 0
fi

log "operator: commit the JSON snapshots + INDEX.toml sha256 only after the plotter succeeds"
exit 0
