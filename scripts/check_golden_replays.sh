#!/usr/bin/env bash
# Bit-identity CompactReplay goldens plus the full forge-replay suite.
#
# PR CI already runs `cargo test --workspace`, which includes
# `crates/forge-replay/tests/golden_replay.rs`. This script is the
# workflow_dispatch / scheduled path: it re-runs the golden gate, then the
# crate suite, and prints the flip-log remedy on mismatch.
#
# Usage:
#   scripts/check_golden_replays.sh
#   UPDATE_GOLDEN_REPLAYS=1 scripts/check_golden_replays.sh   # rewrite goldens

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FLIP_LOG="docs/results/replay_flip_log.md"
GOLDEN_TEST=(cargo test -p forge-replay --test golden_replay -- --nocapture)

if ! "${GOLDEN_TEST[@]}"; then
  cat <<EOF

CompactReplay golden mismatch.
If the format change is intentional:
  1. UPDATE_GOLDEN_REPLAYS=1 cargo test -p forge-replay --test golden_replay
  2. Append a row to ${FLIP_LOG} with the new config_hash and reason.
If it is not intentional, do not regenerate — fix the replay path.

EOF
  exit 1
fi

cargo test -p forge-replay
echo "golden CompactReplay bit-identity + forge-replay suite passed"
