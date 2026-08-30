#!/usr/bin/env bash
# Build crates/forge-wasm into web/pkg/ for the static in-browser demo.
#
# The single source for `make wasm`, the Playwright webServer in
# tests/web-e2e/, and the build step in gh-pages.yml / hf-space.yml -- so the
# E2E suite can never test a differently-built artifact than the one deployed.
#
# NOTE: wasm-pack resolves a RELATIVE --out-dir against the *crate* directory,
# not the invocation cwd. `--out-dir web/pkg` silently emits to
# crates/forge-wasm/web/pkg -- the bug this script exists to make unwritable.
# Always an absolute path.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${FORGE_WASM_OUT_DIR:-$REPO_ROOT/web/pkg}"

exec wasm-pack build "$REPO_ROOT/crates/forge-wasm" \
  --target web --out-dir "$OUT_DIR" --no-typescript "$@"
