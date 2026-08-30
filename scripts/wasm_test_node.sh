#!/usr/bin/env bash
# Run crates/forge-wasm's tests inside a real wasm runtime via Node.
#
# WHY THIS WRAPPER EXISTS: `wasm-pack test --node` exits 0 when the crate
# contains no `#[wasm_bindgen_test]` functions. Its runner walks the compiled
# module's `__wbgt_*` export table -- emitted only by that macro -- and when it
# finds none it prints "no tests to run!" and succeeds. A plain
# `wasm-pack test --node` as a CI gate would therefore stay green forever if the
# wasm tests were ever renamed, moved into a private module, or `#[cfg]`-ed out.
#
# So: run it, tee the output, and fail on the vacuous-green marker. Plain
# `#[test]` functions are compiled for wasm32 but never executed by this runner,
# which is expected -- they are covered by `cargo test --workspace` on the host.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log="$(mktemp)"
trap 'rm -f "$log"' EXIT

# wasm-pack writes the harness output to stderr; capture both streams.
wasm-pack test --node "$REPO_ROOT/crates/forge-wasm" 2>&1 | tee "$log"

if grep -q "no tests to run!" "$log"; then
  echo "ERROR: wasm-pack found no #[wasm_bindgen_test] functions in forge-wasm." >&2
  echo "       It exits 0 in that case, so this gate would be vacuously green." >&2
  echo "       Check crates/forge-wasm/tests/ -- the tests are gated on" >&2
  echo "       #![cfg(target_arch = \"wasm32\")] and must use #[wasm_bindgen_test]." >&2
  exit 1
fi

if ! grep -qE "running [1-9][0-9]* test" "$log"; then
  echo "ERROR: wasm-pack output contained no 'running N tests' line with N > 0." >&2
  echo "       Refusing to report success without evidence that tests executed." >&2
  exit 1
fi
