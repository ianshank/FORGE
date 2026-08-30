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

# wasm-pack builds one wasm binary per test target. The lib target (src/lib.rs)
# holds this crate's 22 plain `#[test]` unit tests, which the wasm harness never
# executes -- it collects only `__wbgt_*` exports, which only
# `#[wasm_bindgen_test]` emits -- so "no tests to run!" from *that* binary is
# expected and correct. What must never happen is every binary reporting
# nothing, which is the state the crate was in before tests/wasm_bindings.rs
# existed, and which wasm-pack reports with exit code 0.
if ! grep -qE "^running [1-9][0-9]* test" "$log"; then
  echo "ERROR: no wasm test binary reported 'running N tests' with N > 0." >&2
  echo "       wasm-pack exits 0 in that case, so this gate would be" >&2
  echo "       vacuously green. Check crates/forge-wasm/tests/: the tests are" >&2
  echo "       gated on #![cfg(target_arch = \"wasm32\")] and must be marked" >&2
  echo "       #[wasm_bindgen_test] -- a plain #[test] emits no __wbgt_ export" >&2
  echo "       and is invisible to this runner." >&2
  exit 1
fi
