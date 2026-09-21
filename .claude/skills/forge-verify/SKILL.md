---
name: forge-verify
description: Run FORGE's pre-PR validation gate (Rust fmt/clippy/test, Python ruff/mypy/pytest, mc-bot, dashboard, and optionally coverage/cargo-deny/gitleaks, or the wasm-runtime + browser-E2E stack for changes touching crates/forge-wasm or web/) and report a clear per-category pass/fail summary. Use before opening or updating a PR, or when the user asks to "verify", "run the full test suite", or "check CI will pass" locally.
---

Run this repo's pre-PR validation sequence and report results per category
(Rust, Python, mc-bot, dashboard, and — for `verify-full` — coverage/deny/
gitleaks, or — for `verify-wasm` — the wasm-runtime and browser-E2E
layers), not just a final pass/fail. Every step here is a thin wrapper
around a target in the root `Makefile`, which is itself just the commands
documented in `CONTRIBUTING.md`; if this skill and those two files ever
disagree, they've drifted and the drift itself is worth flagging.

## Steps

1. Ask which mode if it isn't already clear from the request:
   - `make verify` (default) — fmt-check, clippy, cargo test, ruff, mypy,
     pytest, openspec-validate, mc-bot, dashboard. This is what CI's blocking jobs run.
   - `make verify-full` — everything in `verify`, plus `cargo tarpaulin`
     (85% floor), `cargo deny check`, and `gitleaks`. Slower; run before
     a PR that changes dependencies or when coverage regressions matter.
   - `verify-wasm` — `make verify` plus the two carve-outs below that
     structurally can't be part of it (`make wasm-test`, `make web-e2e`):
     the full wasm-specific gate stack for a PR touching
     `crates/forge-wasm`, `forge-core`, `forge-types`, `web/`, or
     `tests/web-e2e/`. Needs `wasm-pack` and a Chromium download (both
     network-fetched by the targets themselves). Use this instead of
     plain `verify` whenever the change is in that surface — `verify`
     alone would pass while leaving the wasm-runtime and browser-E2E
     layers, which have each caught real defects `clippy`/`cargo test`
     didn't (see PR #134's "why three verification layers"), unchecked.
   Skip asking if the user already said "full"/"with coverage" (→
   `verify-full`) or "wasm"/named `crates/forge-wasm`/`web/` specifically
   (→ `verify-wasm`).

2. Run `make -n <target>` first (dry-run) if there's any doubt about what
   will execute — it's free and prints the resolved command list without
   side effects.

3. Run the chosen target via Bash. `verify` and `verify-full` are each one
   Makefile target; `verify-wasm` is `make verify` followed by
   `make wasm-test && make web-e2e` — run as separate commands (not
   chained into one target) so a `wasm-test` failure is attributed clearly
   rather than reported as a generic `verify-wasm` failure. These take
   several minutes; do not interrupt early on a slow step.

4. If it fails, don't just report "verify failed" — identify which
   sub-target failed (the Makefile stops at the first failing prerequisite,
   so the last `make[1]: Entering directory` / command line before the
   error names it) and re-run *that one target alone*
   (e.g. `make lint`, `make py-test`) to get a focused error without
   re-running everything else.

5. Report a per-category table: Rust (fmt/clippy/test), Python (ruff/
   mypy/pytest), mc-bot, dashboard, (if `verify-full`) coverage/deny/
   gitleaks, and (if `verify-wasm`) wasm-runtime tests/browser E2E —
   pass/fail each, with the specific failing command and a short excerpt
   of the error for anything that failed. Don't claim something is green
   without having actually run it in this pass.

## Not covered by `make verify` / `make verify-full`

(`wasm-test` and `web-e2e` below are covered by the `verify-wasm` mode
above — this section is about what plain `verify`/`verify-full` skip.)

- **ONNX feature surface** (`onnx` / `onnx-reload` / `mc-live-bundled`):
  `make onnx-check` needs a real ONNX Runtime ≥1.23.2 shared library and
  `ORT_DYLIB_PATH` pointing at it — it isn't part of the default sequence
  because most local setups don't have that. Only run it when the change
  actually touches `forge-agent`'s ONNX code, `forge-mc-runner`'s
  `onnx-reload`/`mc-live*` paths, or their Cargo features; see the
  `onnx-features` job in `.github/workflows/ci.yml` for how to fetch the
  runtime if it's needed.
- **WASM runtime tests** (`make wasm-test`): runs the crate's
  `#[wasm_bindgen_test]`s under Node via `wasm-pack`, which has to be
  network-installed (`scripts/install_wasm_pack.sh`), so it isn't in the
  default sequence. `make wasm-check` (clippy on `wasm32-unknown-unknown`)
  **is** part of `verify` — it self-skips with an actionable message when
  the target is missing, though `rust-toolchain.toml` lists it so rustup
  users always have it. Run `wasm-test` when the change touches
  `crates/forge-wasm`, `forge-core`, `forge-types`, or `web/`; see the
  `wasm` job in `.github/workflows/ci.yml`.
- **WASM demo E2E** (`make web-e2e`): `node:test` unit coverage for
  `web/app.js`'s pure helpers plus Playwright driving the real demo page in
  Chromium against a fresh wasm-pack build — needs `wasm-pack` and a
  Chromium download, so it isn't in the default sequence either. Run it
  when the change touches `web/` or `tests/web-e2e/`; see the non-blocking
  `wasm-e2e` job in `.github/workflows/ci.yml`.
- **Markdown lint** (`npx --yes markdownlint-cli2 "**/*.md"`) isn't a
  Makefile target; run it directly if the change touches docs. (Hook
  self-tests *are* covered — `make hooks-test` runs as part of `verify`.)
- Opt-in/marker-gated Python tests (`lmstudio`, `e2e_long`,
  `minecraft_e2e`) are deliberately excluded from `py-test` — they need
  external services or long-running compose stacks. Only run them if the
  change specifically targets that surface.
