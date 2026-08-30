<!--
Thanks for contributing to FORGE. Fill in the sections below. Keep the checklist
honest — an unchecked box with a one-line reason is better than a checked box that
isn't true. See CONTRIBUTING.md for the full gate list.
-->

## Summary

<!-- What does this change do, and why? Link any issue: Closes #NNN -->

## Scope of change

<!-- Which crates/surfaces are touched? Note any cross-language contract
(schema_id, trajectory/manifest, WS protocol) affected. -->

## How it was verified

<!-- The exact commands you ran locally, and their result. -->

- [ ] `cargo fmt --all --check`
- [ ] `cargo clippy --workspace --all-targets --features forge-cloud/gcs -- -D warnings`
- [ ] `cargo test --workspace --features forge-cloud/gcs`
- [ ] `make wasm-check` (if `crates/forge-wasm`, `forge-core`, `forge-types`, or `web/` changed)
- [ ] `cargo tarpaulin … --fail-under 85` (if Rust coverage-affecting)
- [ ] `cargo deny check` (if dependencies changed)
- [ ] Python: `ruff check` + `mypy --config-file pyproject.toml` + `pytest tests/python`
- [ ] mc-bot: `npm run typecheck && npm run lint && npm test`
- [ ] Docs updated (architecture.md / CHARTER.md / CHANGELOG.md as applicable)

## Invariant check (see docs/CHARTER.md)

<!-- Confirm the change preserves the relevant invariants, or explain the
Deliberate Exception if it doesn't. -->

- [ ] Determinism preserved (same seed + actions ⇒ identical state)
- [ ] No hard-coded values (constants flow through config/Default)
- [ ] Wire protocols versioned & backward-compatible (if a schema changed)
- [ ] No secrets/credentials committed
