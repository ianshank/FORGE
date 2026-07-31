## Why

`docs/CHARTER.md` makes falsifiable claims: each invariant names the file, Cargo
feature, or CI job enforcing it. A repo-wide scan found those citations had
rotted. Two docs contradicted a charter Deliberate Exception, the document the
charter designates as the single source of truth for the crate list omitted a
third of the workspace, a crate deleted in `e5eca3d` survived in two diagrams,
and one cited enforcement mechanism was never built. A stale enforcement
citation is worse than none: it asserts a guarantee is defended when it is not.

## What Changes

- Correct `docs/CHARTER.md` Invariants 2, 3 and 6 so every *Enforced by*
  citation names something that exists and would fail if the invariant broke.
- Restate the `schema_id` contract as three-language (Rust ↔ JS ↔ Python) with a
  per-value pin table, and name the third wire-version constant.
- Extend `docs/architecture.md` from 11 to all 26 workspace crates, so its role
  as declared source of truth is true.
- Remove the deleted `forge-procgen` from the two ASCII diagrams that survived
  the `e5eca3d` sweep, and drop the stale `forge-scenario` path citation.
- Correct the `mc-live` feature definition in `docs/architecture.md` and
  `Agent.md` to match `Cargo.toml` and Deliberate Exception 2.
- **BREAKING (build surface):** delete the unimplemented `live-test-stub` Cargo
  feature and every doc reference to it.
- Ratify the two Permanent Non-Goals, removing the pending-confirmation hedge.
- Add an automated alignment guard so this class of drift fails CI.

## Capabilities

### New Capabilities

- `charter-alignment`: machine-checkable consistency between `docs/CHARTER.md`,
  the docs it delegates to, and the code and CI it cites.

### Modified Capabilities

- None. No `openspec/specs/` existed before this change; every requirement is
  ADDED.

## Impact

- `docs/CHARTER.md` — Invariants 2, 3, 6; Permanent Non-Goals; deferred list.
- `docs/architecture.md` — container table, two ASCII diagrams, feature matrix,
  duplicate heading.
- `Agent.md`, `ANTIGRAVITY.md`, `README.md`, `CLAUDE.md`, `CHANGELOG.md`,
  `docs/next_steps.md`, `benchmarks/baselines/README.md`,
  `docs/results/replay-compression-sweep.md`.
- `crates/forge-mc-runner/Cargo.toml` — one feature removed;
  `crates/forge-mc-runner/src/live.rs` — one stale comment.
- `crates/forge-env-mc/src/*.rs`, `mc-bot/src/*.ts`,
  `tests/python/training/test_muzero_mc_schema_id.py` — doc-comment paths only.
- New: `tests/python/test_charter_alignment.py`, running under the existing
  `python-test` gate. No new CI dependency.
