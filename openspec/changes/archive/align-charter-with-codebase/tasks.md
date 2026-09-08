## 1. Adopt OpenSpec artifacts

- [x] 1.1 Create `openspec/changes/align-charter-with-codebase/` with
      `proposal.md`, `design.md`, `tasks.md`, and
      `specs/charter-alignment/spec.md`
- [x] 1.2 Cross-reference `docs/next_steps.md` so the two task homes agree

## 2. Correct docs/CHARTER.md

- [x] 2.1 Invariant 3: drop `live-test-stub` from the *Enforced by* list
- [x] 2.2 Invariant 3: point `Runner<E, M>` at `crates/forge-mc-runner/src/runner.rs`,
      keeping `live.rs` named as the injection site
- [x] 2.3 Invariant 2: restate `schema_id` as Rust ↔ JS ↔ Python with a pin table
- [x] 2.4 Invariant 2: add the `protocol.rs` paired tests and the third wire
      version constant `SCHEMA_VERSION`
- [x] 2.5 Invariant 6: recategorise `deny.toml`; name `security.yml` and the
      non-blocking jobs the gate list deliberately omits
- [x] 2.6 Ratify Permanent Non-Goals; base the hard-coding non-goal on
      `docs/hardcoded-values-audit.md`, not on a non-existent CI check
- [x] 2.7 Qualify the block-embeddings deferred bullet for parity with the
      README. Do **not** edit the README — it was correct as written

## 3. Repair docs/architecture.md as declared source of truth

- [x] 3.1 Extend the Container Descriptions table to all 26 workspace members
- [x] 3.2 Remove the `forge-`/`procgen` box from the C4 container diagram
- [x] 3.3 Remove `procgen` from the §4.2 dependency graph
- [x] 3.4 Correct the feature matrix to `mc-live = ["dep:forge-env-mc"]`
- [x] 3.5 Replace the `live-test-stub` feature-matrix entry with `mc-live-bundled`
- [x] 3.6 Remove the stray duplicate `### 3.2 forge-worldgen` heading

## 4. Reconcile the remaining docs

- [x] 4.1 `Agent.md` — "8 crates" → 26; correct the `mc-live` row; delete the
      `live-test-stub` build row
- [x] 4.2 `ANTIGRAVITY.md` — 80% → 85% coverage target
- [x] 4.3 Convert Windows-absolute `file:///c:/…` links to repo-relative
- [x] 4.4 `README.md` — "23 Rust crates" → 26
- [x] 4.5 `CLAUDE.md` — remove the `live-test-stub` build line
- [x] 4.6 `docs/next_steps.md` — retitle the row citing the deleted
      `crates/forge-scenario/src/config.rs`
- [x] 4.7 `benchmarks/baselines/README.md` — record the uncommitted
      `multi_agent_scaling.json` coverage gap
- [x] 4.8 `CHANGELOG.md` — `[Unreleased]` entry

## 5. Remove dead code and stale paths

- [x] 5.1 Delete the `live-test-stub` feature from
      `crates/forge-mc-runner/Cargo.toml`
- [x] 5.2 Update the stale "live-stub smoke test" comment in `live.rs`
- [x] 5.3 Retarget all 20 stale `mc-bot/**/*.js` doc-comment paths to `.ts`

## 6. Add the alignment guard

- [x] 6.1 Create `tests/python/test_charter_alignment.py` — stdlib + pytest only
- [x] 6.2 Failure messages state the remedy, not just the mismatch
- [x] 6.3 Confirm the guard fails on the pre-fix tree and passes after
