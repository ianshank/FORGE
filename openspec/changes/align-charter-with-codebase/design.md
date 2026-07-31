## Context

`docs/CHARTER.md` is ~260 lines of deliberately falsifiable claims. It was added
2026-07-09, after most of the code it describes, and inherited framing from
`README.md`. The drift found by the audit is concentrated in citations, not in
the invariants themselves — all seven remain accurate as principles, and
Invariants 1, 4, 5 and 7 verified clean end to end.

## Goals / Non-Goals

**Goals:**

- Every *Enforced by* citation names something that exists and would fail.
- The document the charter designates as source of truth actually is one.
- This class of drift is caught mechanically thereafter.

**Non-Goals:**

- Weakening or renegotiating any invariant.
- Rewriting `docs/architecture.md`'s in-flight-PR narrative voice.
- Running benchmarks to regenerate the missing throughput baseline.
- Flipping `cargo-deny` from advisory to blocking — a separate decision.

## Decisions

### Decision: Guard lives in Python under the existing `python-test` gate
Follows `tests/python/test_check_zero_alloc.py`, which already validates a
repo-level artifact from pytest. `python-test` is a job the charter itself
names, so the guard is covered by the invariant it protects. `ci.yml` is parsed
with a regex rather than by adding PyYAML — the CI venv installs
`maturin pytest numpy gymnasium httpx jsonschema` and nothing else.

### Decision: Respect the charter's delegation instead of overriding it
An earlier draft would have asserted that every workspace crate is named in
`docs/CHARTER.md`. That was rejected on review: the charter's table header reads
"Representative crates" and the text explicitly delegates the full list to
`docs/architecture.md`. CI-mandating exhaustiveness would convert a deliberately
illustrative table into the thing it declines to be, and would churn the charter
on every crate add or remove — relocating the rot rather than removing it.

The guard instead checks the delegation target for completeness, and checks both
documents for names that are no longer members. Both assertions failed on the
pre-fix tree — six crates and `forge-procgen` respectively — so the guard has
immediate bite rather than being purely forward-looking.

### Decision: Require a `cfg` site only for features cited as enforcement
A blanket "every feature must have a `cfg` site" rule would false-positive on
aggregates (`mc-live-bundled`) and dependency switches (`dhat-heap`,
`onnx-bundled`), which legitimately have none. The narrower rule — a feature the
charter names *as enforcement* must be gated on in code — is what catches the
`live-test-stub` class without punishing legitimate aggregates. A separate,
weaker reachability check covers orphaned flags generally.

### Decision: Delete `live-test-stub` rather than implement it
Nothing selected it: no `cfg` site, no CI job, no test. The v0.4 self-improvement
smoke test its manifest comment named contains zero references to it.
Implementing a `MockMinecraftEnv` harness is real work with no current caller,
and Invariant 3's remaining citations (`--dry-run` with `StubEnv`,
`mc_env_mock.rs`, the generic `Runner<E, M>`, the `forge-mc-runner-bin` smoke)
already cover the exercisable-without-hardware guarantee.

### Decision: Leave `README.md`'s deferred-items list untouched
The audit initially flagged "Complex learned block embeddings" as stale, since a
learned `nn.Embedding(36, 8)` ships enabled by default. Review refuted this:
commit `a2dcfaa` added both the `use_raw_block_id` default and that README line,
in one maintainer commit titled "…and doc alignment". The qualifier "complex" is
doing deliberate work. Only the charter's copy — which had dropped the
`(T3 Phase 2 candidate)` qualifier — needed adjusting.

## Risks / Trade-offs

- [Split-name detection could false-positive on prose] → it only reconstructs a
  name when `forge-` is followed by a non-identifier character and the next line
  carries an identifier at the same column, and the result is checked against
  the workspace-member set plus discovered CI job names.
- [Deleting a public Cargo feature is a build-surface change] → the feature was
  unimplemented, so no build that worked before can break. Recorded in
  `CHANGELOG.md` regardless.
- [The guard encodes assumptions about doc structure] → each assertion fails
  loudly with the remedy in the message, and parser breakage is caught by
  non-empty assertions on the extracted sets.

## Open Questions

- Whether to adopt `openspec/` permanently alongside `docs/next_steps.md`, which
  `docs/CHARTER.md` names as the home for working tasks. This change ships both,
  cross-referenced; consolidating is a follow-up decision.
