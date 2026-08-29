## Why

FORGE's benchmark tooling will manufacture a plausible aggregate from records
that never measured the system, and nothing in the repository stops it.

`scripts/mc_plot_baseline.py`'s `summarize_snapshot` averages `total_reward`
over every record in a snapshot with no filter. Fed the committed random
baseline — thirty records of which twenty-nine ran a single step after the
environment reported it could not execute the step — it yields a mean of
-0.0767, rendered as `-0.077`. The capture script produced those records
because it marks an environment-reported failure as a truncation and proceeds
to the next episode, so a bot that cannot act is recorded identically to an
episode that ran to its budget.

The discriminating evidence is already in the artifacts and unused. Every
record carries a non-zero protocol-error count, and every record reports an
observation dimension of zero against a handshake that declares nine hundred
and twenty. The aggregator reads neither.

That number is not in the repository, and the accompanying prose is honest:
the report is marked partial, states the data is not statistically meaningful,
and counts episodes with a non-empty rollout in its own column. The gap is
that this honesty is a hand-written caveat rather than an enforced property,
and the documented render command writes its output over the file holding the
caveat.

No gate covers this. Each of the seven charter invariants names enforcement
over code; none governs evidence. The nine gates chained under `make verify`
never read `docs/results/`. The only job that touches baselines runs on manual
dispatch, asserts a record count rather than an episode count, and never
brings its compose stack up, so it cannot have run successfully.

## What Changes

- Record an environment-reported step failure as a distinct outcome rather
  than as a truncation, and abort immediately on the error codes that indicate
  a cross-language contract violation rather than a transient fault.
- Halt a capture after a pinned number of consecutive non-evidential episodes.
- Exclude non-evidential records from every aggregate, report the evidential
  and excluded counts, and refuse to emit a comparison below a pinned floor.
- Decide evidentiality from signals already present in committed artifacts, so
  the rule applies to the existing corpus and not only to future captures.
- Gate the results corpus on two decidable properties: a table row citing a
  committed snapshot must agree with it, and a snapshot with no evidential
  records must carry a declaration.
- Index the expected evidence set with per-file digests, so removing or
  silently editing a snapshot requires an index change in the same diff.
- Assert that the gate cannot be excluded from the suite by a marker filter.
- Re-derive the results table from the artifacts it cites, and declare the two
  existing snapshots in place.
- Land the gate advisory for one cycle before flipping it to blocking,
  following the promotion doctrine the security workflow already states.
- Correct two verified omissions in the charter's own lists of deliberately
  non-blocking jobs and advisory scanners.

## Capabilities

### New Capabilities

- `evidence-integrity`: a record that did not measure the system cannot enter
  an aggregate, and a published table cannot disagree with the artifact it
  cites.

### Modified Capabilities

- None. No `openspec/specs/` existed before this change; every requirement is
  ADDED.

## Impact

- `scripts/v05_manual_baseline.py` — the error-frame branch that sets
  `truncated`; the consecutive-failure halt; the contract-violation abort. The
  file has no test today.
- `scripts/mc_plot_baseline.py` — `summarize_snapshot`, the evidential
  predicate, the reported counts, and the refusal path through `main`.
- `tests/python/test_mc_plot_baseline_unit.py` — all six tests change; the
  shared fixture encodes the outlawed record shape and asserts a mean over it.
- `docs/results/v0.5-first-real-run.md` — the results table; its two artifact
  links are transposed, one row describes a run that was never committed, and
  one row states rewards whose snapshot is recorded as not preserved.
- `docs/results/v0.5-first-real-run-baseline.json`,
  `docs/results/v0.5-first-real-run-baseline-v2.json` — declared in place.
- `docs/architecture.md` — describes the thirty-record artifact as four
  episodes and documents the behaviour this change outlaws as intended.
- `.github/workflows/hf-model.yml` — hard-codes observation and action
  dimensions "per" the results report, so rewriting that report leaves a
  citation nothing checks.
- `docs/CHARTER.md` — Invariant 6's non-blocking job list and its
  advisory-scanner claim, both carrying verified omissions. No renumbering.
- New: `docs/results/INDEX.toml`; `tests/python/test_evidence_integrity.py`
  and `tests/python/test_v05_manual_baseline.py`, stdlib and pytest only,
  running under the existing `python-test` gate. No new CI dependency and no
  workflow change beyond the advisory marker removed at the end of the
  rollout.
