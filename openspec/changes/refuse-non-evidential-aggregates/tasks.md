## 1. Adopt OpenSpec artifacts

- [x] 1.1 Create `openspec/changes/refuse-non-evidential-aggregates/` with
      `proposal.md`, `design.md`, `tasks.md`, and
      `specs/evidence-integrity/spec.md`
- [x] 1.2 Cross-reference `docs/next_steps.md` so the two task homes agree

## 2. Fix the aggregator

- [x] 2.1 Add the evidential predicate: a record is evidential when its
      protocol-error count is zero, its observation dimension equals the
      handshake's, and — for any record that is not a confirmed natural
      terminal, truncated or ambiguous alike — its step count meets the
      pinned floor
- [x] 2.2 Filter `summarize_snapshot` to evidential records and carry the
      evidential and excluded counts on the summary
- [x] 2.3 Refuse to render a comparison below the pinned evidential floor,
      exiting non-zero with a message naming both counts and the floor
- [x] 2.4 Pin both floors as named constants with paired tests, each failure
      message naming the review obligation
- [x] 2.5 Rewrite all six tests in
      `tests/python/test_mc_plot_baseline_unit.py`; the shared fixture and the
      summary constructor both change shape

## 3. Fix the producer

- [x] 3.1 Record an environment-reported step failure as a distinct outcome
      instead of setting the truncation flag
- [x] 3.2 Abort the capture on an unknown-action or malformed-frame error
      code; halt after the pinned number of consecutive transient failures
- [x] 3.3 Assert at capture start that the handshake's schema identifier
      equals a recomputation over the repository's pinned configs, and fail
      the capture on mismatch
- [x] 3.4 Add `tests/python/test_v05_manual_baseline.py` driving the episode
      loop against a faked socket: an error frame yields a non-evidential
      outcome and no truncation flag; the step budget reached with no error
      stays evidential; three consecutive failures halt with exactly three
      records; a failure followed by a success continues; a contract-violation
      code aborts

## 4. Correct the record

- [ ] 4.1 Re-derive all three rows of the results table in
      `docs/results/v0.5-first-real-run.md` from the artifacts; the links are
      transposed, one row describes a run that was never committed, and one
      cites a snapshot matching neither row
- [ ] 4.2 Declare both baseline snapshots in place, recording what they are
      and why they carry no evidential episodes
- [ ] 4.3 Correct `docs/architecture.md`, which describes the thirty-record
      artifact as four episodes and documents the outlawed behaviour as
      intended
- [ ] 4.4 Retarget the stale `mc-bot/src/index.js` reference and the
      hard-coded dimensions in `.github/workflows/hf-model.yml` that cite the
      rewritten report

## 5. Gate

- [ ] 5.1 Add `tests/python/test_evidence_integrity.py`: every table row
      citing a snapshot under `docs/results/` agrees with it, and every
      snapshot with no evidential records carries a declaration
- [ ] 5.2 Negative-case tests over a synthetic tree, following the
      monkeypatched-root fixture pattern already used by the pinned-config
      guard: laundering by declaring everything, a declaration naming a
      nonexistent supersession, a row disagreeing on episode count, malformed
      and empty snapshots, a non-list record collection
- [ ] 5.3 Add `docs/results/INDEX.toml` listing each expected snapshot with
      its digest; the gate fails on a listed path that is missing or whose
      digest disagrees
- [ ] 5.4 Add a self-assertion, in a separate file, that the suite's marker
      expression excludes no marker this gate carries
- [ ] 5.5 Confirm the guard fails on the pre-correction tree and passes after,
      recording both transcripts in the pull-request body
- [ ] 5.6 Failure messages state the remedy, not only the mismatch

## 6. Rollout and charter

- [ ] 6.1 Land the gate advisory for one cycle with the promotion criterion
      written into the step comment, following the doctrine stated in the
      security workflow's header, then remove the marker
- [ ] 6.2 Add the third `workflow_dispatch`-only job to Invariant 6's list of
      deliberately non-blocking gates, which names its two siblings and omits
      it
- [ ] 6.3 Correct Invariant 6's advisory-scanner claim: it names one scanner
      as advisory, but three more run non-blocking and the static-analysis job
      does not run at all unless a repository variable enables it
