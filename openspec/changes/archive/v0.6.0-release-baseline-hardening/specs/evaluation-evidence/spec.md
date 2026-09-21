# Specification: Evaluation Evidence and Benchmark Integrity

## Purpose

Ensure all published performance, determinism, and evaluation results are
strictly grounded in committed, machine-verifiable artifacts, upholding the
principles of the `refuse-non-evidential-aggregates` archive.

## MODIFIED Requirements

### Requirement: Documentation Throughput Claims Bound to Committed Baselines
Published throughput numbers in `README.md`, `BENCHMARKS.md`, and
`docs/CHARTER.md` SHALL NOT exceed the numbers recorded in committed JSON
benchmarks. Single-environment Python step throughput claims SHALL be gated
against `benchmarks/baselines/cloud_agent/pyo3_step.json` (currently 189,439
steps/second). Claims exceeding committed values SHALL fail the
`test_throughput_claim.py` test suite.

#### Scenario: Documentation claim matches or is below committed evidence
- **GIVEN** committed benchmark report `pyo3_step.json` reporting 189,439 steps/sec
- **WHEN** `tests/python/test_throughput_claim.py` scans repository documentation
- **THEN** all claimed throughput numbers (such as 130,000+ or 189k) SHALL be
  less than or equal to the committed value
- **AND** the test suite SHALL pass

#### Scenario: Falsifier: Markdown claim exceeds committed benchmark
- **GIVEN** a documentation edit claiming "250,000+ steps/second from Python"
- **WHEN** `test_throughput_claim.py` executes against `pyo3_step.json` (189k)
- **THEN** the test SHALL fail with an actionable error instructing the user to
  either lower the claim or re-run benchmarks on a verified profile

### Requirement: Bit-Identical Determinism via CompactReplay v2 Goldens
The simulation core SHALL produce bit-identical serialization for golden
episodes. PR CI and weekly scheduled workflows SHALL execute
`crates/forge-replay/tests/golden_replay.rs` against
`tests/golden/replays/v2_seed42.json`. Any byte-level divergence in output,
config hash pinned in `tests/golden/replays/v2_seed42.json`
(`e831b2c14031f0d10d4ebd3fff8259ec66bcc0727335d183c5b994f73eb27071`),
or replay format version SHALL fail the test.

#### Scenario: Golden replay bit-identity verification
- **GIVEN** the canonical replay configuration and random seed 42
- **WHEN** `cargo test -p forge-replay --test golden_replay` runs in CI
- **THEN** serialized replay JSON SHALL match `v2_seed42.json` byte-for-byte
- **AND** the config hash SHALL match the frozen reference value

#### Scenario: Falsifier: Unaudited schema change alters replay serialization
- **GIVEN** a modification to `ForgeConfig` or step serialization logic
- **WHEN** `golden_replay.rs` executes without `UPDATE_GOLDEN_REPLAYS=1`
- **THEN** the test SHALL fail with a diff of expected versus actual bytes
- **AND** CI SHALL reject the pull request

### Requirement: Result Snapshot Integrity and Declaration Auditing
Every benchmark and evaluation snapshot stored under `docs/results/` SHALL be
indexed in `docs/results/INDEX.toml` with its exact SHA-256 digest. Any snapshot
containing zero evidential episodes SHALL carry an accompanying `.declaration`
file explaining its retention. Modifying or deleting a snapshot without an index
update, or laundering a non-evidential snapshot, SHALL fail
`test_evidence_integrity.py`.

#### Scenario: Committed snapshots match indexed checksums
- **GIVEN** baseline snapshots listed in `docs/results/INDEX.toml`
- **WHEN** `tests/python/test_evidence_integrity.py` validates repository state
- **THEN** computed file SHA-256 digests SHALL match indexed values exactly
- **AND** all zero-evidential snapshots SHALL have valid `.declaration` files

#### Scenario: Falsifier: Modifying a result snapshot without index update
- **GIVEN** an existing snapshot file under `docs/results/`
- **WHEN** contents of the snapshot are modified without updating `INDEX.toml`
- **THEN** `test_evidence_integrity.py` SHALL fail with a checksum mismatch
- **AND** the gate SHALL report the offending file and expected hash

### Requirement: Reproducibility Envelope via forge-eval RunManifest
Evaluation runs conducted via the `forge-eval` harness SHALL capture a
complete `RunManifest` including `git_sha`, `git_branch`, `rustc_version`,
`user`, `config_hash`, and scenario file hashes. Generic omnibus files
(such as `run.json`) SHALL remain optional follow-ons and SHALL NOT be
treated as v0.6.0 release blockers.

#### Scenario: Evaluation harness emits complete RunManifest
- **GIVEN** an evaluation execution using `forge-eval`
- **WHEN** outputs are emitted to disk or exported to MLflow/Hugging Face
- **THEN** `manifest.json` SHALL include valid Git SHA, config hash, and scenario
  hashes
- **AND** the evaluation run SHALL be fully reproducible from the recorded manifest

#### Scenario: Falsifier: Evaluation run executed without reproducibility metadata
- **GIVEN** an evaluation output directory produced by `forge-eval`
- **WHEN** the artifacts are inspected for provenance
- **THEN** `manifest.json` MUST be present
- **AND** if absent or missing Git and config hashes, the run SHALL NOT be
  admissible as verified evidence
