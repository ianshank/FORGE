# Design: v0.6.0 Release Baseline Hardening

## Context

FORGE provides a high-performance, deterministic simulation runtime in Rust
with Python and WebAssembly bindings. As part of preparing for the v0.6.0
developer preview, an intake audit revealed discrepancies between intended
release claims and underlying repository reality. While the core engine,
Gymnasium/PettingZoo wrappers, and throughput benchmarks operate reliably in
CI, previous planning drafts introduced assumptions regarding multi-platform
wheels, PyPI automation, automated signing, hardware fault injection, and
phantom version bumps.

This design document establishes the architectural principles, decisions, and
operational boundaries necessary to execute the v0.6.0 release baseline
hardening without scope creep or compromise of technical integrity.

## Goals and Non-Goals

### Goals
- Ground the v0.6.0 release entirely in landed, machine-verifiable artifacts.
- Enforce release kill criteria exclusively through named CI and test targets.
- Clarify Python packaging boundaries: maturin local builds and Linux x86_64
  wheel smoke in CI, documenting Python 3.11 CI execution against >=3.9
  declared support.
- Preserve deterministic replay fixtures and benchmark throughput claim gates.
- Implement honest model artifact publication contracts with mandatory
  untrained warnings and fail-closed secret handling.
- Fence drone autonomy to L0 process constraints (geofence margins, battery
  floors, altitude ceilings).
- Execute the operator runbook for Decision D2 (branch rename to main) and
  release tagging.

### Non-Goals
- Adding cibuildwheel matrices or automating PyPI publishing for v0.6.0.
- Introducing Cosign signing or SBOM generation as release blockers.
- Rewriting or re-architecting existing Gymnasium and PettingZoo wrappers.
- Building physical motor or GPS sensor fault injection products.
- Fabricating or publishing unbacked "trained" MuZero Minecraft weights.
- Bleeding scope into Distilled_Agents or Neuroharness i2 ecosystems.
- Falsifying version history by proposing a bump to 0.6.0 or reverting to 0.2.0.

## Architectural Decisions

### Decision 1: Milestone and Version Grounding (complete-v0.6.0-gates)
- **Choice:** Maintain workspace version 0.6.0 across Cargo.toml,
  python/forge_env/_version.py, dashboard/package.json, and
  dashboard/package-lock.json. Name the release milestone
  `complete-v0.6.0-gates`.
- **Rationale:** Revision D1 in docs/plans/forge_v0_2_0_release_plan.md
  already bumped the version to 0.6.0 to preserve SemVer monotonicity after
  0.5.0. Downgrading to 0.2.0 breaks SemVer and test_version_consistency.py.
  Treating 0.6.0 as a new bump is historically false. The release milestone
  completes the gating verification.

### Decision 2: Default Branch Rename via Decision D2
- **Choice:** Execute branch rename from claude/plan-forge-environment-htAoK
  to main using the GitHub API:
  `gh api --method POST "repos/ianshank/FORGE/branches/claude%2Fplan-forge-environment-htAoK/rename" -f new_name=main`.
- **Rationale:** Branch rename preserves commit SHA provenance, retargets all
  28 open pull requests automatically, transfers branch protection rules, and
  maintains integrity for commit hashes cited in docs and benchmark evidence.
  Re-creating main or squashing into an orphan commit destroys provenance and
  requires manual PR retargeting.

### Decision 3: Python Distribution Reality and CI Matrix Honesty
- **Choice:** Document that v0.6.0 provides source distribution and local
  maturin builds, alongside Linux x86_64 wheels verified via the
  pip-install-clean CI job, and prebuilt containers on GHCR. Clearly document
  that CI currently tests only Python 3.11, while pyproject.toml declares
  requires-python = ">=3.9".
- **Remediation Option:** Document this constraint honestly in README and
  release notes, or narrow requires-python to ">=3.11,<3.12" if full-matrix
  guarantees cannot be run in CI. Defer multi-platform cibuildwheel (macOS,
  Windows, Linux ARM64) and PyPI publishing to post-v0.6.0 follow-on change
  ids.

### Decision 4: Wrap and Harden Existing RL Adapters
- **Choice:** Maintain the existing Gymnasium (ForgeGymnasiumEnv) and
  PettingZoo (ForgeParallelEnv) wrappers in python/forge_env/.
- **Rationale:** These wrappers are fully implemented, inherit from upstream
  base classes, utilize space_builder.py to coerce native data types into
  formal spaces, and drive the real native multi-agent step_multi engine. They
  are validated by api-compliance in CI running the official upstream test
  suites (gymnasium.utils.env_checker.check_env and
  pettingzoo.test.parallel_api_test). They are treated as hardened regression
  gates, not new additions.

### Decision 5: Determinism via CompactReplay v2 Goldens
- **Choice:** Enforce bit-identical replay determinism using
  tests/golden/replays/v2_seed42.json and the golden-replay CI workflow.
- **Rationale:** CompactReplay v2 freezes config hashing and timestamps
  (1970-01-01T00:00:00+00:00) to ensure zero flakiness. Any schema or logic
  change that alters replay bytes fails the golden_replay test, requiring
  explicit maintainer approval via UPDATE_GOLDEN_REPLAYS=1 and an entry in
  docs/results/replay_flip_log.md.

### Decision 6: Benchmark Integrity via test_throughput_claim.py
- **Choice:** Gate all published performance claims against committed PyO3
  benchmark artifacts (benchmarks/baselines/cloud_agent/pyo3_step.json).
- **Rationale:** Throughput claims in README.md, docs/CHARTER.md, and
  BENCHMARKS.md must never exceed committed measurements (189,439 steps/second
  on the cloud_agent profile). test_throughput_claim.py scans documentation
  and asserts that claimed numbers do not exceed the committed baseline.

### Decision 7: Evidence Envelope and Snapshot Auditing
- **Choice:** Retain the evidence architecture established in
  refuse-non-evidential-aggregates: every result snapshot under docs/results/
  must be indexed in docs/results/INDEX.toml with its SHA-256 digest, and any
  snapshot lacking evidential episodes must carry an explicit .declaration file.
- **Rationale:** Prevents manufacturing plausible aggregates from failed
  episodes. The forge-eval crate provides RunManifest and OutputConfig
  reproducibility structures. Generic omnibus files like run.json remain
  optional follow-on ideas rather than release blockers.

### Decision 8: Fail-Closed Model Provenance and Secret Handling
- **Choice:** Mandate that scripts/hf_publish_model.py injects the
  UNTRAINED_WARNING banner by default unless --trained is explicitly passed,
  and prohibit claiming a trained model without evidential capture backing.
  Document GitHub Pages and HF_TOKEN deployments as blocked/non-done until
  repository secrets are configured.
- **Rationale:** The repository currently has an operational bootstrap
  pipeline for random-init weights, but no trained MuZero weights for
  Minecraft. Presenting a random-init bundle as a trained agent violates
  scientific integrity. GitHub Pages requires setting the Actions source, and
  Hugging Face deployment requires setting the HF_TOKEN repository secret. Both
  must be handled by operator action or acknowledged as non-done.

### Decision 9: L0 Drone Process Constraints vs. Physical Faults
- **Choice:** Define drone autonomy in orchard_coverage.toml as L0 synthetic
  coverage governed by deterministic process constraints in
  crates/forge-core/src/systems.rs.
- **Rationale:** The simulation enforces boundary and energy safety: moves
  violating geofence_margin fall back to Action::Noop; actions below
  battery_action_floor fall back to Action::Noop; ascents above max_altitude
  fall back to Action::Noop. True motor failure, GPS denial, and sensor noise
  parameters do not exist in the codebase and must not be advertised as
  available features.

## Named Kill Criteria

The release readiness of FORGE v0.6.0 is determined strictly by the status of
named CI jobs and test artifacts:

1. `test_version_consistency.py`: Asserts workspace version 0.6.0 is identical
   across Cargo.toml, forge_env/_version.py, dashboard/package.json, and
   dashboard/package-lock.json.
2. `api-compliance` CI job: Executes gymnasium.utils.env_checker.check_env and
   pettingzoo.test.parallel_api_test against live native extensions.
3. `pip-install-clean` CI job: Verifies that maturin builds a clean wheel,
   installs into an isolated environment, imports outside the source tree, and
   matches Cargo version.
4. `test_throughput_claim.py`: Verifies that documentation claims do not exceed
   the 189k steps/second baseline in pyo3_step.json.
5. `golden-replay` CI job: Verifies byte-level determinism of
   tests/golden/replays/v2_seed42.json via forge-replay golden tests.
6. `test_evidence_integrity.py` and `docs/results/INDEX.toml`: Verifies that
   all results snapshots are indexed with SHA-256 digests and empty snapshots
   carry valid declarations.
7. Decision D2 execution: Annotated tag v0.6.0 applied only to main following
   successful default branch rename.
