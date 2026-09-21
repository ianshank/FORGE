# Tasks: v0.6.0 Release Baseline Hardening

## 1. Governance and OpenSpec Artifacts

- [x] 1.1 Create `openspec/changes/v0.6.0-release-baseline-hardening/` change package
      with `proposal.md`, `design.md`, `tasks.md`, and 6 capability specs.
- [x] 1.2 Wire `scripts/openspec` wrapper and `openspec-validate` CI/local gates
      executing `openspec validate --all --strict` (real executable gate).
- [x] 1.3 Ratify milestone naming as `complete-v0.6.0-gates` in release
      documentation, confirming that workspace version 0.6.0 is already single-sourced.

## 2. Python Distribution Honesty

- [x] 2.1 Audit `pyproject.toml` and documentation regarding supported Python
      versions; clearly record that CI actively validates Python 3.11 only.
- [x] 2.2 Verify that the `pip-install-clean` CI job builds a valid wheel on
      Linux x86_64 and passes import verification outside the workspace.
- [x] 2.3 Explicitly document that multi-platform wheels via `cibuildwheel` and
      automated PyPI publishing are deferred to post-v0.6.0 follow-on changes.
- [x] 2.4 Document the developer installation path via source checkout,
      `maturin develop`, and container images published to GHCR.

## 3. RL API Conformance Hardening

- [x] 3.1 Verify that `tests/python/test_api_compliance.py` executes full upstream
      `gymnasium.utils.env_checker.check_env` on single-agent wrappers.
- [x] 3.2 Verify that `test_passes_parallel_api_test` executes upstream
      `pettingzoo.test.parallel_api_test` on multi-agent environments.
- [x] 3.3 Confirm that `ForgeParallelEnv` drives native `step_multi` with per-agent
      actions and returns independent observations.
- [x] 3.4 Confirm that the dedicated `api-compliance` CI job is green and blocking.

## 4. Evaluation Evidence and Benchmark Integrity

- [x] 4.1 Confirm that `tests/python/test_throughput_claim.py` validates that
      published markdown claims do not exceed committed `pyo3_step.json` benchmarks.
- [x] 4.2 Verify that `tests/golden/replays/v2_seed42.json` passes byte-identity
      checks in `crates/forge-replay/tests/golden_replay.rs`.
- [x] 4.3 Validate that `docs/results/INDEX.toml` indexes all committed baseline
      snapshots with exact SHA-256 digests via `test_evidence_integrity.py`.
- [x] 4.4 Verify that zero-evidential snapshots (`v0.5-first-real-run-baseline.json`
      and `-v2.json`) maintain valid `.declaration` audit notes.

## 5. Model Artifact Provenance and Fail-Closed Contracts

- [x] 5.1 Verify that `scripts/hf_publish_model.py` defaults to injecting the
      `UNTRAINED_WARNING` banner into Hugging Face model cards.
- [x] 5.2 Enforce that no model bundle is published with `--trained` unless backed
      by evidential capture records meeting the floor of 3.
- [x] 5.3 Document that Hugging Face Space (`ianshank/forge-wasm-demo`) and GitHub
      Pages deployments are blocked until operator secrets (`HF_TOKEN`) and Pages
      settings are configured.

## 6. Drone Autonomy Boundary Verification

- [x] 6.1 Audit `configs/scenarios/orchard_coverage.toml` and verify compilation
      to `ForgeConfig` via `crates/forge-types/src/scenario.rs`.
- [x] 6.2 Confirm that drone flight safety is implemented strictly as deterministic
      process constraints in `crates/forge-core/src/systems.rs` (geofence margins,
      battery action floors, altitude caps converting invalid actions to `Noop`)
      with `aerial_drain_rate` functioning as the background energy model.
- [x] 6.3 Document that parameterized motor failure, GPS denial, and sensor noise
      models are non-goals for v0.6.0.

## 7. Operator Runbook and Release Execution

- [x] 7.1 Verify that all named kill criteria test suites pass on branch:
      `test_version_consistency.py`, `test_api_compliance.py`,
      `test_throughput_claim.py`, `test_evidence_integrity.py`, and
      `test_charter_alignment.py`.
- [x] 7.2 Operator action: Execute Decision D2 by renaming default branch to
      `main` using the GitHub API.
      Evidence: default branch is `main`; tip
      `fd9648008af47ef3acdff43f5c52ef6b232e51fc`.
- [x] 7.3 Operator action: Configure GitHub Pages source to "GitHub Actions" and
      add write-scoped `HF_TOKEN` repository secret.
      Evidence: Pages `build_type=workflow`; site <https://ianshank.github.io/FORGE/> ;
      Pages SUCCESS run 35661478769; `HF_TOKEN` secret set; HF Space SUCCESS run
      35661697902.
- [x] 7.4 Operator action: Create and push annotated tag `v0.6.0` from `main`
      after gates were green.
      Evidence: annotated tag `v0.6.0` on `fd964800` (tag object
      `6bbafb5823148adabb486d7a978cd38f5edf51bb`); release
      <https://github.com/ianshank/FORGE/releases/tag/v0.6.0> .
- [x] 7.5 Archive this change package under
      `openspec/changes/archive/v0.6.0-release-baseline-hardening/`.
      Note: repository root `openspec/specs/` does not exist yet; promoting
      capability deltas into a canonical `openspec/specs/` tree is deferred and
      must not invent that directory in this hygiene change.
