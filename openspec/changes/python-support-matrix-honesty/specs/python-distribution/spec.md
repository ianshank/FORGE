# Spec Delta: python-distribution (support-matrix honesty)

## MODIFIED Requirements

### Requirement: Declared CPython support is a closed set matching CI

The project SHALL declare supported CPython minors as a closed set in packaging metadata. Under Disposition A, the declared floor SHALL be `requires-python = ">=3.11"` (or an equivalent closed set that does not include untested 3.9/3.10).

#### Scenario: Declared floor matches proven CI

- GIVEN mainline CI packaging/smoke jobs run on CPython 3.11
- WHEN an installer reads `requires-python`
- THEN the declared floor does not advertise untested minors below 3.11

#### Scenario: Falsifier - fail-open floor

- GIVEN CI only exercises 3.11
- WHEN `requires-python` remains `>=3.9` (or otherwise wider than CI)
- THEN the change is incomplete and MUST NOT be marked done

### Requirement: CI smoke for every declared minor

For each CPython minor in the declared support set, CI SHALL run a blocking job that builds the native extension (maturin develop and/or maturin build as used today) and imports `forge_env` (prefer the existing `pip-install-clean` shape).

#### Scenario: 3.11 smoke remains blocking

- GIVEN Disposition A
- WHEN CI runs on a PR that breaks extension build or `forge_env` import on 3.11
- THEN the packaging/smoke job fails and blocks merge

#### Scenario: Falsifier - declared minor without job

- GIVEN a minor appears in `requires-python` / support docs
- WHEN no blocking CI smoke covers that minor on linux/amd64
- THEN merge MUST fail the support-matrix honesty gate

### Requirement: Declaration vs CI parity test

The repository SHALL include a test that compares packaging `requires-python` (or a single committed matrix manifest) to CI `python-version` pins and fails on drift.

#### Scenario: Drift fails CI

- GIVEN someone widens `requires-python` without adding CI coverage
- WHEN the parity test runs
- THEN the test fails

#### Scenario: Falsifier - docs claim wider than CI

- GIVEN README/CHARTER claim support for a minor absent from CI
- WHEN docs parity is checked against the closed matrix
- THEN the honesty gate fails until docs or CI are reconciled

### Requirement: api-compliance stays blocking

The `api-compliance` CI job SHALL remain a blocking gate on at least one supported minor (3.11 minimum under A).

#### Scenario: Compliance not dropped

- GIVEN a PR that breaks Gymnasium/PettingZoo compliance on 3.11
- WHEN `api-compliance` runs
- THEN the job fails and blocks merge

### Requirement: Forbid false wheel / PyPI claims

This change SHALL NOT claim cibuildwheel, multi-platform wheels, or PyPI publish as landed. linux/amd64 GitHub-hosted runners remain the only packaging surface in scope.

#### Scenario: Falsifier - inventing PyPI as done

- GIVEN no cibuildwheel/PyPI workflow exists
- WHEN release notes or specs claim PyPI wheels as shipped by this change
- THEN the claim is forbidden and MUST be removed

### Requirement: Forbid widening without coverage

The project SHALL NOT widen `requires-python` (or documented support) to a minor that lacks a green blocking extension-build + import smoke job.

#### Scenario: Falsifier - widen without matrix

- GIVEN someone sets `requires-python = ">=3.9"` again without 3.9/3.10 jobs
- WHEN parity/smoke gates run
- THEN CI fails
