# Specification: Python Distribution and Packaging Topology

## Purpose

Define the packaging, distribution, and runtime environment boundaries for
the FORGE Python package (`forge-env`) for the v0.6.0 release, establishing
honest documentation of verified platforms, version constraints, and artifact
registries.

## MODIFIED Requirements

### Requirement: Document Honest Platform and Python Version Matrix
The repository packaging metadata and release documentation SHALL accurately
reflect the platforms and Python versions validated by automated CI. While
`pyproject.toml` declares `requires-python = ">=3.9"`, release documentation
SHALL clearly document that automated CI validation is performed exclusively
on Python 3.11 on Linux x86_64 (`ubuntu-latest`). If broader Python version
support is claimed without multi-version CI jobs, `requires-python` MUST be
narrowed to tested versions.

#### Scenario: Contributor inspects supported runtime matrix
- **GIVEN** `pyproject.toml` declares `requires-python = ">=3.9"`
- **AND** CI workflow `.github/workflows/ci.yml` runs test jobs on Python 3.11
- **WHEN** release documentation or package metadata is inspected
- **THEN** it SHALL state that Python 3.11 on Linux x86_64 is the primary
  CI-verified environment
- **AND** SHALL NOT claim multi-platform verification that does not exist

#### Scenario: Falsifier: Claiming unvalidated Python versions as CI-tested
- **GIVEN** a release document or packaging claim
- **WHEN** it claims Python 3.9, 3.10, 3.12, and 3.13 are fully verified in CI
- **THEN** an audit against `.github/workflows/ci.yml` SHALL falsify the claim
  because those versions have no automated workflow jobs

### Requirement: Clean Wheel Install Smoke Verification
Every release candidate wheel SHALL be verified via an isolated clean-install
smoke test in CI that builds a release wheel using `maturin`, installs it into
an empty virtual environment outside the source tree, and executes native
imports and environment resets.

The wheel smoke verification SHALL use `ForgeEnv` as the fast native binding
smoke path to assert extension loading, basic stepping, and version agreement
with the workspace. Formal compliance with standard reinforcement learning APIs
SHALL be governed separately by `ForgeGymnasiumEnv` and `ForgeParallelEnv`
under the dedicated `api-compliance` CI job.

#### Scenario: Wheel installs and runs in an isolated environment
- **GIVEN** a freshly built release wheel in `dist/*.whl`
- **WHEN** the wheel is installed into a new virtual environment without source
  access
- **AND** `from forge_env import ForgeEnv` is executed from outside the repo
- **THEN** `ForgeEnv` SHALL instantiate and complete a reset cycle successfully
- **AND** the wheel version SHALL exactly match the Cargo workspace version

#### Scenario: Falsifier: Wheel missing compiled native extensions
- **GIVEN** a wheel built without compiled PyO3 binaries
- **WHEN** `from forge_env import ForgeEnv` is executed in an isolated environment
- **THEN** the import SHALL fail or return `None`
- **AND** the `pip-install-clean` CI job SHALL exit non-zero

### Requirement: Fence Multi-Platform cibuildwheel and PyPI Publishing as Non-Goals
The v0.6.0 release SHALL NOT require or claim automated PyPI publishing or
multi-platform wheels via `cibuildwheel`. Artifact distribution for v0.6.0
SHALL be restricted to local source installation via `maturin`, repository
release tags, and prebuilt Docker images on GitHub Container Registry (GHCR).
Multi-platform wheels for macOS, Windows, and Linux ARM64 SHALL be treated as
deferred follow-on changes.

#### Scenario: Verifying artifact distribution channels
- **GIVEN** the v0.6.0 release package
- **WHEN** distribution channels are enumerated
- **THEN** they SHALL name source builds (`maturin develop`), local wheels, and
  GHCR containers
- **AND** SHALL NOT advertise a PyPI package or multi-platform wheel matrix
  as existing

#### Scenario: Falsifier: Advertising PyPI wheel availability for v0.6.0
- **GIVEN** user installation instructions
- **WHEN** instructions state `pip install forge-env` directly from PyPI
- **THEN** verification against PyPI SHALL falsify the claim because no PyPI
  publication pipeline exists
