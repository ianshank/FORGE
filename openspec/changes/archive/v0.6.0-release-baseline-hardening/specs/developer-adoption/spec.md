# Specification: Developer Adoption and Quickstart Ergonomics

## Purpose

Define reproducible developer adoption workflows for FORGE v0.6.0, focusing
on source builds, local virtual environments, Docker container execution, and
reproducible seed verification.

## MODIFIED Requirements

### Requirement: Reproducible Local Source Installation Journey
The primary developer installation journey SHALL be supported through source
checkout, virtual environment creation, and local `maturin` builds. Clear
instructions SHALL guide the user to compile native extensions and verify
installation with a single Python command.

`ForgeEnv` SHALL serve as the fast native binding smoke path to verify that
compiled Rust extensions load and step correctly outside the repository tree.
Standard reinforcement learning workflows requiring formal Gym/PettingZoo
interfaces SHALL consume `ForgeGymnasiumEnv` (single-agent) and
`ForgeParallelEnv` (multi-agent), which define the ecosystem compliance path
validated by the `api-compliance` CI gate.

#### Scenario: Developer builds and runs from source checkout
- **GIVEN** a clean clone of the FORGE repository
- **AND** a Python 3.11 environment with `pip` and `maturin` installed
- **WHEN** the developer executes `maturin develop` inside `crates/forge-python`
  or runs `pip install -e .`
- **THEN** `from forge_env import ForgeEnv` SHALL succeed as the native binding
  smoke check
- **AND** `ForgeEnv().reset(seed=42)` SHALL return a valid initial observation
- **AND** `ForgeGymnasiumEnv` and `ForgeParallelEnv` SHALL instantiate for RL
  training loops with standards-compliant observation and action spaces

#### Scenario: Falsifier: Incomplete installation prevents environment execution
- **GIVEN** a developer following the documented quickstart commands
- **WHEN** commands fail due to missing dependencies, unstated environment
  variables, or path errors
- **THEN** the quickstart journey SHALL be considered broken
- **AND** CI smoke jobs like `pip-install-clean` SHALL fail

### Requirement: Verified Container Adoption Workflow via GHCR
Developers opting for containerized workflows SHALL be able to pull and run
official FORGE Docker images from GitHub Container Registry (`ghcr.io/ianshank/forge`),
which provide pre-installed Rust, Python, and native dependencies.

#### Scenario: Running simulation in prebuilt GHCR container
- **GIVEN** a developer with Docker installed
- **WHEN** the developer executes the documented Docker run command for
  `forge-mc-runner` or `forge-server`
- **THEN** the container SHALL start cleanly without requiring local Rust or
  Python build toolchains
- **AND** the service endpoints SHALL be accessible on the documented ports

#### Scenario: Falsifier: Container image fails to start or missing dependencies
- **GIVEN** a published GHCR container image
- **WHEN** `docker run` is executed with default configuration
- **THEN** if entrypoint fails or required shared libraries (such as GLIBC or
  libstdc++) are missing, container validation SHALL fail

### Requirement: Deterministic Seed Demonstration Workflow
The developer adoption documentation SHALL provide a self-contained,
minimal example demonstrating deterministic stepping across identical seeds.

#### Scenario: Developer verifies simulation determinism locally
- **GIVEN** a minimal Python script running two independent environments
- **WHEN** both environments are reset with seed 42 and stepped with identical
  action sequences
- **THEN** observations, rewards, and terminations SHALL be identical on every
  step
- **AND** the quickstart documentation SHALL provide this code example directly

#### Scenario: Falsifier: Non-deterministic output under identical seeds
- **GIVEN** two runs initialized with identical seeds and actions
- **WHEN** observations or rewards differ between the runs
- **THEN** `tests/python/test_determinism.py` SHALL fail
- **AND** developer reproducibility claims SHALL be invalidated
