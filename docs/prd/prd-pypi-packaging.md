# PRD — E1: PyPI Packaging

**Epic Slug**: `pypi-packaging`  
**Priority**: P0  
**Sprint**: 1  
**Size**: L

---

## User Story

> **As a** machine-learning researcher or RL practitioner,  
> **I want to** install FORGE with a single `pip install forge-env` command,  
> **So that** I can integrate the simulation into my training pipeline without building Rust from source.

---

## Problem Statement

Currently FORGE requires users to: install Rust, install `maturin`, clone the repo, and run `maturin build --release`. This 15-minute setup process is a hard barrier to adoption. Beta release requires zero-friction installation.

---

## Acceptance Criteria

| # | Given | When | Then |
|---|---|---|---|
| AC1 | A user has Python 3.9+ and pip | `pip install forge-env` | Package installs without error on Linux (manylinux), macOS, Windows |
| AC2 | User runs `python -c "from forge_env import ForgeEnv; env = ForgeEnv(); env.reset()"` | After pip install | No ImportError, env resets correctly |
| AC3 | The wheel is uploaded to PyPI | A new version tag is pushed to `main` | GitHub Actions publishes the wheel automatically |
| AC4 | User pins `forge-env==0.2.0b1` | Running `pip install forge-env==0.2.0b1` | Exact version installs (version metadata correct) |
| AC5 | Library consumer on macOS ARM64 | `pip install forge-env` | macOS universal2 or arm64 wheel installs natively |

---

## Out of Scope

- Windows wheels (may target Windows after beta)
- Publishing to conda-forge (post-beta)
- Source distribution (sdist) — binary wheels only for beta

---

## Success Metrics

- PyPI page live at `https://pypi.org/project/forge-env`
- Install time < 30 seconds on manylinux runner
- Zero `pip install` failures on supported platforms in CI
- Download count tracked via PyPI stats

---

## Open Questions

1. Should we publish under `forge-env` or the org name `forge-rl`?
2. Do we need manylinux2010 or manylinux2014 minimum? (affects Python 3.9 compat)
3. Secret management for PyPI token — repo secret or environment secret?

---

## Implementation Notes

- Use `maturin publish` in a dedicated `release.yml` workflow
- Build matrix: `ubuntu-latest`, `macos-latest` (ARM + x86), `windows-latest`
- Use `cibuildwheel` or `maturin`'s built-in cross-compilation for Linux
- Tag format: `v0.2.0-beta.1` → PyPI pre-release `0.2.0b1`
