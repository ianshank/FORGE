# Proposal: python-support-matrix-honesty

**Status:** proposed  
**Date:** 2026-09-21  
**Repo:** ianshank/FORGE  
**Base tip:** main @ fd964800  
**Workspace / tag:** 0.6.0 already cut. Ship this change as packaging honesty under SemVer advice **0.6.1** (Conductor tags later). Do not say bump to 0.6.0.

## Why

v0.6.0 closed the stage-gate and documented that CI exercises CPython **3.11** on linux/amd64 while `pyproject.toml` still declares `requires-python = ">=3.9"`. That is a fail-open packaging contract: pip will install onto minors FORGE has never proven in CI.

Closed change `v0.6.0-release-baseline-hardening` (Conductor archiving; not an active product change) fenced cibuildwheel/PyPI and recorded the honesty gap. This package closes declared-vs-tested parity only.

## What Changes

- **Disposition A (binding):** narrow `requires-python` to `>=3.11` so the declared floor matches proven CI.
- Keep existing 3.11 blocking jobs (`python-test`, `pip-install-clean`, `api-compliance` at least).
- Add a parity test that fails if declaration drifts below the CI matrix.
- Align README / CHARTER / CONTRIBUTING / CHANGELOG wording to the same closed matrix.
- Prefer extending `pip-install-clean` with an explicit `python-version` matrix over inventing a parallel packaging stack.

## Non-goals

- cibuildwheel; PyPI publish; Cosign/SBOM
- Trained MuZero / changing HF `trained=` defaults
- Motor/GPS fault product; omnibus `run.json`
- Distilled_Agents; Neuroharness i2
- Re-opening Gymnasium / PettingZoo / golden / throughput as greenfield
- Claiming multi-platform wheels because the Python minor matrix is honest
- macOS / Windows / ARM64 wheel matrices

## Kill criteria (named artifacts only)

| Gate | Artifact |
| --- | --- |
| Declared vs tested parity | Test parsing `requires-python` vs CI `python-version` pins (or shared manifest) fails on drift |
| Smoke per supported minor | maturin build/develop + import `forge_env` + `pip check` (reuse `pip-install-clean` shape) |
| API compliance | `api-compliance` remains blocking on at least 3.11 |
| Docs parity | README / CHARTER / CONTRIBUTING / CHANGELOG state the same matrix |
| OpenSpec | `openspec validate --all --strict` green when CLI exists |
| No false wheel claim | No cibuildwheel/PyPI landed by this change |

## Citations

- Archive / prior: `refuse-non-evidential-aggregates`
- Closed stage-gate: `v0.6.0-release-baseline-hardening` (archived or archiving; cite only as prior fence)
- Intake binding: Disposition A CONDITIONAL GO must-fixes applied

## Product fence

FORGE ≠ Distilled_Agents ≠ Neuroharness i2.
