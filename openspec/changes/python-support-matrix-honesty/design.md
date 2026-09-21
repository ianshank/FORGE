# Design: python-support-matrix-honesty

## Context

Live main (`fd964800`) ships workspace `0.6.0` with annotated tag `v0.6.0`. CI Python jobs are pinned to 3.11. `pyproject.toml` still declares a wider `requires-python` floor (`>=3.9`). There is no `openspec/specs/` tree yet; deltas live under `openspec/changes/` until Conductor promotes archived specs.

## Decision: Disposition A (binding)

**A - Narrow** is the Intake default and is binding for this change.

- Set `requires-python` to match proven CI: floor `>=3.11` (closed set starts at 3.11 on linux/amd64).
- **B - Expand** is an override only. Spec Writer does **not** choose B here. Any future B requires an explicit design rationale and CI-cost note, plus a green smoke job for every declared minor.

Kill rule (unchanged under A or B): every declared CPython minor has a green blocking CI job that builds the extension and imports `forge_env`.

## CI shape

1. Prefer extending existing `pip-install-clean` with a `python-version` matrix over inventing a parallel packaging stack.
2. Under A, the matrix is effectively `{3.11}` until more minors are added with jobs.
3. `api-compliance` remains blocking on at least 3.11. Expanding compliance to every minor is optional cost-scope, not required by A.
4. Runners: linux/amd64 GitHub-hosted only. No cibuildwheel. No macOS/Windows/ARM64 wheels in this change.

## Parity test

Add or extend a unit/meta test that:

1. Parses `requires-python` from `pyproject.toml` (or a single committed matrix manifest both CI and the test read).
2. Compares against workflow `python-version` pins used by packaging/smoke jobs.
3. Fails on drift (declared wider than tested, or CI missing a declared minor).

## Docs and SemVer

- One matrix table across README / CHARTER / CONTRIBUTING / CHANGELOG.
- Separate "Python minor support" (this change) from "multi-platform wheels" (still deferred).
- SemVer advice: bump workspace to **0.6.1** when Conductor cuts the packaging honesty release; tag `v0.6.1`. Include a CHANGELOG migration note that 3.9/3.10 are no longer declared. Not a 0.7.0 story.

## Out of scope pointers

Defer to backlog change ids (names advisory): wheel-matrix-pypi-phase1, ghcr-release-provenance, trained MuZero, fault injection, evidence omnibus `run.json`.
