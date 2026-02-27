# PRD — E2: Benchmark Regression CI

**Epic Slug**: `benchmark-regression-ci`  
**Priority**: P0  
**Sprint**: 2  
**Size**: M

---

## User Story

> **As a** FORGE core contributor,  
> **I want** CI to automatically detect if a PR degrades step throughput by more than 10%,  
> **So that** performance regressions are caught before merging and the 130K+ steps/sec guarantee is maintained.

---

## Problem Statement

FORGE's primary competitive differentiator is raw speed. Without automated regression tracking, performance regressions can creep in undetected. The `forge-bench` Criterion benchmarks exist but are not integrated into CI decision gates.

---

## Acceptance Criteria

| # | Given | When | Then |
|---|---|---|---|
| AC1 | A PR modifies `forge-core` | CI runs on the PR | Criterion benchmarks run and produce JSON output |
| AC2 | Benchmark result exceeds 10% slowdown vs baseline | CI finishes | PR check fails with a comment showing the delta |
| AC3 | Benchmark result is within 10% | CI finishes | PR check passes |
| AC4 | A new PR improves throughput | After merge | Baseline JSON artifact is updated and stored |
| AC5 | A developer views the PR | At any time | A CI comment shows step-throughput trend (last 10 runs) |

---

## Out of Scope

- Multi-metric dashboards (only `step_throughput` and `world_create` benchmarks gated)
- Historical charts in the UI (separate epic E5)
- Benchmarks on non-Linux runners (noise; Linux-only)

---

## Success Metrics

- Zero undetected throughput regressions > 10% since feature launch
- Benchmark CI step completes in < 3 minutes
- Baseline artifact stored as a GitHub Actions artifact and cached

---

## Open Questions

1. Which Criterion benchmarks are primary gates? (`step_throughput` vs also `world_create`)
2. Use `critcmp` CLI or custom Python script for delta computation?
3. What is the acceptable variance threshold for shared CI runners? (recommend 10%, not 5%)

---

## Implementation Notes

- Add `bench.yml` workflow: `cargo bench -p forge-bench -- --output-format=json`
- Use `critcmp` to compare against stored baseline artifact
- Store baseline as GitHub Actions artifact, keyed by `main` SHA
- Post PR comment via `actions/github-script`
