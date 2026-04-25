# Benchmark baselines

Reference results for the FORGE measurement harnesses introduced in PR #36
(`crates/forge-bench/src/bin/allocation_audit.rs`,
`crates/forge-bench/benches/multi_agent_scaling.rs`,
`tests/python/test_step_throughput.py`).

These JSON files are the source of truth for the claims-verification table:
the CI alloc-audit job uses
`benchmarks/runner/check_zero_alloc.py --max-bytes 0` to gate every PR
against the `total_bytes == 0` invariant, and the throughput numbers are
human-readable evidence for the README's performance claims.

## Profiles

The two abstract profiles are referenced in
`crates/forge-bench/src/env.rs` and `multi_agent_scaling.rs`:

- **`reference_a/`** — committed, hardware: GitHub Actions `ubuntu-latest`
  runner (the same machine the `alloc-audit` CI job runs on, x86_64
  Linux). Treat this profile as "anyone-can-reproduce-on-CI".
- **`reference_b/`** — placeholder, awaiting a second hardware profile to
  be named (Apple Silicon laptop / dedicated workstation / arm64
  cloud instance — to be decided alongside the next benchmark
  refresh). Do not block PRs on the absence of `reference_b/`.

## Files

| File | Producer | How to regenerate |
|---|---|---|
| `<profile>/alloc_audit.json` | `target/release/allocation_audit` | `cargo run -p forge-bench --bin allocation_audit --features dhat-heap --release -- --warmup 1024 --iters 10000 --out benchmarks/baselines/<profile>/alloc_audit.json` |
| `<profile>/multi_agent_scaling.json` | Criterion (`multi_agent_scaling` bench) | `cargo bench -p forge-bench --bench multi_agent_scaling -- --save-baseline <profile>` then export the relevant rows |

## Updating

When you commit a regenerated `alloc_audit.json`, make sure it still
satisfies the zero-allocation gate below. If you also refresh
throughput baselines such as `<profile>/multi_agent_scaling.json`,
update the corresponding human-readable performance documentation in
the same PR so it stays in sync with the machine-readable evidence.

A regenerated baseline must satisfy the same gate the CI uses:

```bash
python3 benchmarks/runner/check_zero_alloc.py \
    --input benchmarks/baselines/<profile>/alloc_audit.json \
    --max-bytes 0
```
