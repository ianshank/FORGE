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

The README / CHARTER Python steps-per-second headline is gated by
`tests/python/test_throughput_claim.py` against the committed
`cloud_agent/pyo3_step.json` report. Rust multi-agent scaling is a
separate measurement (Criterion `WorldState::step`) and must not be
cited as the Python headline.

## Profiles

- **`reference_a/`** — GitHub Actions `ubuntu-latest` runner (the same
  machine the `alloc-audit` and `bench` CI jobs run on, x86_64 Linux).
  Treat this profile as "anyone-can-reproduce-on-CI". `alloc_audit.json`
  is committed here. `multi_agent_scaling.json` is produced by the
  `bench` job as a CI artifact (`reference_a-multi_agent_scaling`); do
  not copy a non-GHA host's numbers into this directory.
- **`reference_b/`** — user-supplied workstation profile. Populated
  locally on a non-CI host (Apple Silicon laptop, dedicated x86 / arm64
  workstation). The directory ships with only a `.gitkeep`; do not block
  PRs on the absence of populated baseline files.
- **`cloud_agent/`** — labeled measurements from the Cursor cloud agent
  VM that closed the evidence gap. Hardware is recorded inside each JSON
  (`hardware.cpu`, `hardware.arch`). These numbers are **not**
  interchangeable with `reference_a` or `reference_b`.

## Files

| File | Producer | How to regenerate |
|---|---|---|
| `<profile>/alloc_audit.json` | `target/release/allocation_audit` | `cargo run -p forge-bench --bin allocation_audit --features dhat-heap --release -- --warmup 1024 --iters 10000 --agents 1,8,16,32,64,128 --out benchmarks/baselines/<profile>/alloc_audit.json` |
| `<profile>/multi_agent_scaling.json` | Criterion (`multi_agent_scaling` bench) + exporter | `make bench-export PROFILE=<profile>` (runs `cargo bench -p forge-bench --bench multi_agent_scaling` then `python3 benchmarks/runner/export_criterion_scaling.py`) |
| `<profile>/pyo3_step.json` | `tests/python/test_step_throughput.py` | `FORGE_RUN_STEP_THROUGHPUT=1 FORGE_STEP_THROUGHPUT_OUT=benchmarks/baselines/<profile>/pyo3_step.json pytest tests/python/test_step_throughput.py -s --no-cov` (requires `maturin develop` / the native `forge_env` extension) |

Each `multi_agent_scaling.json` row records **both** `env_steps_per_sec`
(whole-world `step()` calls per second, comparable to the Python
headline) and `agent_steps_per_sec` (`num_agents * env_steps_per_sec`,
matching Criterion's `Throughput::Elements(num_agents)`). Do not cite
the agent-normalised figure as the Python steps/second claim.

The audit's `--agents` flag accepts a comma-separated list of agent
counts; the same sweep can be set via the `FORGE_BENCH_AGENT_COUNTS`
environment variable (also honoured by `multi_agent_scaling`). The
default sweep `1,8,16,32,64,128` mirrors the bench so the two artefacts
agree on the canonical fan-out. Each audit row is labelled
`<base>@n=<count>` (for example `Move_Up@n=8`) and carries a typed
`num_agents` field for downstream tooling.

## Updating

When you commit a regenerated `alloc_audit.json`, make sure it still
satisfies the zero-allocation gate below. If you also refresh
throughput baselines such as `<profile>/multi_agent_scaling.json` or
`<profile>/pyo3_step.json`, update the corresponding human-readable
performance documentation in the same PR so it stays in sync with the
machine-readable evidence. `tests/python/test_throughput_claim.py` will
fail if a published Python steps/second floor exceeds the committed
PyO3 report.

A regenerated allocation baseline must satisfy the same gate the CI uses:

```bash
python3 benchmarks/runner/check_zero_alloc.py \
    --input benchmarks/baselines/<profile>/alloc_audit.json \
    --max-bytes 0
```
