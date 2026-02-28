# Agent.md — forge-bench

## Persona

You are the **Performance Guardian** — the benchmarking suite that validates FORGE's hot-path performance claims. You measure step throughput, world creation overhead, and serialization costs across varying world sizes and agent counts using statistically rigorous Criterion benchmarks. You ensure the simulation engine stays under its <1µs per step target and catch performance regressions before they reach production.

## Design Patterns

### Criterion Benchmark Framework
All benchmarks use Criterion.rs with `harness = false` for custom control:
- **Benchmark groups** for parameterized comparisons (grid size, agent count)
- **BenchmarkId** for labeled parameter variants within groups
- **black_box()** to prevent dead-code elimination of results
- **HTML reports** for detailed statistical output in `target/criterion/`

### Deterministic Benchmark Configuration
`make_config(width, height, num_agents)` creates reproducible configs:
- Fixed seed (42) for deterministic world generation
- `max_episode_length = 0` to disable truncation during measurement
- Only simulation-relevant parameters are set; rendering and curriculum are irrelevant

### Scalability Testing Dimensions
Benchmarks test two independent scaling axes:
- **World size**: 16x16, 32x32, 64x64, 128x128 (quadratic growth in tile count)
- **Agent count**: 1, 2, 4, 8 agents on fixed 64x64 grid

### Baseline Measurement
`bench_step_noop` establishes the pure-overhead floor — a Noop action on 64x64 with 1 agent measures the minimum cost of the system pipeline without meaningful work.

### Comprehensive Hot-Path Coverage
Five benchmark groups cover the critical operations:
1. **step_single_agent** — step throughput vs world size
2. **step_multi_agent** — step throughput vs agent count
3. **step_noop** — baseline overhead measurement
4. **world_creation** — `WorldState::new()` including full world generation
5. **serialization** — `to_bytes()` state snapshot performance

## Crate Dependencies

- **Depends on**: `forge-types` (ForgeConfig), `forge-core` (WorldState — the primary benchmark target), `forge-agent` (agent baselines for episode benchmarks)
- **Depended on by**: None (leaf benchmarking crate)
- **External dependencies**: `criterion`, `rand`, `rand_pcg`

## Module Layout

| File | Purpose |
|------|---------|
| `benches/step_throughput.rs` | All Criterion benchmarks — step single/multi-agent, noop baseline, world creation, serialization |

## Key Invariants

- **All benchmarks use seed 42**: Fixed seed ensures deterministic, reproducible measurements
- **`max_episode_length = 0`**: Disables truncation during measurement to prevent early termination
- **`black_box()` on all results**: Prevents dead-code elimination of benchmark outputs
- **Grid size parameters**: 16x16, 32x32, 64x64, 128x128 (quadratic tile count scaling)
- **Agent count parameters**: 1, 2, 4, 8 on fixed 64x64 grid
- **Noop baseline measures pure overhead**: Minimum cost of the system pipeline without meaningful work

## Skills

- **Benchmark authoring**: Add new Criterion benchmarks for new systems or operations
- **Performance regression detection**: Compare benchmark results across commits
- **Profiling integration**: Use `cargo flamegraph` or `perf` alongside Criterion results
- **Parameterized testing**: Design benchmark matrices that isolate specific scaling factors
- **Statistical interpretation**: Analyze Criterion's confidence intervals and outlier detection

## Sub-Agents

| Sub-Agent | Role |
|-----------|------|
| **Step Profiler** | Measures per-tick simulation throughput across configuration dimensions |
| **Creation Timer** | Benchmarks world initialization including procedural generation |
| **Serialization Profiler** | Measures state snapshot performance for MCTS-relevant workloads |
| **Regression Detector** | Compares benchmark results against baseline to flag performance changes |

## Tools

| Tool | Purpose |
|------|---------|
| `cargo bench -p forge-bench` | Run all benchmarks with Criterion |
| `cargo bench -p forge-bench -- step` | Run only step-related benchmarks |
| `cargo bench -p forge-bench -- --save-baseline <name>` | Save results for later comparison |
| `cargo bench -p forge-bench -- --baseline <name>` | Compare against a saved baseline |
| `cargo clippy --workspace -- -D warnings` | Lint with zero-warning policy |
| `cargo fmt` | Format before committing |
| `open target/criterion/report/index.html` | View detailed HTML benchmark reports |
