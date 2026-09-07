# `forge-bench`

> **Architectural Layer**: Tier 4: Applications & Runners

Criterion performance microbenchmarks, simulation throughput tests, and DHAT heap allocation profiling harness.

---

## Architectural Role & Dependencies

As defined in the **FORGE Workspace Layering Model** (`docs/architecture.md` Section 4.2):

* **Tier**: `Tier 4: Applications & Runners`
* **Allowed Downstream Consumers (`wrappers`)**: Benchmarking workflows (`cargo bench -p forge-bench`, `make alloc-audit`)
* **Workspace Dependencies**: `forge-agent`, `forge-core`, `forge-types`

Dependency boundaries are strictly enforced via `deny.toml` `[bans]`. Introducing inverted or unapproved dependencies will cause immediate CI failure.

---

## Key Types & Public API

- `env`: Benchmark fixture harness for measuring steps per second.
- `DHAT profiling`: Verifies 0 heap allocations on `WorldState::step_into`.

---

## Feature Flags

- `dhat-heap` - Instruments the DHAT allocator to audit zero-heap-allocation hot paths.

---

## Usage Example

```rust
// Run standard benchmarks:
// cargo bench -p forge-bench

// Run allocation audit:
// cargo run -p forge-bench --features dhat-heap --bin allocation_audit
```

---

## Testing & Verification

Run tests for this crate:

```bash
cargo test -p forge-bench
cargo clippy -p forge-bench -- -D warnings
```
