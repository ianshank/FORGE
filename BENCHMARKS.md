# FORGE Performance & Determinism Benchmarks

Every number in this document links to a machine-readable report committed
under `benchmarks/baselines/<profile>/`, produced by a harness in this
repository, on hardware recorded inside the report. Published floors are
gated in CI by `tests/python/test_throughput_claim.py`; the zero-allocation
contract is gated by `benchmarks/runner/check_zero_alloc.py --max-bytes 0`.
The published floor is 130,000+ steps/second from Python.

## 1. Profiles

| Profile | Host | Toolchain | Report |
|---|---|---|---|
| `cloud_agent` | Intel Xeon, x86_64 Linux 6.12 (cloud VM) | Rust 1.94.1 (`rust-toolchain.toml`), Python 3.12.3 | `benchmarks/baselines/cloud_agent/` |
| `reference_a` | GitHub Actions `ubuntu-latest` | as CI | `benchmarks/baselines/reference_a/` (allocation audit) |

MSRV is Rust 1.85 (`Cargo.toml` `rust-version`); the Python package supports
3.9+. GPUs are not used by any measurement here.

## 2. Throughput

| Measurement | Value | Report |
|---|---|---|
| Single agent, fixed action (`Move Right`), PyO3 boundary, 100,000 iters | 189,439 steps/s (mean 5.28 μs, p99 6.53 μs) | `cloud_agent/pyo3_step.json` |
| Published floor ("steps/second from Python") | 130,000+ | gated by `test_throughput_claim.py` |
| Rust `WorldState::step`, multi-agent sweep | see `env_steps_per_sec` per `num_agents` row | `cloud_agent/multi_agent_scaling.json` |

The Rust sweep measures whole-world `step()` calls per second inside Rust; do
not compare it to the Python-boundary headline.

To reproduce:

    FORGE_RUN_STEP_THROUGHPUT=1 FORGE_STEP_THROUGHPUT_OUT=/tmp/pyo3_step.json \
      pytest tests/python/test_step_throughput.py -s --no-cov
    make bench-export PROFILE=<profile>

## 3. Zero-allocation hot path

`WorldState::step_into` allocates 0 bytes / 0 blocks after warm-up across
1, 8, 16, 32, 64, and 128 agents for every action variant
(`reference_a/alloc_audit.json`). Reproduce with `make alloc-audit`.

## 4. Determinism

Two environments constructed with the same seed and fed the same action
sequence produce byte-identical observations and identical reward
trajectories.

- Rust: `step_determinism` (proptest over all 22 action variants, compares
  serialized state including the RNG stream) and golden state hashes over
  three seeds — `cargo test -p forge-core`. Aerial morphology has a second
  golden set (`test_golden_aerial_state_hash`) that does not rewrite the
  ground digests.
- Rust ↔ Python surface parity: 1,000-step lockstep of `WorldEnv` against
  `ForgeEnv` — `cargo test -p forge-env-forge`.
- Python: `tests/python/test_determinism.py` compares `obs.tobytes()` and
  rewards over 10,000 steps in CI; the 1,000,000-step run is opt-in:

      FORGE_RUN_LONG_DETERMINISM=1 pytest tests/python/test_determinism.py \
        --determinism-steps 1000000 -v --no-cov

  Result to publish once run: "0 mismatching bytes over N steps".

## 5. API compliance

- Gymnasium: `gymnasium.utils.env_checker.check_env(ForgeGymnasiumEnv())`
- PettingZoo: `pettingzoo.test.parallel_api_test(ForgeParallelEnv(n_agents=2))`

Reproduce with `pip install -e ".[compliance]"` then
`pytest tests/python/test_api_compliance.py -v --no-cov`. Both run as required
CI checks (`api-compliance` job).
