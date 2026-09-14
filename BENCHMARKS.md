# FORGE Performance & Determinism Benchmarks

Unverifiable performance claims degrade trust in a simulation engine, so this
document holds itself to a rule: **every measured value below is read from a
machine-readable report committed to this repository**, produced by a harness in
this repository, on hardware recorded inside the report itself. Nothing
measured here is typed in by hand, and there are no placeholder rows — a
measurement that has not been taken is listed as not taken.

Toolchain floors in §1 (Rust `1.94.1`, MSRV `1.85`, Python 3.9+) are not
benchmark measurements. They are read from `rust-toolchain.toml`,
`Cargo.toml`'s `rust-version`, and `pyproject.toml`'s `requires-python`.

Three of these claims are enforced rather than asserted. CI fails if a published
throughput floor exceeds the committed measurement
(`tests/python/test_throughput_claim.py`), if the hot path allocates a single
byte (`benchmarks/runner/check_zero_alloc.py --max-bytes 0`), or if determinism
or upstream API compliance regress (`tests/python/test_determinism.py`,
`tests/python/test_api_compliance.py`).

## 1. Profiles and environment

Measurements are **not** interchangeable between profiles. Each report records
its own hardware, and a number may only be cited alongside the profile it came
from.

| Profile | Host | Recorded in |
| :--- | :--- | :--- |
| `cloud_agent` | Intel Xeon, x86_64, Linux 6.12, Python 3.12.3 | `benchmarks/baselines/cloud_agent/` |
| `reference_a` | GitHub Actions `ubuntu-latest`, x86_64 Linux | `benchmarks/baselines/reference_a/` |
| `reference_b` | User workstation. **Not populated** — the directory ships with only a `.gitkeep` | `benchmarks/baselines/reference_b/` |

Toolchain: Rust `1.94.1` (pinned in `rust-toolchain.toml`), built `--release`.
The declared MSRV floor is Rust `1.85` (`Cargo.toml`'s `rust-version`). The
Python package supports 3.9+; the committed measurements were taken on 3.12.3.

No measurement in this document uses a GPU. FORGE's simulation core is
CPU-bound integer arithmetic; GPU hardware affects model training, which is out
of scope here.

## 2. Throughput

### 2.1 Python boundary (single agent)

The headline number: `env.step(action)` round-trips through the real PyO3
wrapper that `examples/train_ppo.py` uses, driving `Action::Move(Right)` rather
than the cheaper `Noop` path.

| Metric | Value |
| :--- | :--- |
| Throughput | 189,439 steps/s |
| Mean latency | 5.28 µs |
| Median latency | 5.28 µs |
| p99 latency | 6.53 µs |
| Iterations | 100,000 after 1,000 warm-up |

Source: `benchmarks/baselines/cloud_agent/pyo3_step.json` (commit `555b249`).

The floor published in `README.md` and `docs/CHARTER.md` is **130,000+
steps/second from Python**, deliberately below the measurement so the claim
survives host-to-host variance. `tests/python/test_throughput_claim.py` fails
the build if any published floor rises above the committed report.

To reproduce:

```bash
maturin develop
FORGE_RUN_STEP_THROUGHPUT=1 \
FORGE_STEP_THROUGHPUT_OUT=benchmarks/baselines/<profile>/pyo3_step.json \
  pytest tests/python/test_step_throughput.py -s --no-cov
```

### 2.2 Multi-agent scaling (in-process Rust)

Criterion driving `WorldState::step` directly on a 128×128 world. **These are
not comparable to §2.1** — they never cross the Python boundary — and the
agent-normalised column must never be quoted as the Python headline.

`env_steps_per_sec` counts whole-world steps; `agent_steps_per_sec` is that
multiplied by the agent count, and is the figure that shows batching pays off.

| Agents | Square: env steps/s | Square: agent steps/s | Hex: env steps/s | Hex: agent steps/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 85,592 | 85,592 | 72,011 | 72,011 |
| 8 | 53,989 | 431,912 | 20,965 | 167,721 |
| 16 | 36,778 | 588,454 | 11,364 | 181,818 |
| 32 | 21,614 | 691,633 | 6,124 | 195,974 |
| 64 | 12,228 | 782,593 | 3,008 | 192,530 |
| 128 | 6,358 | 813,769 | 1,497 | 191,588 |

Source: `benchmarks/baselines/cloud_agent/multi_agent_scaling.json`.

Square-grid throughput scales close to linearly in agents; hex topology
saturates around 16 agents because its neighbour lookups dominate. Both are
recorded because FORGE ships both topologies.

To reproduce:

```bash
make bench-export PROFILE=<profile>
```

### 2.3 Not measured

A PPO-rollout throughput figure is **not published**, because no committed
harness produces one. Adding it means adding a reporting mode to
`examples/train_ppo_cleanrl.py` and committing its JSON, not estimating a
number.

## 3. Memory: the zero-allocation hot path

`WorldState::step_into(&mut StepResult)` performs **zero heap allocations after
warm-up**. This is a hard CI gate, not an aspiration.

| Property | Value |
| :--- | :--- |
| Rows measured | 72 (12 action variants × 6 agent counts) |
| Agent counts | 1, 8, 16, 32, 64, 128 |
| Iterations per row | 10,000, after 1,024 warm-up |
| `total_bytes` | 0 across every row |
| `total_blocks` | 0 across every row |

Source: `benchmarks/baselines/reference_a/alloc_audit.json`, gated on every PR by
the `alloc-audit` job.

The convenience `step()` wrapper *does* allocate a fresh `StepResult` per call —
pass a reused buffer via `step_into` for the zero-allocation contract. Wire-bound
environments such as `forge-env-mc::MinecraftEnv` are an on-the-record exception
(`docs/CHARTER.md`, Deliberate Exception 1): they reuse caller buffers, but their
WebSocket I/O and JSON parsing allocate.

To reproduce:

```bash
make alloc-audit
```

## 4. The determinism guarantee

Two environments constructed with the same seed and fed the same action sequence
produce **byte-identical observations and exactly equal rewards**. Not
approximately equal — the comparison is on raw bytes, so a difference in the last
mantissa bit fails.

The guarantee is defended at three levels, each independently runnable:

| Level | What it compares | Where |
| :--- | :--- | :--- |
| Rust core | Serialized world state including the RNG stream, over a property-test-generated action sequence, plus golden state hashes across three seeds | `cargo test -p forge-core` |
| Rust env trait | 1,000-step lockstep of `WorldEnv` against `ForgeEnv`, comparing rewards and termination flags | `cargo test -p forge-env-forge` |
| Python boundary | `ndarray.tobytes()` of every observation component and exact reward equality, across episode boundaries | `pytest tests/python/test_determinism.py` |

The Python level closes a real gap: a determinism bug introduced in observation
conversion or space fitting would not move a single Rust hash while corrupting
every Python rollout.

Depth is configurable rather than fixed, so the same code backs both a fast PR
gate and a release soak:

```bash
# Default depth (10,000 steps), as the api-compliance CI job runs it
pytest tests/python/test_determinism.py -v --no-cov

# Release soak
make api-compliance-soak DETERMINISM_STEPS=1000000
```

Report results as "0 diverging bytes over N steps". A percentage would be
meaningless: the assertion is exact equality, so the only passing value is zero.

## 5. API compliance

FORGE is tested against the official compliance suites of both major Python RL
frameworks, by running those suites — not by re-implementing their checks.

| Framework | Suite | Result |
| :--- | :--- | :--- |
| Gymnasium | `gymnasium.utils.env_checker.check_env` | Passes |
| PettingZoo | `pettingzoo.test.parallel_api_test` | Passes |

Verified against Gymnasium 1.3.0 and PettingZoo 1.27.0. `ForgeGymnasiumEnv`
subclasses `gymnasium.Env` and `ForgeParallelEnv` subclasses
`pettingzoo.ParallelEnv`; observations are fitted to their declared spaces, so
`observation_space.contains(obs)` holds for every returned observation rather
than merely for values that look plausible.

`gymnasium.make` is supported through explicit registration, which follows
FORGE's no-import-side-effects rule (`docs/CHARTER.md`, Invariant 1):

```python
import gymnasium as gym
from forge_env import register_envs

register_envs()
env = gym.make("Forge-v0")
```

To reproduce:

```bash
pip install -e '.[compliance]'
pytest tests/python/test_api_compliance.py -v --no-cov
```

Note that `forge_env.utils.check_env` is a lightweight in-repo shape check and
is **not** evidence of compliance. Only the upstream suites above are.

## 6. Regenerating this document's evidence

Every command below writes a committed artefact. Run them on one machine, commit
the JSON and any prose change together, and never mix profiles.

```bash
PROFILE=reference_b   # or cloud_agent to refresh

maturin develop
FORGE_RUN_STEP_THROUGHPUT=1 \
FORGE_STEP_THROUGHPUT_OUT=benchmarks/baselines/$PROFILE/pyo3_step.json \
  pytest tests/python/test_step_throughput.py -s --no-cov

make bench-export PROFILE=$PROFILE

cargo run -p forge-bench --bin allocation_audit --features dhat-heap --release -- \
  --warmup 1024 --iters 10000 --agents 1,8,16,32,64,128 \
  --out benchmarks/baselines/$PROFILE/alloc_audit.json
python3 benchmarks/runner/check_zero_alloc.py \
  --input benchmarks/baselines/$PROFILE/alloc_audit.json --max-bytes 0
```

See `benchmarks/baselines/README.md` for the per-profile rules on which numbers
may be committed where.
