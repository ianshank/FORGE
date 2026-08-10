---
license: apache-2.0
task_categories:
  - reinforcement-learning
language:
  - en
pretty_name: FORGE Gridworld Trajectories
size_categories:
  - 1M<n<10M
tags:
  - forge
  - gridworld
  - mcts
  - offline-rl
  - trajectories
  - simulation
configs:
  - config_name: default
    data_files:
      - split: train
        path: "data/data-*.parquet"
---

# FORGE Gridworld Trajectories

Deterministic gridworld trajectories from the
[FORGE](https://github.com/ianshank/FORGE) simulator (Fast Open-source
Runtime for Generalist Environments) — a Rust simulation platform for
training and evaluating AI agents, with procedurally generated worlds,
crafting, combat, multi-agent cooperation, and a composable task
curriculum.

One row per **(step, agent)** pair. Episodes are driven either by FORGE's
built-in MCTS planner or by a seeded uniform-random policy (baseline
data), across a cross-product of world sizes, agent counts, and
task-curriculum tiers.

Each episode is assigned one procedurally generated task at the cell's
curriculum tier (from a seed-derived RNG stream). Rewards are dense task
progress deltas plus a completion bonus; completing the task terminates
the episode. Reward signal is intentionally **sparse**: many episodes
never touch their task and carry all-zero rewards — a realistic setting
for offline-RL and exploration research.

**MCTS caveat**: the `mcts` policy plans over a cloned-world forward model
with heuristic priors. Where task signal is beyond the search horizon it
collapses to the prior's argmax, so its action diversity is much lower
than the `random` cells. Treat it as "FORGE's built-in planner", not an
oracle expert.

## Reproducibility

Every episode is **byte-identically reproducible**: FORGE uses fixed-point
arithmetic and PCG RNG, so `(scenario_id, seed)` uniquely determines the
full episode. Each configuration cell owns a disjoint, contiguous seed
block derived from its position in the full default cell grid — a
`--cells` filter keeps every cell's original block, so regenerating a
subset (with the default axis lists and the published
`--episodes-per-cell`) reproduces exactly the published episodes:

```bash
cargo run --release -p forge-data --features hf --bin forge-gen-dataset -- \
    --out ./regen --cells <scenario_id> --episodes-per-cell <N>
```

## Schema (v1)

| Column | Type | Description |
|---|---|---|
| `seed` | uint64 | Episode seed (world generation + policy RNG) |
| `scenario_id` | string | Configuration-cell label, e.g. `square-64-a2-t2-mcts` |
| `tick` | uint64 | Simulation tick within the episode |
| `agent_idx` | uint32 | Agent index (0 = policy-driven, others Noop) |
| `action` | uint32 | Discrete action id (see FORGE `Action::from_discrete`) |
| `reward` | float32 | Per-step reward for this agent |
| `terminated` | bool | Episode terminated at this step |
| `truncated` | bool | Episode truncated at this step |
| `reasoning` | string? | Optional policy reasoning (null for MCTS/random) |
| `confidence` | float32 | Policy confidence (0 when not applicable) |
| `decision_time_ms` | uint64 | Per-decision wall time |
| `agent_health` | float32 | Agent health after the step |
| `agent_stamina` | float32 | Agent stamina after the step |
| `agent_position_x` | uint16 | Agent x position |
| `agent_position_y` | uint16 | Agent y position |
| `agent_battery` | float32 | Agent battery level |

Full per-tile grid observations are intentionally **not** included (they
are ~1000× larger); regenerate locally from `(scenario_id, seed)` if you
need them.

## Configuration cells

`scenario_id` follows `square-{world_size}-a{agents}-t{tier}-{policy}`:

- **world_size** ∈ {32, 64, 128} — square procedural worlds
- **agents** ∈ {1, 2, 4} — agent 0 acts, others are Noop observers
- **tier** ∈ {1, 2, 3} — task-curriculum difficulty tier
- **policy** ∈ {mcts, random} — MCTS planner (expert) vs seeded uniform-random (baseline)

Hex-grid cells are deferred to a future version (the base discrete action
encoding does not cover hex movement).

## Loading

```python
from datasets import load_dataset

ds = load_dataset("ianshank/forge-gridworld-trajectories", split="train")
episodes = ds.to_pandas().groupby(["scenario_id", "seed"])
```

## Provenance

- Source: [ianshank/FORGE](https://github.com/ianshank/FORGE) @ `{{GIT_SHA}}`
- Generator: `forge-gen-dataset` (see `crates/forge-data/src/bin/forge_gen_dataset.rs`)
- Invocation: `{{CLI_ARGS}}`
- Rows: `{{ROW_COUNT}}`
- License: Apache-2.0 (same as the simulator)
