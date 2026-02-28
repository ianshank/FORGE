# Agent.md — forge-task

## Persona

You are the **Task Architect** — the system that defines what agents must accomplish and how they are rewarded. You evaluate a recursive DSL of predicates and composition operators against live world state, compute dense and sparse rewards, generate procedural tasks across six difficulty tiers, and manage an adaptive curriculum that shifts difficulty based on agent success rates. You bridge the gap between raw simulation and structured reinforcement learning by turning world state into reward signals.

## Design Patterns

### Recursive Task DSL
Tasks are expressed as trees of `TaskComposition` nodes over atomic `Predicate` leaves:
- **Atom(Predicate)** — single testable condition (AgentAt, AgentHas, AgentNear, TimeElapsed, etc.)
- **And(Vec)** — all subtasks must be satisfied; progress = average
- **Or(Vec)** — at least one must be satisfied; progress = max
- **Sequence(Vec)** — must be satisfied in order; tracks `sequence_index`
- **Before(Box, u64)** — deadline constraint; fails if tick exceeds limit
- **While(Box, Box)** — condition must hold while goal is pursued; fails on violation
- **Without(Box, u32)** — forbidden action constraint

This recursive structure allows arbitrarily complex goals through compositional nesting.

### Two-Channel Predicate Evaluation
Every predicate returns `PredicateResult { satisfied: bool, progress: f32 }`:
- `satisfied` — binary completion check
- `progress` — continuous [0.0, 1.0] metric for reward shaping

Progress metrics are domain-specific: `AgentAt` uses `1.0 - (distance / 100.0)`, `AgentHas` uses `current_count / target_count`, `AgentNear` uses proximity ratio. This enables dense rewards without hand-crafted shaping functions.

### Tiered Procedural Generation
`generate_task()` creates tasks at six difficulty tiers, each introducing progressively complex operators:
- **Tier 1**: Single atomic predicate (navigate or collect)
- **Tier 2**: Two predicates combined with AND
- **Tier 3**: Sequences, OR branches, deadlines
- **Tier 4**: Complex nesting with While, Without, nested deadlines
- **Tier 5**: Multi-agent coordination (requires 2+ agents)
- **Tier 6**: Maximum complexity with deeply nested operators

Reward scales linearly with tier: `reward = base_reward × tier`.

### Adaptive Curriculum Controller
`CurriculumController` manages difficulty through:
- **Rolling-window success tracking**: fixed-size history of episode outcomes
- **Target-driven weight adjustment**: compares success rate against `target_success_rate`
- **Hysteresis deadband**: ±0.05 prevents oscillatory adjustments
- **Warmup period**: no adjustments until minimum episodes completed
- **Tier weight distribution**: probability array `[f32; 6]` normalized to sum to 1.0

### Dense Reward Distribution
Per-tick reward = `progress_delta × reward_scale` distributed equally to all alive agents. Completion bonus = `task.reward × reward_scale`. Dense reward weights are normalized across atoms to sum to ~1.0.

### EvalContext Pattern
`EvalContext` provides an immutable snapshot of world state (agents, tick, grid, objects) for predicate evaluation. This decouples evaluation from mutation — tasks never modify world state.

## Crate Dependencies

- **Depends on**: `forge-types` (Predicate, TaskComposition, ActiveTask, Agent, Grid, Object, config structs)
- **Depended on by**: `forge-core` (called at step 10 of `run_systems()` via `evaluate_tasks()`)
- **External dependencies**: `rand`, `rand_pcg`, `tracing`

## Module Layout

| File | Purpose |
|------|---------|
| `src/lib.rs` | Crate root — re-exports evaluator, generator, curriculum public API |
| `src/predicate.rs` | `EvalContext`, `PredicateResult`, atomic predicate evaluation against world state |
| `src/composer.rs` | `evaluate_composition()` — recursive evaluation of `TaskComposition` trees, `CompositionResult` |
| `src/evaluator.rs` | `evaluate_tasks()` — top-level task evaluation, dense/sparse reward computation, termination logic |
| `src/generator.rs` | `generate_task()` — procedural task generation across 6 difficulty tiers |
| `src/curriculum.rs` | `CurriculumController` — adaptive tier sampling via rolling-window success tracking |
| `src/difficulty.rs` | Difficulty estimation heuristics for auto-tier and step-count inference |

## Key Invariants

- **Evaluation is pure**: Task evaluation never mutates world state — `EvalContext` is read-only
- **Progress values in [0.0, 1.0]**: All `PredicateResult::progress` values are clamped to this range
- **Sequence tracking is monotonic**: `sequence_index` only advances forward, never regresses
- **Dense reward weights normalize to ~1.0**: Per-atom weights across a task sum to approximately 1.0
- **Tier range**: `TaskTier` values are clamped to 1-6 on construction
- **Reward distribution**: Dense per-tick rewards are split equally among alive agents

## Skills

- **Predicate design**: Define new atomic conditions (e.g., AgentCrafted, ZoneControl, ResourceDepleted)
- **Composition authoring**: Build complex task trees using the DSL operators
- **Reward engineering**: Tune dense reward weights and progress metrics for learning efficiency
- **Curriculum tuning**: Adjust target success rate, window size, and adjustment rate
- **Task generation**: Extend procedural generation with new tier templates
- **Difficulty estimation**: Improve step-count and tier estimation heuristics

## Sub-Agents

| Sub-Agent | Role |
|-----------|------|
| **Predicate Evaluator** | Evaluates atomic predicates against world state, returns (satisfied, progress) |
| **Composition Evaluator** | Recursively evaluates composite task trees (And/Or/Sequence/Before/While/Without) |
| **Task Generator** | Procedurally generates tasks at specified tiers with appropriate complexity |
| **Difficulty Estimator** | Infers tier and minimum step count from task composition structure |
| **Curriculum Controller** | Adapts tier sampling distribution based on rolling success rate |
| **Reward Distributor** | Computes per-agent dense and sparse rewards from task progress |

## Tools

| Tool | Purpose |
|------|---------|
| `cargo test -p forge-task` | Run unit tests for predicates, composition, generation, curriculum |
| `cargo clippy --workspace -- -D warnings` | Lint with zero-warning policy |
| `cargo fmt` | Format before committing |
| `proptest` | Verify dense reward weight normalization, curriculum weight stability |
| `tracing` | Structured logging with `#[instrument]` on evaluation and generation functions |
