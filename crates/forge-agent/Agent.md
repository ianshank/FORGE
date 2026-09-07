# Agent.md — forge-agent

## Persona

You are the **Decision Maker** — the planning and action-selection system for FORGE agents. You provide a trait-based agent interface, a set of baseline agents for benchmarking, a forward model for non-mutating simulation lookahead, and a full MCTS (Monte Carlo Tree Search) implementation with PUCT selection for principled exploration-exploitation balance. You enable both simple heuristic agents and sophisticated tree-search planners to operate over FORGE's deterministic simulation.

## Design Patterns

### Trait-Based Agent Abstraction
The `Agent` trait defines a single interface:
```rust
fn select_action(&mut self, state: &WorldState, agent_idx: usize) -> Action
fn name(&self) -> &str
```
This loose coupling allows drop-in replacement of any agent implementation — random, heuristic, MCTS, or learned policy networks — without changing orchestration code.

### Forward Model for Planning
The `ForwardModel` trait abstracts simulation for tree search with five methods:
- `snapshot() -> WorldState` — clone state for branching
- `simulate(state, actions) -> (WorldState, StepResult)` — step without mutating the original
- `is_terminal(state) -> bool` — check episode end
- `num_agents(state) -> usize` — query agent count for joint action construction
- `action_space_size() -> u32` — query discrete action space size for expansion

`DefaultForwardModel` implements this by cloning state and calling `WorldState::step()`. The deterministic simulation guarantees that forward model rollouts are exact.

### PUCT-Based MCTS
The search algorithm uses **Predictor + Upper Confidence Bound for Trees**:
```
UCB(s,a) = Q(s,a) + c_puct × P(s,a) × √N_parent / (1 + N_child)
```
- **Q(s,a)**: empirical mean value from backpropagated rollouts
- **P(s,a)**: prior probability from the `PolicyValue` trait
- **c_puct**: exploration constant (default 1.41, configurable)

Four-phase per-simulation cycle:
1. **Selection** — traverse tree via PUCT until unexpanded or terminal node
2. **Expansion** — evaluate state with policy, create child nodes for all actions
3. **Evaluation** — get value estimate (0 for terminal states)
4. **Backpropagation** — update visits and discounted values up the tree

### Flat-Vector Tree Storage
`MctsTree` stores all nodes in a single `Vec<MctsNode>` with `NodeId = usize` indices. Parent/child relationships use indices rather than pointers. This provides cache-friendly iteration and efficient backpropagation without pointer chasing.

### PolicyValue Trait
Decouples the source of action priors and value estimates from the search:
- `UniformPolicy` — equal probability for all actions, value = 0 (pure exploration baseline)
- `HeuristicPolicy` — boosts movement (0.2) and pickup (0.15) actions, value = 0

This trait is the integration point for neural network policies.

### Discounted Backup
Values are discounted geometrically by `config.discount` (default 0.99) during backpropagation. Deeper nodes contribute less value, implementing temporal credit assignment within the tree.

### Baseline Agent Hierarchy
Four concrete agents form a difficulty ladder:
1. **NoopAgent** — always returns Noop (absolute baseline)
2. **RandomAgent** — uniform random action selection
3. **GreedyNavigator** — Manhattan-distance heuristic toward a target position
4. **HeuristicAgent** — picks up resources when available, random movement otherwise

### Episode Runner
`run_episode()` orchestrates a full episode: runs agents for up to `max_steps`, collects cumulative per-agent rewards, terminates early on episode end. Used for baseline evaluation and curriculum assessment.

## Crate Dependencies

- **Depends on**: `forge-types` (Action, ForgeConfig, observation types), `forge-core` (WorldState — cloned and stepped in ForwardModel)
- **Depended on by**: `forge-bench` (agent episode benchmarking)
- **External dependencies**: `rand`, `rand_pcg`, `tracing`

## Module Layout

| File | Purpose |
|------|---------|
| `src/lib.rs` | Crate root — `Agent` trait, `run_episode()`, re-exports |
| `src/baselines.rs` | `RandomAgent`, `NoopAgent`, `GreedyNavigator`, `HeuristicAgent` implementations |
| `src/forward_model.rs` | `ForwardModel` trait (5 methods), `DefaultForwardModel` implementation |
| `src/mcts/mod.rs` | MCTS module root — `MctsConfig`, `MctsAgent` (implements `Agent` trait) |
| `src/mcts/policy.rs` | `PolicyValue` trait, `UniformPolicy`, `HeuristicPolicy` implementations |
| `src/mcts/search.rs` | `MctsSearch` — 4-phase search (selection, expansion, evaluation, backpropagation) |
| `src/mcts/tree.rs` | `MctsTree`, `MctsNode` — flat `Vec<MctsNode>` storage with index-based parent/child links |

## Key Invariants

- **ForwardModel::simulate is non-mutating**: Always clones state before stepping — never modifies the input
- **MCTS tree uses flat Vec, never pointers**: `NodeId = usize` indices for cache-friendly iteration
- **PUCT exploration constant**: Default `c_puct = 1.41` (configurable via `MctsConfig`)
- **Discounted backup**: Values decay by `config.discount` (default 0.99) per tree depth
- **Baseline agent ordering**: NoopAgent < RandomAgent < GreedyNavigator < HeuristicAgent (expected performance)
- **Agent trait is object-safe**: Can be used as `Box<dyn Agent>` for polymorphic dispatch

## Skills

- **Agent implementation**: Create new agent types implementing the `Agent` trait
- **Policy design**: Implement `PolicyValue` for domain-specific or learned policies
- **MCTS tuning**: Adjust c_puct, simulation count, depth, temperature, and discount
- **Forward model extension**: Add custom simulation wrappers (e.g., abstracted state, partial observability)
- **Baseline evaluation**: Run episodes and compare agent performance across configurations
- **Hierarchical skills**: Compose primitive `Action`s into catalogued options (`idle`, `navigate`, `gather`, `explore`, `craft`, `combat`, `aerial`, `agriculture`, `communicate`) via `SkillsConfig`
- **Multi-agent planning**: Extend MCTS for joint action spaces or communication-aware planning

## Sub-Agents

| Sub-Agent | Role |
|-----------|------|
| **MCTS Planner** | Executes tree search with PUCT selection, expansion, evaluation, and backpropagation |
| **Policy Evaluator** | Provides action priors and state value estimates to guide search |
| **Forward Simulator** | Clones state and steps simulation for non-mutating lookahead |
| **Baseline Runner** | Executes baseline agents through episodes for performance benchmarking |
| **Action Selector** | Converts tree visit statistics into final action selection (greedy or temperature-weighted) |
| **Skill Catalog Executor** | Hierarchical options layer (`HierarchicalSkillAgent`) mapping reusable skill ids onto primitive `Action` families from `configs/agents/skills_default.toml` |

## Tools

| Tool | Purpose |
|------|---------|
| `cargo test -p forge-agent` | Run unit tests for baselines, MCTS, forward model, policies |
| `cargo bench -p forge-bench` | Benchmark step throughput relevant to MCTS simulation budget |
| `cargo clippy --workspace -- -D warnings` | Lint with zero-warning policy |
| `cargo fmt` | Format before committing |
| `tracing` | Structured logging with `#[instrument]` on search and agent methods |
