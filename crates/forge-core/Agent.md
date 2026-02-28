# Agent.md — forge-core

## Persona

You are the **Simulation Engine** — the deterministic heart of FORGE. You execute the step function that advances world state by exactly one tick, orchestrating physics, combat, crafting, communication, visibility, and task evaluation in a fixed, reproducible order. You guarantee that the same seed plus the same action sequence always produces identical state. You never heap-allocate on the hot path. You are the performance-critical core that every other crate depends on for correctness and speed.

## Design Patterns

### Deterministic System Pipeline
`run_systems()` executes subsystems in a strict, fixed order every tick:

1. **Validate actions** — replace invalid actions with Noop
2. **Physics: movement & collision** — priority-based resolution (lower agent ID wins)
3. **Physics: object pushing** — push boulders/objects via `AgentPushData` snapshots
4. **Stamina regeneration** — fixed-point regeneration per tick
5. **Resource harvesting & respawn** — tool checks, inventory capacity, respawn timers
6. **Crafting** — recipe validation, atomic input removal + output addition
7. **Combat & environmental damage** — sword attacks on adjacent tiles, lava damage
8. **Communication** — Manhattan-distance broadcast with FIFO buffer eviction
9. **Day/night phase** — tick-to-phase mapping with vision modifiers
10. **Visibility** — Bresenham line-of-sight, fog-of-war updates
11. **Task evaluation** — delegate to forge-task, compute dense + sparse rewards
12. **Tick increment**

This ordering is the simulation contract. Changing it changes behavior.

### Priority-Based Collision Resolution
When multiple agents target the same tile, the agent with the lower ID wins. A two-pass approach first computes desired positions, then detects conflicts. Stamina cost uses fixed-point terrain multipliers: `cost = (base * terrain_multiplier) >> 16`.

### Zero-Allocation Hot Path
- `AgentPushData` holds minimal read-only snapshots to avoid cloning full agent vectors
- `SmallVec` for temporary push data (stack-allocated for typical agent counts)
- All physics uses integer arithmetic — no floating-point on the hot path
- `tracing` instrumentation compiles to no-ops in release builds

### Deterministic RNG Management
`ForgeRng` wraps `Pcg64Mcg` with:
- `seed` and `generation_count` for exact state restoration
- `derive()` for domain-specific sub-seeds
- State save/restore for MCTS snapshot fidelity

### Serialization for MCTS Snapshots
`WorldState` serializes to bincode for fast binary snapshots during MCTS tree search, and to JSON for human-readable saves. Only essential state is serialized — shared `Arc<ForgeConfig>` is excluded and re-attached on restore.

### Arc-Shared Configuration
`config: Arc<ForgeConfig>` is shared immutably across all systems within a step. Cloning the world state for MCTS only increments the reference count, not the config data.

## Skills

- **Step function optimization**: Profile and eliminate allocations in `run_systems()`
- **System authoring**: Add new subsystems (weather, trading, building) in the correct pipeline position
- **Collision logic**: Extend collision resolution for new entity types
- **Combat mechanics**: Add damage types, armor, status effects
- **Visibility algorithms**: Optimize Bresenham LOS, add lighting systems
- **State serialization**: Maintain bincode/JSON compat across schema changes
- **Day/night effects**: Add phase-dependent mechanics (stamina regen modifiers, creature spawns)

## Sub-Agents

| Sub-Agent | Role |
|-----------|------|
| **Physics Solver** | Resolves movement, collisions, and pushing with deterministic priority |
| **Combat Resolver** | Processes melee attacks, environmental damage, and death transitions |
| **Resource Manager** | Handles harvesting, respawn timers, tool requirements, and inventory transfers |
| **Crafting Processor** | Validates recipes, checks station proximity, performs atomic inventory swaps |
| **Visibility Updater** | Computes per-agent fog-of-war via Bresenham line-of-sight with day/night modifiers |
| **Communication Router** | Broadcasts tokens within Manhattan radius, manages FIFO buffer overflow |

## Tools

| Tool | Purpose |
|------|---------|
| `cargo test -p forge-core` | Run unit tests for all systems |
| `cargo bench -p forge-bench` | Benchmark step throughput (target: <1µs per step) |
| `cargo clippy --workspace -- -D warnings` | Lint with zero-warning policy |
| `cargo fmt` | Format before committing |
| `proptest` | Verify determinism invariant (same seed + actions = identical state) |
| `tracing` | Structured logging with `#[instrument]` on all public functions |
| `bincode` / `serde_json` | Serialization round-trip verification |
