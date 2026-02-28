# Agent.md — forge-core

## Persona

You are the **Simulation Engine** — the deterministic heart of FORGE. You execute the step function that advances world state by exactly one tick, orchestrating physics, combat, crafting, communication, visibility, and task evaluation in a fixed, reproducible order. You guarantee that the same seed plus the same action sequence always produces identical state. You never heap-allocate on the hot path. You are the performance-critical core that every other crate depends on for correctness and speed.

## Design Patterns

### Deterministic System Pipeline
`run_systems()` in `src/systems.rs` executes subsystems in a strict, fixed order every tick. The numbering matches the source code comments:

1. **Validate actions** — replace invalid actions with Noop (dead agents, out-of-range slots/tokens)
2. **Physics: movement & collision** — priority-based resolution (lower agent ID wins)
2b. **Physics: object pushing** — push boulders/objects via `AgentPushData` snapshots (avoids full agent clone)
3. **Stamina regeneration** — fixed-point regeneration per tick
4. **Resource harvesting & respawn** — tool checks, inventory capacity, respawn timers
5. **Crafting** — recipe validation, station proximity check, atomic input removal + output addition
6. **Combat & environmental damage** — sword attacks on adjacent tiles, lava/water damage
7. **Communication** — Manhattan-distance broadcast with FIFO buffer eviction
8. **Day/night phase** — tick-to-phase mapping (computed before visibility so phase affects vision range)
9. **Visibility** — Bresenham line-of-sight, fog-of-war updates with day/night vision modifiers
10. **Task evaluation** — delegates to `forge_task::evaluator::evaluate_tasks()`, computes dense + sparse rewards
11. **Tick increment**

Observation generation happens in `WorldState::step()` after `run_systems()` returns — it is not part of the system pipeline.

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

## Crate Dependencies

- **Depends on**: `forge-types` (all shared types), `forge-worldgen` (world generation in `WorldState::new()`), `forge-task` (task evaluation in `run_systems()` step 10)
- **Depended on by**: `forge-agent` (forward model clones + steps), `forge-python` (Gymnasium wrapper), `forge-wasm` (browser wrapper), `forge-bench` (performance measurement)
- **External dependencies**: `serde`, `serde_json`, `bincode`, `fixed`, `smallvec`, `rand`, `rand_pcg`, `tracing`

## Module Layout

| File | Purpose |
|------|---------|
| `src/lib.rs` | Crate root — re-exports `WorldState` and public API |
| `src/world.rs` | `WorldState` struct, `new()`, `step()`, `reset()`, `generate_observation()`, serialization (`to_bytes`/`from_bytes`/`to_json`) |
| `src/systems.rs` | `run_systems()` — the deterministic system pipeline orchestrator, action validation |
| `src/physics.rs` | `process_movements()`, `process_pushes()`, `regenerate_stamina()`, collision resolution |
| `src/combat.rs` | `process_combat()` (melee attacks), `apply_environmental_damage()` (lava, water) |
| `src/communication.rs` | `process_communication()` — Manhattan-distance token broadcast with FIFO eviction |
| `src/crafting.rs` | `process_crafting()` — recipe validation, station proximity, atomic inventory swaps |
| `src/day_night.rs` | `compute_day_phase()` — tick-to-phase mapping (dawn/day/dusk/night) |
| `src/resource.rs` | `process_harvesting()`, `tick_respawn()` — resource extraction and regeneration |
| `src/visibility.rs` | `update_visibility()` — Bresenham line-of-sight, fog-of-war, day/night vision modifiers |
| `src/rng.rs` | `ForgeRng` — `Pcg64Mcg` wrapper with seed tracking, derive, save/restore |

## Key Invariants

- **Determinism**: Same seed + same action sequence = identical `WorldState` on any platform
- **Zero allocation on hot path**: `run_systems()` must not heap-allocate (SmallVec for temporaries, integer math only)
- **System ordering is the contract**: Changing the order of steps in `run_systems()` changes simulation behavior
- **Config is immutable during step**: `Arc<ForgeConfig>` is shared read-only across all systems
- **Observation generation happens outside `run_systems()`**: in `WorldState::step()` after the pipeline returns
- **Action padding**: If fewer actions than agents are provided, excess agents receive Noop

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
