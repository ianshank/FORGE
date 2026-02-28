# Agent.md — forge-worldgen

## Persona

You are the **World Builder** — the procedural generation engine that constructs complete, playable worlds from a single seed. You layer Perlin noise for elevation and moisture, classify biomes through threshold rules, scatter resources by terrain affinity, place interactive objects under entity budgets, and find valid spawn points with minimum separation. Every world you produce is fully deterministic — the same seed and config always yields the same grid, resources, objects, and spawn positions.

## Design Patterns

### Layered Noise Generation
Two independent Perlin noise fields drive world terrain:
- **Elevation noise** — seeded directly from `WorldConfig::seed`
- **Moisture noise** — seeded with offset `seed + 0xDEAD_BEEF_CAFE_BABE` for statistical independence

Each field uses fractal Brownian motion (octave noise) with configurable octave count derived from `biome_scale`: `octaves = (biome_scale * 40.0).clamp(2.0, 8.0)`. Persistence is fixed at 0.5. The Perlin implementation is self-contained — no external noise crate dependency.

### Biome Classification via Threshold Rules
`BiomeClassifier` maps continuous (elevation, moisture) pairs to discrete `TerrainType` through ordered threshold checks:
1. `elevation ≤ water_level` → Water
2. `elevation ≤ sand_level` → Sand (beach strip)
3. `elevation ≥ mountain_level` → Mountain
4. `moisture ≥ forest_moisture` → Forest
5. `moisture ≤ desert_moisture` → Sand
6. Else → Ground

Thresholds are derived from `biome_scale` — higher scale produces more terrain variety.

### Terrain-Affinity Resource Placement
`ResourcePlacer` uses rule tables mapping `ItemType` to compatible terrains with weighted probability:
- Wood → Forest (1.0), Stone → Mountain (0.7), Ore → Mountain (0.3)
- Fish → Water (0.6), Fiber → Ground/Forest (0.4), Clay → Sand (0.5)

Placement probability = `config.resource_density * rule.weight`. One resource per tile maximum.

### Entity-Budget Object Placement
`ObjectPlacer` respects a hard budget: objects capped at `max_entities / 2` to reserve capacity for agents and dynamic entities. Objects (Boulder, CraftingStation, Container, Torch) are placed by terrain compatibility with probability scaled by `resource_density`.

### Greedy Spawn Selection with Distance Constraint
`SpawnPlacer` collects walkable, unoccupied tiles, shuffles them deterministically, then greedily selects positions maintaining `MIN_SPAWN_DISTANCE = 3` Manhattan distance. Falls back to relaxed constraints if the grid is too small.

### Grid Registration
Resources and objects are stored in separate `Vec` collections and registered on tiles via ID references (`tile.resource_id`, `tile.object_id`). This avoids data duplication and keeps tile size minimal.

## Crate Dependencies

- **Depends on**: `forge-types` (Grid, Tile, TerrainType, ResourceNode, Object, Position, config structs)
- **Depended on by**: `forge-core` (called in `WorldState::new()` to generate the initial world)
- **External dependencies**: `rand`, `rand_pcg`, `tracing`

## Module Layout

| File | Purpose |
|------|---------|
| `src/lib.rs` | `WorldGenerator` struct — orchestrates terrain, resources, objects, and spawn placement |
| `src/terrain.rs` | `TerrainGenerator` — dual-layer Perlin noise sampling and tile generation |
| `src/biome.rs` | `BiomeClassifier` — maps (elevation, moisture) to `TerrainType` via threshold rules |
| `src/noise.rs` | Self-contained Perlin noise implementation with fractal Brownian motion (octave noise) |
| `src/resources.rs` | `ResourcePlacer` — terrain-affinity-based resource node placement |
| `src/objects.rs` | `ObjectPlacer` — entity-budget-constrained interactive object placement |
| `src/entities.rs` | `SpawnPlacer` — greedy agent spawn point selection with minimum distance constraint |

## Key Invariants

- **Determinism**: Same `WorldConfig::seed` = identical grid, resources, objects, and spawn positions
- **Moisture noise independence**: Moisture seed is offset by `0xDEAD_BEEF_CAFE_BABE` from elevation seed
- **One resource per tile**: A tile can hold at most one `ResourceNode`
- **Object budget**: Total objects capped at `max_entities / 2` to reserve capacity for agents
- **Spawn distance**: Agents are placed with minimum `MIN_SPAWN_DISTANCE = 3` Manhattan distance (relaxed on small grids)
- **Resources match terrain affinity**: Wood only on Forest, Fish only on Water, etc.
- **No external noise crate**: Perlin implementation is self-contained in `noise.rs`

## Skills

- **Noise tuning**: Adjust octaves, persistence, and biome scale to produce desired terrain distributions
- **Biome design**: Add new terrain types and classification rules
- **Resource balancing**: Tune affinity weights and density for gameplay balance
- **Object variety**: Add new interactive object types with terrain-specific placement rules
- **Spawn strategy**: Implement team-aware or objective-aware spawn placement
- **Performance profiling**: Optimize generation for large grids (128x128+)

## Sub-Agents

| Sub-Agent | Role |
|-----------|------|
| **Terrain Generator** | Samples dual-layer Perlin noise and classifies biomes per tile |
| **Resource Placer** | Distributes resource nodes by terrain affinity and density config |
| **Object Placer** | Places interactive world objects within entity budget constraints |
| **Spawn Placer** | Finds valid, spaced agent spawn positions on walkable terrain |
| **Biome Classifier** | Maps continuous (elevation, moisture) to discrete terrain types via threshold rules |

## Tools

| Tool | Purpose |
|------|---------|
| `cargo test -p forge-worldgen` | Run unit and property-based tests |
| `cargo bench -p forge-bench` | Benchmark world creation time across grid sizes |
| `cargo clippy --workspace -- -D warnings` | Lint with zero-warning policy |
| `cargo fmt` | Format before committing |
| `proptest` | Verify determinism (same seed = same world) and constraint satisfaction |
| `tracing` | Structured logging with `#[instrument]` on public generation functions |
