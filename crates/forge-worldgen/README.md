# `forge-worldgen`

> **Architectural Layer**: Tier 1: Domain Primitives

Procedural terrain and world generation using deterministic Simplex/Perlin noise algorithms, biome placement, and resource distribution.

---

## Architectural Role & Dependencies

As defined in the **FORGE Workspace Layering Model** (`docs/architecture.md` Section 4.2):

* **Tier**: `Tier 1: Domain Primitives`
* **Allowed Downstream Consumers (`wrappers`)**: `forge-core`
* **Workspace Dependencies**: `forge-types`

Dependency boundaries are strictly enforced via `deny.toml` `[bans]`. Introducing inverted or unapproved dependencies will cause immediate CI failure.

---

## Key Types & Public API

- `WorldGenerator`: Main procedural generator producing initial grid state.
- `biome`: Biome distribution (forest, plains, desert, mountains).
- `resources`: Mineral, vegetation, and water spawning routines.
- `noise`: Deterministic seeded coherent noise generators.

---

## Feature Flags

- *(None — zero optional feature flags; completely self-contained)*

---

## Usage Example

```rust
use forge_worldgen::WorldGenerator;
use forge_types::config::WorldConfig;

let config = WorldConfig::default();
let generator = WorldGenerator::new(&config);
// generator.generate(...) creates a fully populated simulation world.
```

---

## Testing & Verification

Run tests for this crate:

```bash
cargo test -p forge-worldgen
cargo clippy -p forge-worldgen -- -D warnings
```
