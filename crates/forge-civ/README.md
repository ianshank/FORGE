# `forge-civ`

> **Architectural Layer**: Tier 1: Domain Primitives

Grid topology abstractions (square and hexagonal grids) and pathfinding algorithms (A*, Dijkstra) for simulation movement.

---

## Architectural Role & Dependencies

As defined in the **FORGE Workspace Layering Model** (`docs/architecture.md` Section 4.2):

* **Tier**: `Tier 1: Domain Primitives`
* **Allowed Downstream Consumers (`wrappers`)**: `forge-core`
* **Workspace Dependencies**: `forge-types`

Dependency boundaries are strictly enforced via `deny.toml` `[bans]`. Introducing inverted or unapproved dependencies will cause immediate CI failure.

---

## Key Types & Public API

- `GridTopology`: Trait defining neighborhood connectivity and distance metrics.
- `SquareTopology`: Traditional 4-way and 8-way square grid topology.
- `HexTopology`: Axial-coordinate hexagonal grid topology.
- `pathfinding`: Grid pathfinding routines.

---

## Feature Flags

- *(None — zero optional feature flags; completely self-contained)*

---

## Usage Example

```rust
use forge_civ::{GridTopology, SquareTopology};
use forge_types::grid::Position;

let topology = SquareTopology::new(100, 100);
let neighbors = topology.neighbors(Position::new(10, 10));
assert_eq!(neighbors.len(), 4);
```

---

## Testing & Verification

Run tests for this crate:

```bash
cargo test -p forge-civ
cargo clippy -p forge-civ -- -D warnings
```
