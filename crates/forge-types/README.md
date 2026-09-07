# `forge-types`

> **Architectural Layer**: Tier 0: Core Foundation

Foundational types, fixed-point math, entity IDs, observation schemas, grid definitions, and configuration structs for the entire FORGE workspace.

---

## Architectural Role & Dependencies

As defined in the **FORGE Workspace Layering Model** (`docs/architecture.md` Section 4.2):

* **Tier**: `Tier 0: Core Foundation`
* **Allowed Downstream Consumers (`wrappers`)**: All workspace crates (Tier 0 through Tier 5)
* **Workspace Dependencies**: None (Foundation Isolation invariant)

Dependency boundaries are strictly enforced via `deny.toml` `[bans]`. Introducing inverted or unapproved dependencies will cause immediate CI failure.

---

## Key Types & Public API

- `ForgeConfig`: Top-level configuration hierarchy with strict deserialization.
- `Action`: Discrete simulation actions (movement, interaction, crafting, communication).
- `Position`: 2D discrete grid coordinates.
- `AgentId`: Strongly typed agent identifier.
- `Tile`: Discrete terrain tile representation.

---

## Feature Flags

- *(None — zero optional feature flags; completely self-contained)*

---

## Usage Example

```rust
use forge_types::config::ForgeConfig;
use forge_types::grid::Position;
use forge_types::Action;

let config = ForgeConfig::default();
let pos = Position::new(10, 20);
let action = Action::MoveNorth;
assert_eq!(pos.x, 10);
```

---

## Testing & Verification

Run tests for this crate:

```bash
cargo test -p forge-types
cargo clippy -p forge-types -- -D warnings
```
