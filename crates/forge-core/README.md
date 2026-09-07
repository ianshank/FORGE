# `forge-core`

> **Architectural Layer**: Tier 2: Engine & Cognitive

Core deterministic simulation engine executing the 13-phase simulation pipeline, zero-allocation physics step, and state transitions.

---

## Architectural Role & Dependencies

As defined in the **FORGE Workspace Layering Model** (`docs/architecture.md` Section 4.2):

* **Tier**: `Tier 2: Engine & Cognitive`
* **Allowed Downstream Consumers (`wrappers`)**: `forge-agent`, `forge-bench`, `forge-data`, `forge-env-forge`, `forge-eval`, `forge-mangomas`, `forge-python`, `forge-replay`, `forge-server`, `forge-wasm`
* **Workspace Dependencies**: `forge-civ`, `forge-task`, `forge-types`, `forge-worldgen`

Dependency boundaries are strictly enforced via `deny.toml` `[bans]`. Introducing inverted or unapproved dependencies will cause immediate CI failure.

---

## Key Types & Public API

- `WorldState`: Primary simulation state holding grid, agents, entities, and scratch buffers.
- `WorldState::step`: Advances simulation by one discrete tick.
- `WorldState::step_into`: Zero-heap-allocation step using pre-allocated buffers.
- `WorldState::reset`: Resets world to initial state with new seed.

---

## Feature Flags

- *(None — zero optional feature flags; completely self-contained)*

---

## Usage Example

```rust
use forge_core::WorldState;
use forge_types::config::ForgeConfig;
use forge_types::Action;

let mut world = WorldState::new(ForgeConfig::default());
world.reset(42);
let result = world.step(&[Action::MoveNorth]);
```

---

## Testing & Verification

Run tests for this crate:

```bash
cargo test -p forge-core
cargo clippy -p forge-core -- -D warnings
```
