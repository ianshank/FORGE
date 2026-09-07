# `forge-env-forge`

> **Architectural Layer**: Tier 3: Agents & Interfaces

Adapter bridging FORGE's native `WorldState` into the generic `forge-env::Env` trait interface.

---

## Architectural Role & Dependencies

As defined in the **FORGE Workspace Layering Model** (`docs/architecture.md` Section 4.2):

* **Tier**: `Tier 3: Agents & Interfaces`
* **Allowed Downstream Consumers (`wrappers`)**: Generic RL runners and benchmark harnesses
* **Workspace Dependencies**: `forge-core`, `forge-env`, `forge-types`

Dependency boundaries are strictly enforced via `deny.toml` `[bans]`. Introducing inverted or unapproved dependencies will cause immediate CI failure.

---

## Key Types & Public API

- `WorldEnv`: Wraps `WorldState` to implement `forge_env::Env`.
- `FlatForgeEnv`: Adapter providing flattened 1D observation vectors for neural networks.
- `ObsFlattener`: Converts structured `Observation` into contiguous float vectors.

---

## Feature Flags

- *(None — zero optional feature flags; completely self-contained)*

---

## Usage Example

```rust
use forge_env_forge::WorldEnv;
use forge_types::config::ForgeConfig;
use forge_env::Env;

let mut env = WorldEnv::new(ForgeConfig::default());
let obs = env.reset(Some(42)).unwrap();
```

---

## Testing & Verification

Run tests for this crate:

```bash
cargo test -p forge-env-forge
cargo clippy -p forge-env-forge -- -D warnings
```
