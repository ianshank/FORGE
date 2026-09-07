# `forge-data`

> **Architectural Layer**: Tier 4: Applications & Runners

Offline RL datasets, Minari and MineRL format adapters, expert demonstration generators, and dataset validation utilities.

---

## Architectural Role & Dependencies

As defined in the **FORGE Workspace Layering Model** (`docs/architecture.md` Section 4.2):

* **Tier**: `Tier 4: Applications & Runners`
* **Allowed Downstream Consumers (`wrappers`)**: `forge-cloud`
* **Workspace Dependencies**: `forge-agent`, `forge-core`, `forge-observability`, `forge-replay`, `forge-task`, `forge-types`

Dependency boundaries are strictly enforced via `deny.toml` `[bans]`. Introducing inverted or unapproved dependencies will cause immediate CI failure.

---

## Key Types & Public API

- `DataLoader`: Batch iteration over offline trajectory datasets.
- `DemoGenerator`: Generates scripted expert trajectories for imitation learning.
- `MinariAdapter`: Converts TrajectoryV2 records into standard Minari HDF5 datasets.

---

## Feature Flags

- `hf` - Hugging Face Hub dataset publishing and streaming.

---

## Usage Example

```rust
use forge_data::loader::TrajectoryLoader;

// Load recorded simulation episodes for offline model training.
```

---

## Testing & Verification

Run tests for this crate:

```bash
cargo test -p forge-data
cargo clippy -p forge-data -- -D warnings
```
