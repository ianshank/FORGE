# `forge-mangomas`

> **Architectural Layer**: Tier 4: Applications & Runners

MangoMAS multi-agent benchmark harness, curriculum progression manager, and distributed scenario collection engine.

---

## Architectural Role & Dependencies

As defined in the **FORGE Workspace Layering Model** (`docs/architecture.md` Section 4.2):

* **Tier**: `Tier 4: Applications & Runners`
* **Allowed Downstream Consumers (`wrappers`)**: MangoMAS training pipelines
* **Workspace Dependencies**: `forge-agent`, `forge-core`, `forge-integration-layer`, `forge-task`, `forge-types`

Dependency boundaries are strictly enforced via `deny.toml` `[bans]`. Introducing inverted or unapproved dependencies will cause immediate CI failure.

---

## Key Types & Public API

- `BatchRunner`: Parallel scenario execution worker pool.
- `CurriculumManager`: Adaptive scenario difficulty scheduler.
- `SwarmController`: Multi-agent policy coordination.

---

## Feature Flags

- *(None — zero optional feature flags; completely self-contained)*

---

## Usage Example

```rust
use forge_mangomas::config::MangoMasConfig;

let config = MangoMasConfig::default();
```

---

## Testing & Verification

Run tests for this crate:

```bash
cargo test -p forge-mangomas
cargo clippy -p forge-mangomas -- -D warnings
```
