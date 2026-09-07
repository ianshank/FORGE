# `forge-task`

> **Architectural Layer**: Tier 1: Domain Primitives

Hierarchical task network (HTN), goal evaluation, procedural curriculum generation, and task reward computation.

---

## Architectural Role & Dependencies

As defined in the **FORGE Workspace Layering Model** (`docs/architecture.md` Section 4.2):

* **Tier**: `Tier 1: Domain Primitives`
* **Allowed Downstream Consumers (`wrappers`)**: `forge-core`, `forge-data`, `forge-mangomas`
* **Workspace Dependencies**: `forge-types`

Dependency boundaries are strictly enforced via `deny.toml` `[bans]`. Introducing inverted or unapproved dependencies will cause immediate CI failure.

---

## Key Types & Public API

- `ActiveTask`: State of an agent's current task objective.
- `TaskEvaluator`: Predicate evaluation checking completion or failure conditions.
- `Curriculum`: Multi-stage task progression generator.
- `Difficulty`: Task complexity scoring.

---

## Feature Flags

- *(None — zero optional feature flags; completely self-contained)*

---

## Usage Example

```rust
use forge_task::composer::TaskComposer;
use forge_types::config::ForgeConfig;

let config = ForgeConfig::default();
let composer = TaskComposer::new(&config);
```

---

## Testing & Verification

Run tests for this crate:

```bash
cargo test -p forge-task
cargo clippy -p forge-task -- -D warnings
```
