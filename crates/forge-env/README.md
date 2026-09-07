# `forge-env`

> **Architectural Layer**: Tier 0: Core Foundation

Environment abstraction traits (`Env`, `FlatObsEnv`) and specification descriptors (`ObsSpec`, `ActionSpec`) providing a standard interface for RL agents.

---

## Architectural Role & Dependencies

As defined in the **FORGE Workspace Layering Model** (`docs/architecture.md` Section 4.2):

* **Tier**: `Tier 0: Core Foundation`
* **Allowed Downstream Consumers (`wrappers`)**: `forge-env-mc`, `forge-env-forge`, `forge-mc-runner`
* **Workspace Dependencies**: None (Foundation Isolation invariant)

Dependency boundaries are strictly enforced via `deny.toml` `[bans]`. Introducing inverted or unapproved dependencies will cause immediate CI failure.

---

## Key Types & Public API

- `Env`: Core environment trait requiring `reset` and `step`.
- `FlatObsEnv`: Specialized trait for environments returning flat 1D observation vectors.
- `StepOutput`: Result of taking an environment step (observation, reward, done, truncated, info).
- `ObsSpec` / `ActionSpec`: Observation and action space descriptors.

---

## Feature Flags

- *(None — zero optional feature flags; completely self-contained)*

---

## Usage Example

```rust
use forge_env::{Env, StepOutput};

// Any environment implementing `Env` can be driven generically:
// fn evaluate_policy<E: Env>(env: &mut E, steps: usize) { ... }
```

---

## Testing & Verification

Run tests for this crate:

```bash
cargo test -p forge-env
cargo clippy -p forge-env -- -D warnings
```
