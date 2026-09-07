# `forge-python`

> **Architectural Layer**: Tier 3: Agents & Interfaces

PyO3 native extension (`forge_env`) exposing FORGE simulation engines and Gymnasium/PettingZoo environments directly to Python.

---

## Architectural Role & Dependencies

As defined in the **FORGE Workspace Layering Model** (`docs/architecture.md` Section 4.2):

* **Tier**: `Tier 3: Agents & Interfaces`
* **Allowed Downstream Consumers (`wrappers`)**: Python runtime (`import forge_env`)
* **Workspace Dependencies**: `forge-core`, `forge-types`

Dependency boundaries are strictly enforced via `deny.toml` `[bans]`. Introducing inverted or unapproved dependencies will cause immediate CI failure.

---

## Key Types & Public API

- `ForgeEnvPy`: Python class wrapping `WorldState` conforming to Gymnasium API.
- `numpy_zero_copy`: Zero-copy NumPy array views over observation memory buffers.

---

## Feature Flags

- *(None — zero optional feature flags; completely self-contained)*

---

## Usage Example

```rust
// Built using Maturin into a Python wheel:
// maturin develop --release
```

---

## Testing & Verification

Run tests for this crate:

```bash
cargo test -p forge-python
cargo clippy -p forge-python -- -D warnings
```
