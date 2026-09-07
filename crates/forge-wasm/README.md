# `forge-wasm`

> **Architectural Layer**: Tier 3: Agents & Interfaces

WebAssembly bindings enabling the full FORGE simulation engine to run client-side in web browsers with zero backend dependencies.

---

## Architectural Role & Dependencies

As defined in the **FORGE Workspace Layering Model** (`docs/architecture.md` Section 4.2):

* **Tier**: `Tier 3: Agents & Interfaces`
* **Allowed Downstream Consumers (`wrappers`)**: Browser JavaScript runtime (`web/index.html`)
* **Workspace Dependencies**: `forge-core`, `forge-types`

Dependency boundaries are strictly enforced via `deny.toml` `[bans]`. Introducing inverted or unapproved dependencies will cause immediate CI failure.

---

## Key Types & Public API

- `ForgeWasmEnv`: wasm-bindgen exported class providing `step`, `reset`, and state queries.
- `render_ascii`: Browser-accessible ASCII terminal renderer.

---

## Feature Flags

- *(None — zero optional feature flags; completely self-contained)*

---

## Usage Example

```rust
use forge_wasm::ForgeWasmEnv;

let mut env = ForgeWasmEnv::try_new("{}").unwrap();
let state = env.reset(Some(42));
```

---

## Testing & Verification

Run tests for this crate:

```bash
cargo test -p forge-wasm
cargo clippy -p forge-wasm -- -D warnings
```
