# `forge-server`

> **Architectural Layer**: Tier 3: Agents & Interfaces

High-performance Axum WebSocket and REST HTTP server providing live browser streaming and external API control of the FORGE world.

---

## Architectural Role & Dependencies

As defined in the **FORGE Workspace Layering Model** (`docs/architecture.md` Section 4.2):

* **Tier**: `Tier 3: Agents & Interfaces`
* **Allowed Downstream Consumers (`wrappers`)**: Root integration tests (`forge-integration-tests`)
* **Workspace Dependencies**: `forge-core`, `forge-observability`, `forge-types`

Dependency boundaries are strictly enforced via `deny.toml` `[bans]`. Introducing inverted or unapproved dependencies will cause immediate CI failure.

---

## Key Types & Public API

- `ForgeServer`: Axum server instance managing client sessions and simulation stepping.
- `ServerConfig`: Network bind address, port, tick rate, and CORS policy.
- `WsMessage`: Bidirectional protocol messages for real-time visualization.

---

## Feature Flags

- *(None — zero optional feature flags; completely self-contained)*

---

## Usage Example

```rust
use forge_server::config::ServerConfig;

let config = ServerConfig::default();
// forge_server::run(config).await?;
```

---

## Testing & Verification

Run tests for this crate:

```bash
cargo test -p forge-server
cargo clippy -p forge-server -- -D warnings
```
