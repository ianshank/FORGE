# `forge-env-mc`

> **Architectural Layer**: Tier 1: Domain Primitives

Minecraft environment adapter implementing `forge-env::Env` over a WebSocket connection to the Node.js mineflayer bot (`mc-bot`).

---

## Architectural Role & Dependencies

As defined in the **FORGE Workspace Layering Model** (`docs/architecture.md` Section 4.2):

* **Tier**: `Tier 1: Domain Primitives`
* **Allowed Downstream Consumers (`wrappers`)**: `forge-mc-runner`
* **Workspace Dependencies**: `forge-env`

Dependency boundaries are strictly enforced via `deny.toml` `[bans]`. Introducing inverted or unapproved dependencies will cause immediate CI failure.

---

## Key Types & Public API

- `MinecraftEnv`: Full `Env` trait implementation connecting to `mc-bot`.
- `ProtocolClient`: WebSocket client communicating with `mc-bot`.
- `ActionMap`: Bidirectional mapping between discrete action IDs and mineflayer commands.
- `RewardConfig`: Parses `rewards.toml` and computes the canonical SHA
  folded into `schema_id`. Nested `config_path` / `crafting_config_path`
  files are hashed by content (missing nested files fail closed).
- `BlockEmbeddings`: Observation-layout pin for
  `block_embeddings.toml` — **not** part of today's two-input `schema_id`.

---

## Feature Flags

- `testing` - Enables mock socket endpoints for unit tests without a running Minecraft server.

---

## Usage Example

```rust
use forge_env_mc::config::MinecraftEnvConfig;

let config = MinecraftEnvConfig::default();
// let env = MinecraftEnv::connect(config).await?;
```

---

## Testing & Verification

Run tests for this crate:

```bash
cargo test -p forge-env-mc
cargo clippy -p forge-env-mc -- -D warnings
```
