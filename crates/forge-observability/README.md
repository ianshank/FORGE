# `forge-observability`

> **Architectural Layer**: Tier 0: Core Foundation

Centralized tracing, logging, and metrics initialization for FORGE binaries and services, supporting both text and JSON formats.

---

## Architectural Role & Dependencies

As defined in the **FORGE Workspace Layering Model** (`docs/architecture.md` Section 4.2):

* **Tier**: `Tier 0: Core Foundation`
* **Allowed Downstream Consumers (`wrappers`)**: `forge-server`, `forge-mc-runner`, `forge-eval`, `forge-data`
* **Workspace Dependencies**: None (Foundation Isolation invariant)

Dependency boundaries are strictly enforced via `deny.toml` `[bans]`. Introducing inverted or unapproved dependencies will cause immediate CI failure.

---

## Key Types & Public API

- `init_tracing`: Initializes global tracing subscriber based on environment options.
- `TracingOptions`: Configuration options for log levels and output format.
- `LogFormat`: Enum of supported log formats (`Text`, `Json`).
- `LOG_FORMAT_ENV`: Environment variable `FORGE_LOG_FORMAT`.

---

## Feature Flags

- *(None — zero optional feature flags; completely self-contained)*

---

## Usage Example

```rust
use forge_observability::{init_tracing, TracingOptions};

init_tracing(TracingOptions::new("info"));
tracing::info!("FORGE service initialized");
```

---

## Testing & Verification

Run tests for this crate:

```bash
cargo test -p forge-observability
cargo clippy -p forge-observability -- -D warnings
```
