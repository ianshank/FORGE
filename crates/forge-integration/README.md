# `forge-integration`

> **Maturity**: `[Research]` (Research Stack Component — cognitive cross-layer orchestration)  
> **Architectural Layer**: Tier 3: Agents & Interfaces

Cross-primitive orchestration layer harmonizing cognitive, social, task, and memory subsystems into a unified agent workflow.

---

## Architectural Role & Dependencies

As defined in the **FORGE Workspace Layering Model** (`docs/architecture.md` Section 4.2):

* **Tier**: `Tier 3: Agents & Interfaces`
* **Allowed Downstream Consumers (`wrappers`)**: `forge-mangomas`
* **Workspace Dependencies**: `forge-cognitive`, `forge-memory`, `forge-social`, `forge-types`

Dependency boundaries are strictly enforced via `deny.toml` `[bans]`. Introducing inverted or unapproved dependencies will cause immediate CI failure.

---

## Key Types & Public API

- `Orchestrator`: Multi-agent subsystem coordinator.
- `IntegrationMetrics`: Telemetry tracking cross-subsystem event propagation.

---

## Feature Flags

- *(None — zero optional feature flags; completely self-contained)*

---

## Usage Example

```rust
use forge_integration_layer::orchestrator::Orchestrator;
use forge_integration_layer::config::IntegrationConfig;

let orch = Orchestrator::new(IntegrationConfig::default());
```

---

## Testing & Verification

Run tests for this crate:

```bash
cargo test -p forge-integration
cargo clippy -p forge-integration -- -D warnings
```
