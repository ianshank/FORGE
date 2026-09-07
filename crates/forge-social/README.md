# `forge-social`

> **Architectural Layer**: Tier 1: Domain Primitives

Multi-agent social dynamics, trust networks, reputation scoring, alliance formation, and communication protocol modeling.

---

## Architectural Role & Dependencies

As defined in the **FORGE Workspace Layering Model** (`docs/architecture.md` Section 4.2):

* **Tier**: `Tier 1: Domain Primitives`
* **Allowed Downstream Consumers (`wrappers`)**: `forge-integration-layer`
* **Workspace Dependencies**: `forge-types`

Dependency boundaries are strictly enforced via `deny.toml` `[bans]`. Introducing inverted or unapproved dependencies will cause immediate CI failure.

---

## Key Types & Public API

- `TrustMatrix`: Pairwise trust scores between simulation agents.
- `ReputationTracker`: History of agent cooperativeness and defections.
- `Alliance`: Dynamic coalitions and team groupings.
- `SocialReward`: Social utility reward function calculation.

---

## Feature Flags

- *(None — zero optional feature flags; completely self-contained)*

---

## Usage Example

```rust
use forge_social::trust::TrustMatrix;
use forge_types::entity::AgentId;

let mut trust = TrustMatrix::new();
trust.record_interaction(AgentId(1), AgentId(2), 0.8);
```

---

## Testing & Verification

Run tests for this crate:

```bash
cargo test -p forge-social
cargo clippy -p forge-social -- -D warnings
```
