# `forge-memory`

> **Maturity**: `[Research]` (Research Stack Component — persistent cognitive memory architecture)  
> **Architectural Layer**: Tier 1: Domain Primitives

Working, episodic, and semantic memory architectures for agents, including vector buffer retrieval and observation replay buffers.

---

## Architectural Role & Dependencies

As defined in the **FORGE Workspace Layering Model** (`docs/architecture.md` Section 4.2):

* **Tier**: `Tier 1: Domain Primitives`
* **Allowed Downstream Consumers (`wrappers`)**: `forge-cognitive`, `forge-integration-layer`
* **Workspace Dependencies**: `forge-types`

Dependency boundaries are strictly enforced via `deny.toml` `[bans]`. Introducing inverted or unapproved dependencies will cause immediate CI failure.

---

## Key Types & Public API

- `EpisodicMemory`: Ring-buffered storage of recent agent experience.
- `SemanticStore`: Structured associative memory for world concepts.
- `PreferenceMemory`: Agent utility and preference tracking.

---

## Feature Flags

- *(None — zero optional feature flags; completely self-contained)*

---

## Usage Example

```rust
use forge_memory::episodic::EpisodicMemory;

let mut memory = EpisodicMemory::new(1000);
// Store and recall agent experiences across simulation episodes.
```

---

## Testing & Verification

Run tests for this crate:

```bash
cargo test -p forge-memory
cargo clippy -p forge-memory -- -D warnings
```
