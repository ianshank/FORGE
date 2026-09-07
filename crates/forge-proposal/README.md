# `forge-proposal`

> **Architectural Layer**: Tier 1: Domain Primitives

Proposal template engine, agency guidelines, budgeting calculations, and document rendering for research initiatives.

---

## Architectural Role & Dependencies

As defined in the **FORGE Workspace Layering Model** (`docs/architecture.md` Section 4.2):

* **Tier**: `Tier 1: Domain Primitives`
* **Allowed Downstream Consumers (`wrappers`)**: Root integration tests (`forge-integration-tests`)
* **Workspace Dependencies**: `forge-types`

Dependency boundaries are strictly enforced via `deny.toml` `[bans]`. Introducing inverted or unapproved dependencies will cause immediate CI failure.

---

## Key Types & Public API

- `Proposal`: Top-level proposal structure and metadata.
- `CostModel`: Budget and direct/indirect cost calculations.
- `Render`: Document output generation.

---

## Feature Flags

- *(None — zero optional feature flags; completely self-contained)*

---

## Usage Example

```rust
use forge_proposal::proposal::Proposal;

let proposal = Proposal::default();
```

---

## Testing & Verification

Run tests for this crate:

```bash
cargo test -p forge-proposal
cargo clippy -p forge-proposal -- -D warnings
```
