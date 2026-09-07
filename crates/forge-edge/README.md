# `forge-edge`

> **Maturity**: `[Research]` (Research Stack Component — compute-budget bounded edge MCTS inference)  
> **Architectural Layer**: Tier 4: Applications & Runners

Edge deployment runtime featuring compute-budget adaptive MCTS search, execution latency estimation, and edge telemetry collection.

---

## Architectural Role & Dependencies

As defined in the **FORGE Workspace Layering Model** (`docs/architecture.md` Section 4.2):

* **Tier**: `Tier 4: Applications & Runners`
* **Allowed Downstream Consumers (`wrappers`)**: Edge robotics and constrained runtime binaries
* **Workspace Dependencies**: `forge-agent`, `forge-replay`, `forge-types`

Dependency boundaries are strictly enforced via `deny.toml` `[bans]`. Introducing inverted or unapproved dependencies will cause immediate CI failure.

---

## Key Types & Public API

- `EdgeAgent`: Lightweight agent optimizing decision frequency under strict latency constraints.
- `AdaptiveMctsSearch`: Dynamically scales search iterations based on remaining frame time.
- `LatencyEstimator`: Moving-average estimation of hardware inference time.

---

## Feature Flags

- *(None — zero optional feature flags; completely self-contained)*

---

## Usage Example

```rust
use forge_edge::EdgeAgent;

// Runs real-time planning bounded by microsecond compute budgets.
```

---

## Testing & Verification

Run tests for this crate:

```bash
cargo test -p forge-edge
cargo clippy -p forge-edge -- -D warnings
```
