# `forge-agent`

> **Architectural Layer**: Tier 3: Agents & Interfaces

Agent frameworks, heuristic baselines, Latent MCTS search planning, and neural network forward model integration.

---

## Architectural Role & Dependencies

As defined in the **FORGE Workspace Layering Model** (`docs/architecture.md` Section 4.2):

* **Tier**: `Tier 3: Agents & Interfaces`
* **Allowed Downstream Consumers (`wrappers`)**: `forge-bench`, `forge-data`, `forge-edge`, `forge-mangomas`, `forge-mc-runner`
* **Workspace Dependencies**: `forge-core`, `forge-types`

Dependency boundaries are strictly enforced via `deny.toml` `[bans]`. Introducing inverted or unapproved dependencies will cause immediate CI failure.

---

## Key Types & Public API

- `LatentMctsSearch`: Monte Carlo Tree Search operating on latent world representations.
- `LatentForwardModel`: Interface for latent transition, policy, and value predictions.
- `RandomAgent`: Heuristic baseline agent choosing random valid actions.
- `RuleBasedAgent`: State-machine scripted heuristic policy.

---

## Feature Flags

- `default` - Standard baselines and CPU MCTS.
- `onnx` - Enables ONNX Runtime integration via `ort` for neural forward models.
- `onnx-bundled` - Links statically against bundled ONNX runtime binaries.

---

## Usage Example

```rust
use forge_agent::baselines::RandomAgent;

let agent = RandomAgent::new(42);
```

---

## Testing & Verification

Run tests for this crate:

```bash
cargo test -p forge-agent
cargo clippy -p forge-agent -- -D warnings
```
