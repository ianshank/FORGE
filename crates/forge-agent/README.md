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

- `HierarchicalSkillAgent`: Options/HRL executor over `configs/agents/skills_default.toml`.
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
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;

let rng = Pcg64Mcg::seed_from_u64(42);
let agent = RandomAgent::new(rng, 0);
```

Hierarchical skill options (config-driven reusable primitives) live in
`forge_agent::skills::HierarchicalSkillAgent` and are catalogued in
`configs/agents/skills_default.toml`.

---

## Testing & Verification

Run tests for this crate:

```bash
cargo test -p forge-agent
cargo clippy -p forge-agent -- -D warnings
```
