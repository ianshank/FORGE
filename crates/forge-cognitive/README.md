# `forge-cognitive`

> **Architectural Layer**: Tier 2: Engine & Cognitive

LLM-backed cognitive deliberation, prompt orchestration, teacher trace parsing, and reasoning agents for FORGE.

---

## Architectural Role & Dependencies

As defined in the **FORGE Workspace Layering Model** (`docs/architecture.md` Section 4.2):

* **Tier**: `Tier 2: Engine & Cognitive`
* **Allowed Downstream Consumers (`wrappers`)**: `forge-integration-layer`
* **Workspace Dependencies**: `forge-memory`, `forge-types`

Dependency boundaries are strictly enforced via `deny.toml` `[bans]`. Introducing inverted or unapproved dependencies will cause immediate CI failure.

---

## Key Types & Public API

- `CognitiveAgent`: Higher-order agent leveraging LLM reasoning loops.
- `ReasoningEngine`: Prompt building, few-shot demonstration injection, and response parsing.
- `Provider`: LLM API provider abstraction (OpenAI, Anthropic, LM Studio).

---

## Feature Flags

- *(None — zero optional feature flags; completely self-contained)*

---

## Usage Example

```rust
use forge_cognitive::agent::CognitiveAgent;
use forge_cognitive::config::CognitiveConfig;

let config = CognitiveConfig::default();
```

---

## Testing & Verification

Run tests for this crate:

```bash
cargo test -p forge-cognitive
cargo clippy -p forge-cognitive -- -D warnings
```
