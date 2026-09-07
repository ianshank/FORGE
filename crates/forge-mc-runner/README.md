# `forge-mc-runner`

> **Architectural Layer**: Tier 4: Applications & Runners

Headless Minecraft episode runner: connects to `mc-bot`, runs Latent MCTS search, writes TrajectoryV2 records, and supports ONNX model hot-reloading.

---

## Architectural Role & Dependencies

As defined in the **FORGE Workspace Layering Model** (`docs/architecture.md` Section 4.2):

* **Tier**: `Tier 4: Applications & Runners`
* **Allowed Downstream Consumers (`wrappers`)**: Deployment binaries / CLI (`forge-mc-runner`)
* **Workspace Dependencies**: `forge-agent`, `forge-env`, `forge-env-mc`, `forge-observability`, `forge-replay`

Dependency boundaries are strictly enforced via `deny.toml` `[bans]`. Introducing inverted or unapproved dependencies will cause immediate CI failure.

---

## Key Types & Public API

- `Runner`: Primary execution loop driving episodes and checkpointing.
- `RunnerConfig`: Configuration controlling max steps, episode counts, and policy parameters.
- `HotReloader`: File watcher checking for updated ONNX model weights.

---

## Feature Flags

- `default` - Standard runner with baseline policies.
- `onnx-reload` - Dynamic file-watcher reloading ONNX policy models between episodes.
- `mc-live` - Live connection support against remote Minecraft servers.
- `mc-live-bundled` - Links bundled dependencies for standalone production deployments.

---

## Usage Example

```rust
// Invoked via CLI:
// cargo run -p forge-mc-runner -- --config configs/minecraft/runner.toml
```

---

## Testing & Verification

Run tests for this crate:

```bash
cargo test -p forge-mc-runner
cargo clippy -p forge-mc-runner -- -D warnings
```
