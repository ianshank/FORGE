# `forge-replay`

> **Architectural Layer**: Tier 3: Agents & Interfaces

Deterministic simulation recording, compact delta encoding, replay file serialization, and TrajectoryV2 export.

---

## Architectural Role & Dependencies

As defined in the **FORGE Workspace Layering Model** (`docs/architecture.md` Section 4.2):

* **Tier**: `Tier 3: Agents & Interfaces`
* **Allowed Downstream Consumers (`wrappers`)**: `forge-cloud`, `forge-data`, `forge-edge`, `forge-eval`, `forge-mc-runner`
* **Workspace Dependencies**: `forge-core`, `forge-types`

Dependency boundaries are strictly enforced via `deny.toml` `[bans]`. Introducing inverted or unapproved dependencies will cause immediate CI failure.

---

## Key Types & Public API

- `CompactReplay`: seed + config + action sequence. Format version 2 uses a
  portable SHA-256 config hash and treats unknown action ids as hard errors.
- `BehavioralCoverage`: tiles, action histogram, predicate-arm activations,
  seeds, and constraint-violation classes measured on CompactReplay (cite ECC;
  exact measurement, not a new coverage invention).
- `ReplayPlayer`: Plays back recorded sessions with bit-identical determinism.
- `TrajectoryV2`: High-efficiency dataset format for offline RL and imitation learning.

---

## Feature Flags

- `default` - Standard replay file writing and decompression.
- `hf` - Hugging Face Hub upload integration for dataset publication.

---

## Usage Example

```rust
use forge_replay::v2::TrajectoryV2Writer;

// Records observations, actions, and rewards for downstream offline RL.
```

---

## Testing & Verification

Run tests for this crate:

```bash
cargo test -p forge-replay
cargo test -p forge-replay --test golden_replay
cargo clippy -p forge-replay -- -D warnings
```

Golden CompactReplay JSON lives in `tests/golden/replays/`. Regenerating
requires `UPDATE_GOLDEN_REPLAYS=1` and a row in `docs/results/replay_flip_log.md`.
