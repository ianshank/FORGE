# mc-bot

Node bridge from mineflayer to FORGE's `forge-env-mc` Rust client.

Protocol v1 — JSON-only WebSocket. Wire format defined in
[`src/protocol.js`](src/protocol.js); must agree byte-for-byte with
`crates/forge-env-mc/src/protocol.rs`.

## Layout

```
mc-bot/
  src/
    protocol.js          - ClientMsg parser + ServerMsg builders
                           (Hello may carry optional `grid_shape` since v0.5)
    action_map.js        - loads + validates action_map.toml; canonicalises
    schema_id.js         - SHA256 over canonical action map (xlang-pinned)
    hash.js              - shared FNV-1a `stableStringHash` + `finiteNumber`
    observation.js       - flat scalar observation vector + `flat_vector_dim` pad
    observation_grid.js  - v0.5 ego-centric block-grid encoder
                           (xlang-pinned `BLOCK_FEATURE_CHANNELS`)
    reward/
      index.js           - composer over named built-ins
      builtins/
        survival.js
        inventory_acquired.js
        distance_to_goal.js
        health_delta.js
        composite.js
    reset.js             - teleport-based episode reset (v1 strategy)
  test/                  - node:test suites (no install needed for these)
```

## v0.5 `[observation]` config knobs

Every value below is read from the bot's TOML config (defaults at the
right). No hard-coded literals at any call site.

| Key | Default | Purpose |
|---|---|---|
| `include_block_grid` | `false` (JS) / `true` (env.toml) | Toggle the v0.5 block-grid prefix on the observation vector |
| `grid_radius` | `5` | X/Z half-extent → 11 tiles per side |
| `grid_height_radius` | `0` | Y half-extent (0 = single eye-level slice; matches MuZero 2D CNN) |
| `grid_channels` | `7` | Per-tile feature count; MUST match `BLOCK_FEATURE_CHANNELS.length` |
| `block_id_hash_mod` | `4096` | Modulus for `block_type_hash` channel |
| `biome_id_hash_mod` | `256` | Modulus for `biome_id_hash` channel |
| `grid_hardness_scale` | `10` | Normaliser for the `hardness` channel |
| `dangerous_block_names` | `["lava", "fire", "magma_block", ...]` | Names the `is_dangerous` channel matches |
| `flat_vector_dim` | unset (env.toml: `73`) | Zero-pad the flat suffix up to this length |
| `expected_dim` | unset (env.toml: `920`) | Runner-side cross-check; reject Hello on mismatch |

The bot's `Hello` handshake includes a `grid_shape` payload whenever
`include_block_grid = true`, and the runner cross-checks it against
`[observation.expected_grid_shape]` in `configs/minecraft/env.toml`.
Reorder either side without coordinating the other and both the JS
`BLOCK_FEATURE_CHANNELS` test and the Rust
`xlang_block_feature_channels_pinned_to_known_good` test fail
simultaneously.

## Tests

```bash
# Dep-free tests — work without npm install:
npm run test:no-deps

# Full suite (needs npm install for mineflayer/ws/etc.):
npm install
npm test
```

Cross-language regression gate: the `xlang_schema_id_pinned_to_known_good`
test in `crates/forge-env-mc/src/action_map.rs` and the
`xlang schema_id matches Rust` test in `test/schema_id.test.js` both
pin the same hex string. If one fails, the other will too — investigate
the canonical-form drift on both sides before bumping.

## Status

- ✅ Protocol message types + parser + builders
- ✅ Action map loader + validator + schema_id (xlang-checked vs Rust)
- ✅ Reward registry with 5 built-ins (survival, inventory_acquired,
       distance_to_goal, health_delta, composite)
- ✅ Reset strategy (teleport) with stub-bot tests
- ⏳ `index.js` entry point + mineflayer wire-up (needs real MC server)
- ⏳ prismarine-viewer integration
- ⏳ End-to-end episode against a Paper Minecraft server

See `docs/plans/minecraft_rl_integration_plan_v2.md` Phase 3 for the
full design and the protocol contract.
