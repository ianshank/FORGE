# mc-bot

Node bridge from mineflayer to FORGE's `forge-env-mc` Rust client.

Protocol v1 — JSON-only WebSocket. Wire format defined in
[`src/protocol.js`](src/protocol.js); must agree byte-for-byte with
`crates/forge-env-mc/src/protocol.rs`.

## Layout

```
mc-bot/
  src/
    protocol.js     - ClientMsg parser + ServerMsg builders
    action_map.js   - loads + validates action_map.toml; canonicalises
    schema_id.js    - SHA256 over canonical action map (xlang-pinned)
    reward/
      index.js      - composer over named built-ins
      builtins/
        survival.js
        inventory_acquired.js
        distance_to_goal.js
        health_delta.js
        composite.js
    reset.js        - teleport-based episode reset (v1 strategy)
  test/             - node:test suites (no install needed for these)
```

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
