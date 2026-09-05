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

## `[websocket]` hardening knobs

The control channel drives an op'd bot, so it is treated as a privileged
surface. Every limit below lives in the `[websocket]` table of the bot's
TOML config; the table is optional and every key falls back to the default
shown, so existing `env.toml` files keep parsing unchanged. Defaults are
defined once, in `DEFAULT_WEBSOCKET_LIMITS` (`src/config.ts`) — no literals
at any call site.

| Key | Default | Purpose |
|---|---|---|
| `max_payload_bytes` | `16384` | Cap on a single inbound frame (`ws` defaults to 100 MiB). The largest legitimate client message is ~40 bytes, so this is ~400x headroom |
| `max_queue_depth` | `32` | Maximum client messages in flight. The protocol is request/response, so a well-behaved client sits at 1. Breaching it returns a `BACKPRESSURE` error and closes the socket |
| `idle_timeout_ms` | `120000` | Reclaim the single-client slot after this long with no *application* message. `0` disables. Pongs deliberately do not reset it — a live-but-silent client must not hold the bot forever |
| `ping_interval_ms` | `20000` | Server-initiated keepalive. A ping unanswered for a full interval marks a half-open connection dead and reclaims it. `0` disables |
| `auth_token` | unset | Optional shared secret. Unset = unauthenticated (a prominent startup `warn` is logged) |
| `auth_query_param` | `"token"` | Name of the query-string parameter carrying `auth_token` |

`url`, `host` and `port` remain derived from `ws_url`; setting them under
`[websocket]` has no effect.

### Authenticating the control channel

Set a secret:

```toml
[websocket]
auth_token = "a-long-random-string"
```

A client then supplies it in **either** form — both are checked, header
first, and compared in constant time:

1. `Authorization: Bearer <token>` request header (preferred; keeps the
   secret out of access logs), or
2. `?token=<token>` on the WebSocket URL, e.g.
   `ws://mc-bot:8765/?token=a-long-random-string`
   (rename the parameter with `auth_query_param`).

A handshake without a matching credential is refused with `401` before any
protocol message is exchanged. With `auth_token` unset, behaviour is exactly
as before and the bot logs
`{"event":"websocket_auth_disabled", ...}` at `warn` on startup.

## Building

`src/` is TypeScript only. `npm start` runs it through `tsx`; the container
image compiles it instead:

```bash
npm run build     # tsc -p tsconfig.build.json → dist/
node dist/index.js
```

`tsconfig.build.json` pins `rootDir` to `./src` so the entry point lands at
`dist/index.js` (not `dist/src/index.js`). That matters: `DEFAULT_CONFIG_DIR`
is derived as `<module dir>/../../configs/minecraft`, which must resolve to
`/configs/minecraft` inside the image — the path
`docker/compose.minecraft.yml` mounts the config tree at.

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
- ✅ `index.js` entry point + mineflayer wire-up (v0.5 Phase 1 —
       verified end-to-end against a real `itzg/minecraft-server`)
- ✅ prismarine-viewer integration (browser viewer at :3007)
- ✅ End-to-end episode against a real Minecraft server (v0.5 Phase 1
       — see [`docs/results/v0.5-first-real-run.md`](../docs/results/v0.5-first-real-run.md))
- ✅ v0.5 block-grid observation encoder (`observation_grid.js`) +
       `Hello.grid_shape` cross-language pin
- ⏳ Mineflayer auto-reconnect on MC-side tick timeout (Phase 2 —
       known production-stability gap; the bot's WS layer stays UP
       but the mineflayer connection enters a half-open state after
       the first MC server-side exception, requiring a
       `docker compose restart mc-bot` between captures today)

## Docker vs local-dev hostnames

The shipped `configs/minecraft/env.toml` uses local-dev defaults
(`bot.host = "127.0.0.1"`, `ws_url = "ws://127.0.0.1:8765"`) so
running the bot natively on the host works without edits.

For docker compose runs, `docker/compose.minecraft.yml` mounts
`configs/minecraft/env.docker.toml` over `env.toml` (single-file
overlay) so the bot uses docker DNS hostnames (`bot.host =
"minecraft"`, `ws_url = "ws://mc-bot:8766"`) automatically. No
operator action required.

See `docs/plans/minecraft_rl_integration_plan_v2.md` Phase 3 for the
full design and the protocol contract.
