// Block-grid observation encoder.
//
// Emits an ego-centric cube of per-tile features around the bot's
// position. Layout matches MuZeroConfig.{grid_height, grid_width,
// grid_channels} on the Python side; channel order is pinned by
// BLOCK_FEATURE_CHANNELS and exercised by the cross-language fixture
// tests in mc-bot/test/observation_grid.test.js and
// crates/forge-env-mc/tests/block_grid_channel_order.rs.

import { stableStringHash } from './hash.js';

// prismarine-world's `getBlock(pos)` only reads `pos.{x,y,z}` at
// runtime, so a plain object literal is the minimal surface that
// works against every supported mineflayer version. We sidestep
// pulling the `vec3` package as a hard runtime dependency for the
// dep-free test-runner job in CI.
function blockPos(x, y, z) {
  return { x, y, z };
}

export const DEFAULT_GRID_RADIUS = 5;
// Default height_radius = 0 → single Y-layer (the bot's eye-level
// slice). This matches MuZeroConfig.grid_flat_dim = H * W * C (no
// depth axis), so the Python trainer's 2D CNN consumes the grid
// without reshape gymnastics. Set > 0 in env.toml for 3D variants
// (requires a Conv3d branch on the Python side — deferred to v0.5
// Phase 2).
export const DEFAULT_GRID_HEIGHT_RADIUS = 0;
export const DEFAULT_GRID_CHANNELS = 7;
export const DEFAULT_BLOCK_ID_HASH_MOD = 4096;
export const DEFAULT_BIOME_ID_HASH_MOD = 256;

// Frozen channel order. The Python trainer reads the flattened grid
// as (channels, height, width) — reordering any entry here silently
// mis-trains the CNN. Coordinated test pins live in:
//   mc-bot/test/observation_grid.test.js  (this side)
//   crates/forge-env-mc/tests/block_grid_channel_order.rs  (Rust side)
//   tests/python/training/test_muzero_mc_replay.py  (Python side)
export const BLOCK_FEATURE_CHANNELS = Object.freeze([
  'block_type_hash',
  'light_level',
  'hardness',
  'is_solid',
  'is_liquid',
  'is_dangerous',
  'biome_id_hash',
]);

// Heuristic membership for the is_dangerous channel. Loaded from the
// config when present; fallback list keeps the encoder usable when the
// operator hasn't customised env.toml.
const DEFAULT_DANGEROUS_NAMES = Object.freeze([
  'lava',
  'flowing_lava',
  'fire',
  'soul_fire',
  'magma_block',
  'cactus',
  'sweet_berry_bush',
  'wither_rose',
]);

function finiteNumber(value, fallback = 0) {
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue : fallback;
}

function positiveIntOr(value, fallback) {
  return Number.isInteger(value) && value > 0 ? value : fallback;
}

function nonNegIntOr(value, fallback) {
  return Number.isInteger(value) && value >= 0 ? value : fallback;
}

/**
 * Per-axis dimensions for a configured grid.
 *
 * @param {object} config  Observation config (the `[observation]` table).
 * @returns {{ height: number, width: number, depth: number, channels: number }}
 *   height = 2 * grid_radius + 1 (x-axis)
 *   width  = 2 * grid_radius + 1 (z-axis)
 *   depth  = 2 * grid_height_radius + 1 (y-axis layers)
 *   channels = grid_channels (== BLOCK_FEATURE_CHANNELS.length when default)
 */
export function gridShape(config = {}) {
  const radius = nonNegIntOr(config.grid_radius, DEFAULT_GRID_RADIUS);
  const heightRadius = nonNegIntOr(config.grid_height_radius, DEFAULT_GRID_HEIGHT_RADIUS);
  const channels = positiveIntOr(config.grid_channels, DEFAULT_GRID_CHANNELS);
  return {
    height: 2 * radius + 1,
    width: 2 * radius + 1,
    depth: 2 * heightRadius + 1,
    channels,
  };
}

/**
 * Total flat float count for the block grid under the given config.
 *
 * @param {object} config
 * @returns {number}
 */
export function gridFlatDim(config = {}) {
  const shape = gridShape(config);
  return shape.height * shape.width * shape.depth * shape.channels;
}

function buildDangerousSet(config) {
  const raw = Array.isArray(config?.dangerous_block_names)
    ? config.dangerous_block_names
    : DEFAULT_DANGEROUS_NAMES;
  return new Set(raw.map((name) => String(name)));
}

function blockTypeName(block) {
  if (!block) return '';
  if (typeof block.name === 'string') return block.name;
  if (Number.isFinite(block.type)) return String(block.type);
  return '';
}

function blockLightLevel(block) {
  // mineflayer Block exposes `light` and `skyLight` as 0..15.  Combine
  // and normalise into [0, 1] so the trainer doesn't need to know the
  // raw range.  Undefined → 0 (sentinel for "unknown / unloaded").
  const blockLight = finiteNumber(block?.light, 0);
  const skyLight = finiteNumber(block?.skyLight, 0);
  const combined = Math.max(blockLight, skyLight);
  return combined / 15.0;
}

function blockHardness(block, hardnessScale) {
  // mineflayer's `hardness` is either a finite number or null
  // (bedrock-style indestructible blocks). Map nulls/NaN to a finite
  // "very hard" sentinel that won't NaN-propagate; otherwise normalise
  // by the configured scale (default 10 → most mineable blocks land in
  // a 0..2 range, bedrock-likes saturate near 1.0).
  const raw = block?.hardness;
  if (!Number.isFinite(raw)) {
    return raw === null ? 1.0 : 0.0;
  }
  return raw / hardnessScale;
}

function blockIsSolid(block) {
  if (!block) return 0;
  if (typeof block.boundingBox === 'string') {
    return block.boundingBox === 'block' ? 1 : 0;
  }
  // Fallback to mineflayer's deprecated `solid` boolean.
  return block.solid ? 1 : 0;
}

function blockIsLiquid(block) {
  if (!block) return 0;
  const name = blockTypeName(block);
  if (!name) return 0;
  return name.includes('water') || name.includes('lava') ? 1 : 0;
}

function blockIsDangerous(block, dangerousSet) {
  if (!block) return 0;
  const name = blockTypeName(block);
  if (!name) return 0;
  // Strip the `minecraft:` namespace if present.
  const bare = name.includes(':') ? name.split(':').pop() : name;
  return dangerousSet.has(bare) ? 1 : 0;
}

function biomeIdHash(block, biomeHashMod) {
  if (!block) return 0;
  if (Number.isFinite(block.biome?.id)) {
    return block.biome.id % biomeHashMod;
  }
  if (typeof block.biome?.name === 'string') {
    return stableStringHash(block.biome.name, biomeHashMod);
  }
  return 0;
}

function safeGetBlock(world, pos, missedTilesRef) {
  if (!world || typeof world.getBlock !== 'function') {
    missedTilesRef.count += 1;
    return null;
  }
  try {
    const block = world.getBlock(pos);
    if (!block) {
      missedTilesRef.count += 1;
      return null;
    }
    return block;
  } catch (_error) {
    missedTilesRef.count += 1;
    return null;
  }
}

/**
 * Encode the ego-centric block grid into a flat array of floats.
 *
 * Output layout: row-major over (channel, depth=y, height=x, width=z)
 * — i.e. the slowest-moving index is the channel, then y, then x,
 * then z. This matches a PyTorch tensor reshaped as
 * `(channels, depth, height, width)` which the MuZero CNN expects.
 *
 * Missing chunks / unloaded blocks → all-zero features and a bump on
 * the `missedTiles` counter so the operator log surfaces sampling
 * health per episode.
 *
 * @param {object} bot     mineflayer-shaped bot (only `entity.position`
 *                         and `world.getBlock(Vec3)` are touched).
 * @param {object} config  Observation config sub-table.
 * @returns {{ floats: number[], missedTiles: number, topBlockTypes: Array<[string, number]>, shape: ReturnType<gridShape> }}
 */
export function encodeBlockGrid(bot, config = {}) {
  const shape = gridShape(config);
  const radius = (shape.width - 1) / 2;
  const heightRadius = (shape.depth - 1) / 2;
  const channels = shape.channels;
  const blockIdHashMod = positiveIntOr(config.block_id_hash_mod, DEFAULT_BLOCK_ID_HASH_MOD);
  const biomeIdHashMod = positiveIntOr(config.biome_id_hash_mod, DEFAULT_BIOME_ID_HASH_MOD);
  const hardnessScale = finiteNumber(config.grid_hardness_scale, 10) || 10;
  const dangerousSet = buildDangerousSet(config);
  const blockHistogram = new Map();
  const missedTilesRef = { count: 0 };

  const origin = bot?.entity?.position ?? bot?.position ?? { x: 0, y: 0, z: 0 };
  const baseX = Math.floor(finiteNumber(origin.x, 0));
  const baseY = Math.floor(finiteNumber(origin.y, 0));
  const baseZ = Math.floor(finiteNumber(origin.z, 0));

  const floats = new Array(gridFlatDim(config)).fill(0);

  // Outermost loop: channel.  This lets us pack tile features into
  // separate planes ready for a `view(channels, depth, height, width)`
  // reshape on the Python side.
  const planeSize = shape.depth * shape.height * shape.width;
  for (let dy = -heightRadius; dy <= heightRadius; dy += 1) {
    for (let dx = -radius; dx <= radius; dx += 1) {
      for (let dz = -radius; dz <= radius; dz += 1) {
        const pos = blockPos(baseX + dx, baseY + dy, baseZ + dz);
        const block = safeGetBlock(bot?.world, pos, missedTilesRef);
        const name = blockTypeName(block);
        if (name) {
          blockHistogram.set(name, (blockHistogram.get(name) ?? 0) + 1);
        }
        const features = block
          ? [
              stableStringHash(name, blockIdHashMod) / blockIdHashMod,
              blockLightLevel(block),
              blockHardness(block, hardnessScale),
              blockIsSolid(block),
              blockIsLiquid(block),
              blockIsDangerous(block, dangerousSet),
              biomeIdHash(block, biomeIdHashMod) / biomeIdHashMod,
            ]
          : null;

        const planeOffset =
          (dy + heightRadius) * shape.height * shape.width +
          (dx + radius) * shape.width +
          (dz + radius);

        // Always touch every (channel, position) slot — zero-fill when
        // we have no block, otherwise write the coerced finite value.
        for (let c = 0; c < channels; c += 1) {
          const value = features && c < features.length ? features[c] : 0;
          floats[c * planeSize + planeOffset] = finiteNumber(value, 0);
        }
      }
    }
  }

  const topBlockTypes = [...blockHistogram.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5);

  return {
    floats,
    missedTiles: missedTilesRef.count,
    topBlockTypes,
    shape,
  };
}

/**
 * Helper for the index.js handshake — packs a `grid_shape` payload
 * matching the Rust `protocol::GridShape` deserialisation contract.
 *
 * Returning `null` when block-grid is disabled tells the handshake to
 * omit the field entirely so legacy bots stay protocol-compatible.
 */
export function gridShapePayload(config = {}) {
  const include = config?.include_block_grid === true;
  if (!include) return null;
  const shape = gridShape(config);
  return {
    height: shape.height,
    width: shape.width,
    depth: shape.depth,
    channels: shape.channels,
    vector_dim: positiveIntOr(config?.flat_vector_dim, 0),
  };
}
