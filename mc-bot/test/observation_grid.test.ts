import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import {
  BLOCK_FEATURE_CHANNELS,
  DEFAULT_BIOME_ID_HASH_MOD,
  DEFAULT_BLOCK_ID_HASH_MOD,
  DEFAULT_GRID_CHANNELS,
  DEFAULT_GRID_HEIGHT_RADIUS,
  DEFAULT_GRID_RADIUS,
  encodeBlockGrid,
  gridFlatDim,
  gridShape,
  gridShapePayload,
} from '../src/observation_grid.js';

function makeBotWithWorld(getBlockImpl: any) {
  return {
    entity: { position: { x: 0, y: 64, z: 0 } },
    world: { getBlock: getBlockImpl },
  };
}

function fullSolidBlock(name = 'stone') {
  return {
    name,
    type: 1,
    light: 7,
    skyLight: 7,
    hardness: 1.5,
    boundingBox: 'block',
    biome: { id: 1, name: 'plains' },
  };
}

describe('observation_grid — shape', () => {
  it('defaults produce 11×11×1×7 = 847 floats (matches MuZero 2D CNN)', () => {
    const shape = gridShape({});
    assert.equal(shape.height, 11);
    assert.equal(shape.width, 11);
    assert.equal(shape.depth, 1); // height_radius=0 → single Y-layer
    assert.equal(shape.channels, 7);
    assert.equal(gridFlatDim({}), 11 * 11 * 1 * 7);
  });

  it('grid_height_radius=2 enables a 3D variant (3D CNN deferred to Phase 2)', () => {
    const config = { grid_height_radius: 2 };
    const shape = gridShape(config);
    assert.equal(shape.depth, 5);
    assert.equal(gridFlatDim(config), 11 * 11 * 5 * 7);
  });

  it('rejects negative grid_radius (falls back to default)', () => {
    const shape = gridShape({ grid_radius: -3 });
    assert.equal(shape.height, 2 * DEFAULT_GRID_RADIUS + 1);
    assert.equal(shape.width, 2 * DEFAULT_GRID_RADIUS + 1);
  });

  it('rejects non-positive grid_channels (falls back to default)', () => {
    const shape = gridShape({ grid_channels: 0 });
    assert.equal(shape.channels, DEFAULT_GRID_CHANNELS);
  });

  it('default constants are the documented values', () => {
    assert.equal(DEFAULT_GRID_RADIUS, 5);
    assert.equal(DEFAULT_GRID_HEIGHT_RADIUS, 0);
    assert.equal(DEFAULT_GRID_CHANNELS, 7);
    assert.equal(DEFAULT_BLOCK_ID_HASH_MOD, 4096);
    assert.equal(DEFAULT_BIOME_ID_HASH_MOD, 256);
  });
});

describe('observation_grid — feature channel order pin', () => {
  it('BLOCK_FEATURE_CHANNELS is frozen and matches the cross-language contract', () => {
    // Coordinated test: drift here forces a coordinated update to
    //   crates/forge-env-mc/tests/block_grid_channel_order.rs
    //   tests/python/training/test_muzero_mc_replay.py
    // See observation_grid.js for the contract description.
    assert.deepEqual(BLOCK_FEATURE_CHANNELS as any, [
      'block_type_hash',
      'light_level',
      'hardness',
      'is_solid',
      'is_liquid',
      'is_dangerous',
      'biome_id_hash',
    ]);
    assert.equal(Object.isFrozen(BLOCK_FEATURE_CHANNELS), true);
  });
});

describe('observation_grid — encodeBlockGrid', () => {
  it('emits exactly gridFlatDim floats for default config', () => {
    const bot = makeBotWithWorld(() => fullSolidBlock());
    const result = encodeBlockGrid(bot, {});
    assert.equal(result.floats.length, gridFlatDim({}));
    for (const value of result.floats) {
      assert.equal(Number.isFinite(value), true);
    }
  });

  it('emits 11 * 11 * 1 * 7 = 847 floats for the default 2D variant', () => {
    // First-real-run target shape: defaults (radius=5, height_radius=0,
    // channels=7) → 11*11*1*7 = 847 grid floats, matching
    // MuZeroConfig.grid_flat_dim on the Python side.
    const config = { include_block_grid: true };
    const bot = makeBotWithWorld(() => fullSolidBlock());
    const result = encodeBlockGrid(bot, config);
    assert.equal(result.floats.length, 11 * 11 * 1 * 7);
  });

  it('zero-fills unloaded tiles and bumps missed_tiles', () => {
    let calls = 0;
    const bot = makeBotWithWorld(() => {
      calls += 1;
      return calls % 2 === 0 ? null : fullSolidBlock();
    });
    const config = { grid_radius: 1, grid_height_radius: 0, include_block_grid: true };
    const result = encodeBlockGrid(bot, config);
    // 3 * 3 = 9 tiles; ~half should be null.
    assert.ok(result.missedTiles >= 4 && result.missedTiles <= 5);
    // Verify zero-fill: every channel of the null tiles must be 0.
    // We can't easily index without re-deriving the layout, so just
    // verify the total count of zeros is at least missedTiles * 7.
    const zeroCount = result.floats.filter((v) => v === 0).length;
    assert.ok(zeroCount >= result.missedTiles * 7);
  });

  it('swallows getBlock exceptions and bumps missed_tiles', () => {
    const bot = makeBotWithWorld(() => {
      throw new Error('chunk not loaded');
    });
    const config = { grid_radius: 1, grid_height_radius: 0, include_block_grid: true };
    const result = encodeBlockGrid(bot, config);
    assert.equal(result.missedTiles, 9);
    // Every float must still be finite (NaN propagation guard).
    for (const value of result.floats) assert.equal(Number.isFinite(value), true);
  });

  it('block_type_hash channel respects block_id_hash_mod', () => {
    const config = {
      grid_radius: 0,
      grid_height_radius: 0,
      include_block_grid: true,
      block_id_hash_mod: 16,
      use_raw_block_id: false,
    };
    const bot = makeBotWithWorld(() => fullSolidBlock('minecraft:stone'));
    const result = encodeBlockGrid(bot, config);
    // Single tile, 7 channels → first float is block_type_hash normalised by hash_mod.
    // hash mod 16, normalised → value strictly less than 1.
    const blockTypeHashValue = result.floats[0];
    assert.ok(blockTypeHashValue >= 0 && blockTypeHashValue < 1);
  });

  it('blockTypeEmbeddingIndex maps block name to raw embedding index when use_raw_block_id is enabled', () => {
    const config = {
      grid_radius: 0,
      grid_height_radius: 0,
      include_block_grid: true,
      use_raw_block_id: true,
      block_embeddings: {
        stone: 42,
        dirt: 12,
        unknown: 99,
      },
    };
    
    // Test exact match
    const botStone = makeBotWithWorld(() => fullSolidBlock('stone'));
    const resultStone = encodeBlockGrid(botStone, config);
    assert.equal(resultStone.floats[0], 42);

    // Test bare name extraction (minecraft:dirt -> dirt -> 12)
    const botDirt = makeBotWithWorld(() => fullSolidBlock('minecraft:dirt'));
    const resultDirt = encodeBlockGrid(botDirt, config);
    assert.equal(resultDirt.floats[0], 12);

    // Test unknown fallback (grass_block is not in map -> unknown -> 99)
    const botGrass = makeBotWithWorld(() => fullSolidBlock('grass_block'));
    const resultGrass = encodeBlockGrid(botGrass, config);
    assert.equal(resultGrass.floats[0], 99);
  });

  it('biome_id_hash channel respects biome_id_hash_mod', () => {
    const config = {
      grid_radius: 0,
      grid_height_radius: 0,
      include_block_grid: true,
      biome_id_hash_mod: 8,
    };
    const block = {
      ...fullSolidBlock(),
      biome: { id: 42, name: 'forest' },
    };
    const bot = makeBotWithWorld(() => block);
    const result = encodeBlockGrid(bot, config);
    // Biome channel is at index 6 (last in BLOCK_FEATURE_CHANNELS).
    // Layout: 1 tile × 7 channels = 7 floats; channels are PLANE-MAJOR
    // so channel `c` lives at index `c * planeSize` (planeSize=1).
    const biomeValue = result.floats[6];
    assert.equal(biomeValue, (42 % 8) / 8);
  });

  it('is_solid / is_liquid / is_dangerous return {0,1} sentinels', () => {
    const config = { grid_radius: 0, grid_height_radius: 0, include_block_grid: true };
    const lavaBlock = {
      name: 'lava',
      type: 11,
      boundingBox: 'empty',
      hardness: 100,
      light: 15,
      skyLight: 15,
      biome: { id: 1 },
    };
    const bot = makeBotWithWorld(() => lavaBlock);
    const result = encodeBlockGrid(bot, config);
    // Layout: PLANE-MAJOR — single tile means each channel is at
    // index c*1 = c. So: is_solid=ch3, is_liquid=ch4, is_dangerous=ch5.
    assert.equal(result.floats[3], 0); // boundingBox != 'block'
    assert.equal(result.floats[4], 1); // name contains 'lava'
    assert.equal(result.floats[5], 1); // 'lava' in dangerous set
  });

  it('handles missing bot.world gracefully', () => {
    const result = encodeBlockGrid({}, { include_block_grid: true });
    assert.equal(result.floats.length, gridFlatDim({}));
    assert.equal(result.missedTiles, gridFlatDim({}) / 7);
  });

  it('topBlockTypes returns the 5 most common observed block names', () => {
    const counter = { stone: 0, dirt: 0 };
    const bot = makeBotWithWorld(() => {
      counter.stone += 1;
      const name = counter.stone % 3 === 0 ? 'dirt' : 'stone';
      return { ...fullSolidBlock(name) };
    });
    const config = { grid_radius: 2, grid_height_radius: 0, include_block_grid: true };
    const result = encodeBlockGrid(bot, config);
    assert.ok(result.topBlockTypes.length >= 1);
    assert.ok(result.topBlockTypes[0][0] === 'stone' || result.topBlockTypes[0][0] === 'dirt');
  });
});

describe('observation_grid — gridShapePayload (handshake)', () => {
  it('returns null when include_block_grid is disabled', () => {
    assert.equal(gridShapePayload({ include_block_grid: false }), null);
    assert.equal(gridShapePayload({}), null);
  });

  it('emits a positive-integer payload matching the default 2D encoder shape', () => {
    const payload = gridShapePayload({ include_block_grid: true, flat_vector_dim: 73 });
    assert.ok(payload);
    assert.equal(payload!.height, 11);
    assert.equal(payload!.width, 11);
    assert.equal(payload!.depth, 1); // height_radius=0 default
    assert.equal(payload!.channels, 7);
    assert.equal(payload!.vector_dim, 73);
  });

  it('emits vector_dim=0 when flat_vector_dim is omitted', () => {
    const payload = gridShapePayload({ include_block_grid: true });
    assert.ok(payload);
    assert.equal(payload!.vector_dim, 0);
  });
});
