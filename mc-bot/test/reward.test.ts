import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import { buildReward, buildOne } from '../src/reward/index.js';

describe('reward — survival', () => {
  it('emits the configured constant', () => {
    const fn = buildOne({ kind: 'survival', value: 0.5 });
    assert.equal(fn({}), 0.5);
    assert.equal(fn({ prev: null, curr: null }), 0.5);
  });

  it('defaults to a small positive value', () => {
    const fn = buildOne({ kind: 'survival' });
    assert.ok(fn({}) > 0);
  });
});

describe('reward — inventory_acquired', () => {
  it('awards once per item, then stays at zero', () => {
    const fn = buildOne({
      kind: 'inventory_acquired',
      items: ['minecraft:oak_log', 'minecraft:dirt'],
      value: 10.0,
    });
    const r1 = fn({ curr: { inventory: { 'minecraft:oak_log': 1 } } });
    assert.equal(r1, 10.0);
    const r2 = fn({ curr: { inventory: { 'minecraft:oak_log': 5 } } });
    assert.equal(r2, 0);
    const r3 = fn({ curr: { inventory: { 'minecraft:dirt': 3, 'minecraft:oak_log': 5 } } });
    assert.equal(r3, 10.0);
  });

  it('zero when no items match', () => {
    const fn = buildOne({ kind: 'inventory_acquired', items: ['minecraft:diamond'] });
    assert.equal(fn({ curr: { inventory: { 'minecraft:dirt': 99 } } }), 0);
  });
});

describe('reward — distance_to_goal', () => {
  it('positive when getting closer', () => {
    const fn = buildOne({
      kind: 'distance_to_goal',
      target: { x: 0, y: 0, z: 0 },
      clip: 100,
    });
    const r = fn({
      prev: { position: { x: 10, y: 0, z: 0 } },
      curr: { position: { x: 5, y: 0, z: 0 } },
    });
    assert.equal(r, 5);
  });

  it('negative when getting farther', () => {
    const fn = buildOne({
      kind: 'distance_to_goal',
      target: { x: 0, y: 0, z: 0 },
      clip: 100,
    });
    const r = fn({
      prev: { position: { x: 1, y: 0, z: 0 } },
      curr: { position: { x: 7, y: 0, z: 0 } },
    });
    assert.equal(r, -6);
  });

  it('clipped to ±clip', () => {
    const fn = buildOne({
      kind: 'distance_to_goal',
      target: { x: 0, y: 0, z: 0 },
      clip: 1,
    });
    const r = fn({
      prev: { position: { x: 100, y: 0, z: 0 } },
      curr: { position: { x: 0, y: 0, z: 0 } },
    });
    assert.equal(r, 1); // would be 100 unclipped
  });
});

describe('reward — health_delta', () => {
  it('positive when health rises', () => {
    const fn = buildOne({ kind: 'health_delta', value: 2 });
    const r = fn({ prev: { health: 10 }, curr: { health: 15 } });
    assert.equal(r, 2);
  });

  it('negative when health drops', () => {
    const fn = buildOne({ kind: 'health_delta', value: 2 });
    const r = fn({ prev: { health: 15 }, curr: { health: 10 } });
    assert.equal(r, -2);
  });

  it('zero on no change or missing snapshots', () => {
    const fn = buildOne({ kind: 'health_delta', value: 2 });
    assert.equal(fn({ prev: { health: 10 }, curr: { health: 10 } }), 0);
    assert.equal(fn({}), 0);
  });
});

describe('reward — milestone', () => {
  it('awards once per milestone when conditions are met, then stays at zero', () => {
    const fn = buildOne({
      kind: 'milestone',
    });

    const r1 = fn({
      curr: {
        tick: 0,
        inventory: { 'minecraft:oak_log': 1 }
      }
    });
    assert.equal(r1, 10.0);

    const r2 = fn({
      prev: { tick: 0, inventory: { 'minecraft:oak_log': 1 } },
      curr: { tick: 1, inventory: { 'minecraft:oak_log': 1 } }
    });
    assert.equal(r2, 0);

    const r3 = fn({
      prev: { tick: 1, inventory: { 'minecraft:oak_log': 1 } },
      curr: { tick: 2, inventory: { 'minecraft:oak_log': 1, 'stone_pickaxe': 1 } }
    });
    assert.equal(r3, 25.0);

    const r4 = fn({
      prev: { tick: 2, inventory: { 'minecraft:oak_log': 1, 'stone_pickaxe': 1 } },
      curr: { tick: 3, inventory: { 'minecraft:oak_log': 1, 'stone_pickaxe': 1 } }
    });
    assert.equal(r4, 0);
  });

  it('triggers first_shelter milestone on placed shelter blocks in gridTopBlockTypes', () => {
    const fn = buildOne({
      kind: 'milestone',
    });

    const r = fn({
      curr: {
        tick: 0,
        gridTopBlockTypes: [['minecraft:furnace', 1]],
        inventory: {}
      }
    });
    assert.equal(r, 100.0);
  });

  it('resets achieved set on tick reset', () => {
    const fn = buildOne({
      kind: 'milestone',
    });

    const r1 = fn({
      curr: {
        tick: 0,
        inventory: { 'minecraft:oak_log': 1 }
      }
    });
    assert.equal(r1, 10.0);

    const r2 = fn({
      prev: { tick: 0, inventory: { 'minecraft:oak_log': 1 } },
      curr: { tick: 1, inventory: { 'minecraft:oak_log': 1 } }
    });
    assert.equal(r2, 0);

    const r3 = fn({
      prev: { tick: 1, inventory: { 'minecraft:oak_log': 1 } },
      curr: {
        tick: 0,
        inventory: { 'minecraft:oak_log': 1 }
      }
    });
    assert.equal(r3, 10.0);
  });
});

describe('reward — composite + buildReward', () => {
  it('weighted sum of children', () => {
    const fn = buildOne({
      kind: 'composite',
      weights: { survival: 1.0, health_delta: 0.5 },
      survival: { value: 1 },
      health_delta: { value: 2 },
    });
    const r = fn({ prev: { health: 5 }, curr: { health: 10 } });
    // survival: 1*1 = 1, health_delta: 0.5*2 = 1 → 2
    assert.equal(r, 2);
  });

  it('buildReward returns single fn when only one reward defined', () => {
    const fn = buildReward({ reward: [{ kind: 'survival', value: 0.7 }] });
    assert.equal(fn({}), 0.7);
  });

  it('buildReward sums multiple top-level rewards', () => {
    const fn = buildReward({
      reward: [
        { kind: 'survival', value: 1 },
        { kind: 'survival', value: 2 },
      ],
    });
    assert.equal(fn({}), 3);
  });

  it('rejects empty reward list', () => {
    assert.throws(() => buildReward({ reward: [] }));
    assert.throws(() => buildReward({}));
  });

  it('rejects unknown reward kind', () => {
    assert.throws(() => buildOne({ kind: 'magic' }), /unknown reward kind/);
  });

  it('composite reports the offending child kind in the error', () => {
    assert.throws(
      () => buildOne({ kind: 'composite', weights: { magic: 1.0 } }),
      /composite sub-reward "magic"/,
    );
  });

  it('composite rejects non-object weights', () => {
    assert.throws(() => buildOne({ kind: 'composite', weights: 'oops' }));
  });

  it('composite rejects nested composite to prevent infinite recursion', () => {
    assert.throws(
      () => buildOne({ kind: 'composite', weights: { composite: 1.0 } }),
      /composite cannot contain another composite/,
    );
  });

  it('composite rejects non-finite weight', () => {
    assert.throws(
      () => buildOne({ kind: 'composite', weights: { survival: NaN } }),
      /must be finite/,
    );
  });
});
