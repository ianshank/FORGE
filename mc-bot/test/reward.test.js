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
});
