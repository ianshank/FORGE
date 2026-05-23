import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import {
  computeObsDim,
  observationVectorFromSnapshot,
  snapshotObservation,
  stableStringHash,
} from '../src/observation.js';
import type { Snapshot } from '../src/observation.js';

function stubBot() {
  const slots: any[] = Array.from({ length: 45 }, () => null);
  slots[36] = { type: 17, name: 'oak_log', count: 3 };
  slots[37] = { name: 'dirt', count: 5 };
  return {
    time: { age: 12 },
    entity: {
      position: { x: 10, y: 65, z: -4 },
      velocity: { x: 0.1, y: 0, z: -0.2 },
      yaw: Math.PI / 2,
      pitch: 0,
    },
    health: 18,
    food: 17,
    oxygenLevel: 20,
    experience: { level: 2, progress: 0.5 },
    inventory: {
      slots,
      items() {
        return slots.filter(Boolean);
      },
    },
  };
}

describe('observation', () => {
  it('computes stable dimensions from config toggles', () => {
    assert.equal(computeObsDim({ inventory_slots: 9, inventory_features_per_slot: 2 }), 31);
    assert.equal(computeObsDim({ include_inventory: false }), 13);
    assert.equal(computeObsDim({ include_position: false, include_velocity: false, include_orientation: false, include_vitals: false, include_inventory: false }), 0);
  });

  it('flat_vector_dim zero-pads the flat vector up to the target', () => {
    // 31 raw features + zero-pad up to 73 = 73-float flat surface.
    // The grid is OFF here, so the total observation is exactly 73.
    const config = { flat_vector_dim: 73 };
    assert.equal(computeObsDim(config), 73);
    const snap = snapshotObservation(stubBot(), config);
    assert.equal(snap.obs!.length, 73);
    // Tail must be zero-pad (raw goes to 31 floats with defaults).
    for (let i = 31; i < 73; i += 1) {
      assert.equal(snap.obs![i], 0);
    }
  });

  it('include_block_grid prepends the block-grid floats', () => {
    // Default grid is 11×11×11×7 = 9317 floats with the encoder.  The
    // first-real-run plan targets a 2D variant via
    // grid_height_radius=0 (847 grid floats), so test both shapes
    // resolve to the documented total.
    const flatConfig = {
      flat_vector_dim: 73,
      include_block_grid: true,
      grid_radius: 5,
      grid_height_radius: 0,
    };
    assert.equal(computeObsDim(flatConfig), 11 * 11 * 1 * 7 + 73);
    assert.equal(computeObsDim({ ...flatConfig, grid_height_radius: 5 }), 11 * 11 * 11 * 7 + 73);
  });

  it('include_block_grid disabled emits the 31-float legacy shape', () => {
    // Backwards-compat pin: the existing flat 31-float surface keeps
    // working for downstream consumers that haven't migrated.
    const config = { inventory_slots: 9, inventory_features_per_slot: 2 };
    assert.equal(computeObsDim(config), 31);
  });

  it('builds finite vectors and inventory maps from bot state', () => {
    const config = { inventory_slots: 9, inventory_features_per_slot: 2 };
    const snapshot = snapshotObservation(stubBot(), config);
    assert.equal(snapshot.tick, 12);
    assert.equal(snapshot.obs!.length, computeObsDim(config));
    assert.equal(snapshot.inventory.oak_log, 3);
    assert.equal(snapshot.inventory.dirt, 5);
    for (const value of snapshot.obs!) {
      assert.equal(Number.isFinite(value), true);
    }
  });

  it('handles missing optional bot fields as zeros', () => {
    const snapshot = snapshotObservation({}, { include_inventory: false });
    assert.equal(snapshot.obs!.length, computeObsDim({ include_inventory: false }));
    assert.deepEqual(snapshot.position, { x: 0, y: 0, z: 0 });
  });

  it('hashes item names deterministically', () => {
    assert.equal(stableStringHash('minecraft:dirt'), stableStringHash('minecraft:dirt'));
    assert.notEqual(stableStringHash('minecraft:dirt'), stableStringHash('minecraft:stone'));
  });

  it('stableStringHash falls back to default modulus on bad input', () => {
    // Modulus must be a positive integer; negatives/floats fall back to
    // the compiled default. Both calls should produce the same bucket
    // for the same input, demonstrating the fallback is deterministic.
    assert.equal(
      stableStringHash('minecraft:dirt', -7),
      stableStringHash('minecraft:dirt', 0),
    );
  });

  it('observationVectorFromSnapshot zero-pads when features > 2', () => {
    const config = {
      include_position: false,
      include_velocity: false,
      include_orientation: false,
      include_vitals: false,
      include_inventory: true,
      inventory_slots: 2,
      inventory_features_per_slot: 4,
    };
    const snapshot: Snapshot = {
      tick: 0,
      inventory: {},
      position: { x: 0, y: 0, z: 0 },
      velocity: { x: 0, y: 0, z: 0 },
      yaw: 0,
      pitch: 0,
      health: 0,
      food: 0,
      oxygen: 0,
      experienceLevel: 0,
      experienceProgress: 0,
      hotbar: [{ type: 5, count: 2 }, null],
    };
    const vec = observationVectorFromSnapshot(snapshot, config);
    // 2 slots * 4 features = 8 entries; features 2,3 always zero.
    assert.equal(vec.length, 8);
    assert.equal(vec[2], 0);
    assert.equal(vec[3], 0);
    assert.equal(vec[6], 0);
    assert.equal(vec[7], 0);
  });

  it('observationVectorFromSnapshot returns empty vector when all sections disabled', () => {
    // The dimension-mismatch throw on line 134-136 is a defensive
    // assertion: snapshotObservation always feeds a self-consistent
    // pair, so the throw is unreachable in practice and not testable
    // without bypassing the public API. Cover the all-disabled path
    // instead, which is the most fragile shape — wrong length here
    // would be a coding bug, not user input.
    const config = {
      include_position: false,
      include_velocity: false,
      include_orientation: false,
      include_vitals: false,
      include_inventory: false,
    };
    assert.equal(computeObsDim(config), 0);
    const out = observationVectorFromSnapshot(
      {
        tick: 0,
        inventory: {},
        position: { x: 0, y: 0, z: 0 },
        velocity: { x: 0, y: 0, z: 0 },
        yaw: 0,
        pitch: 0,
        health: 0,
        food: 0,
        oxygen: 0,
        experienceLevel: 0,
        experienceProgress: 0,
        hotbar: [],
      },
      config,
    );
    assert.equal(out.length, 0);
  });

  it('snapshotObservation reads bot.position when entity is missing', () => {
    const bot = {
      tick: 7,
      position: { x: 1, y: 2, z: 3 },
      velocity: { x: 0, y: 0, z: 0 },
      inventory: { items: () => [], slots: [] },
    };
    const snap = snapshotObservation(bot, { include_inventory: false });
    assert.equal(snap.tick, 7);
    assert.equal(snap.position.x, 1);
    assert.equal(snap.position.y, 2);
    assert.equal(snap.position.z, 3);
  });

  it('snapshotObservation reads bot.oxygen when oxygenLevel is missing', () => {
    const bot = {
      entity: { position: { x: 0, y: 0, z: 0 }, velocity: { x: 0, y: 0, z: 0 } },
      time: { age: 0 },
      health: 20,
      food: 20,
      oxygen: 15, // fallback chain — `bot.oxygenLevel ?? bot.oxygen`
      inventory: { items: () => [], slots: [] },
    };
    const snap = snapshotObservation(bot, { include_inventory: false });
    // oxygen comes through as 15, normalised by max=20 → 0.75 in vitals (index 8).
    // Vitals start at offset 6 (pos:3 + vel:3 + orient: skipped? — default orient=true)
    // include_position=t (3) + include_velocity=t (3) + include_orientation=t (2)
    // = 8, then vitals: health, food, oxygen, expLevel, expProgress = indices 8..=12
    // oxygen is at index 10.
    assert.ok(snap.obs!.length >= 11);
    assert.ok(Math.abs(snap.obs![10] - 15 / 20) < 1e-9);
  });

  it('snapshotObservation does not use bot.time.time as a tick fallback', () => {
    // `bot.time.time` is Minecraft time-of-day (0..24000, wraps every MC day)
    // — it is NOT a monotonic tick counter. Using it would make trainer-side
    // sequence ordering / discount calculation jump backwards every day. With
    // only `bot.time.time` present the snapshot must fall through to `bot.tick`
    // (also missing here) and finally to the 0 default.
    const bot = {
      time: { time: 99 }, // `age` missing, only the bad `time` is present
      entity: { position: { x: 0, y: 0, z: 0 }, velocity: { x: 0, y: 0, z: 0 } },
      inventory: { items: () => [], slots: [] },
    };
    const snap = snapshotObservation(bot, { include_inventory: false });
    assert.notEqual(snap.tick, 99);
    assert.equal(snap.tick, 0);
  });

  it('snapshotObservation reads bot.tick when bot.time is missing', () => {
    const bot = {
      tick: 42,
      entity: { position: { x: 0, y: 0, z: 0 }, velocity: { x: 0, y: 0, z: 0 } },
      inventory: { items: () => [], slots: [] },
    };
    const snap = snapshotObservation(bot, { include_inventory: false });
    assert.equal(snap.tick, 42);
  });

  it('buildInventoryMap falls back to slots filter when items() is missing', () => {
    const bot = {
      time: { age: 0 },
      entity: { position: { x: 0, y: 0, z: 0 }, velocity: { x: 0, y: 0, z: 0 } },
      inventory: {
        // No items() function — exercises `(bot.inventory?.slots ?? []).filter(Boolean)`.
        slots: [null, { name: 'oak_log', count: 4 }, null, { type: 99, count: 1 }],
      },
    };
    const snap = snapshotObservation(bot, { include_inventory: false });
    assert.equal(snap.inventory.oak_log, 4);
    // Item with no name uses String(item.type) as the key.
    assert.equal(snap.inventory['99'], 1);
  });
});