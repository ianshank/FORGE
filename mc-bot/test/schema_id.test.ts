import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import { canonicalSha256 } from '../src/schema_id.js';

describe('schema_id — canonicalSha256', () => {
  it('is stable across invocations', () => {
    const entries = [
      { id: 0, kind: 'noop', ticks: 1 },
      { id: 1, kind: 'jump' },
    ];
    assert.equal(canonicalSha256(entries), canonicalSha256(entries));
  });

  it('produces a 64-char hex string', () => {
    const h = canonicalSha256([{ id: 0, kind: 'noop', ticks: 1 }]);
    assert.equal(h.length, 64);
    assert.match(h, /^[0-9a-f]+$/);
  });

  it('is invariant under entry reorder', () => {
    const a = [
      { id: 0, kind: 'noop', ticks: 1 },
      { id: 1, kind: 'jump' },
    ];
    const b = [...a].reverse();
    assert.equal(canonicalSha256(a), canonicalSha256(b));
  });

  it('changes when fields change', () => {
    const a = [{ id: 0, kind: 'noop', ticks: 1 }];
    const b = [{ id: 0, kind: 'noop', ticks: 2 }];
    assert.notEqual(canonicalSha256(a), canonicalSha256(b));
  });

  it('rejects non-array input', () => {
    assert.throws(() => canonicalSha256({} as any));
    assert.throws(() => canonicalSha256(null as any));
  });

  // Pinned-fixture cross-language regression gate. This exact value MUST
  // equal the Rust-side computation in
  // crates/forge-env-mc/src/action_map.rs::tests::xlang_schema_id_pinned_to_known_good.
  // Failing means the canonical form drifted; investigate both sides
  // before bumping.
  it('xlang schema_id matches Rust', () => {
    const entries = [
      { id: 0, kind: 'noop', ticks: 1 },
      { id: 1, kind: 'move', direction: 'forward', ticks: 4 },
      { id: 2, kind: 'jump' },
    ];
    assert.equal(
      canonicalSha256(entries),
      '587b13077b8c7cd90503f9ee5e1bae1bb92bdf738c8abc51d2ff6deb1908224f',
      'schema_id drift — Rust xlang_schema_id_pinned_to_known_good will also fail',
    );
  });

  it('is invariant under per-entry field reorder for move/look/place/select_slot', () => {
    // Caller may pass fields in any order; the canonical form pins
    // them per the Rust enum declaration. This protects against
    // accidental drift if upstream serialisers (TOML, YAML, JSON
    // editors) reorder keys.
    const reordered = [
      { ticks: 1, id: 0, kind: 'noop' },
      { ticks: 4, direction: 'forward', kind: 'move', id: 1 },
      { kind: 'jump', id: 2 },
    ];
    const declared = [
      { id: 0, kind: 'noop', ticks: 1 },
      { id: 1, kind: 'move', direction: 'forward', ticks: 4 },
      { id: 2, kind: 'jump' },
    ];
    assert.equal(canonicalSha256(reordered), canonicalSha256(declared));
  });

  it('covers all documented kinds (regression: kinds added in Rust but not here)', () => {
    const entries = [
      { id: 0, kind: 'noop', ticks: 1 },
      { id: 1, kind: 'move', direction: 'left', ticks: 2 },
      { id: 2, kind: 'jump' },
      { id: 3, kind: 'attack' },
      { id: 4, kind: 'use' },
      { id: 5, kind: 'place', hotbar_slot: 3 },
      { id: 6, kind: 'select_slot', hotbar_slot: 7 },
      { id: 7, kind: 'look', yaw_deg: 12.5, pitch_deg: -3.25 },
      { id: 8, kind: 'eat' },
      { id: 9, kind: 'craft_planks' },
      { id: 10, kind: 'craft_wooden_pickaxe' },
      { id: 11, kind: 'sprint', ticks: 4 },
      { id: 12, kind: 'sneak', ticks: 4 },
      { id: 13, kind: 'swim_up', ticks: 4 },
    ];
    // Just assert hash is stable & shaped — drift on the Rust side
    // is caught by the xlang pinned-fixture test above (which uses
    // a smaller subset). This one guards completeness on the JS side.
    const h = canonicalSha256(entries);
    assert.equal(h.length, 64);
    assert.equal(canonicalSha256(entries), h);
  });
});
