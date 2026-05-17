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
    assert.throws(() => canonicalSha256({}));
    assert.throws(() => canonicalSha256(null));
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
});
