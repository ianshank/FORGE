import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import { buildActionMap } from '../src/action_map.js';

function sampleData() {
  return {
    schema_version: 1,
    action: [
      { id: 0, kind: 'noop', ticks: 1 },
      { id: 1, kind: 'jump' },
      { id: 2, kind: 'attack' },
    ],
  };
}

describe('action_map — buildActionMap', () => {
  it('builds a well-formed map with action_count and schema_id', () => {
    const m = buildActionMap(sampleData());
    assert.equal(m.actionCount, 3);
    assert.equal(typeof m.schemaId, 'string');
    assert.equal(m.schemaId.length, 64);
    assert.deepEqual(m.get(1), { id: 1, kind: 'jump' });
  });

  it('rejects empty entries', () => {
    assert.throws(() => buildActionMap({ action: [] }));
  });

  it('rejects sparse ids', () => {
    const d = { action: [
      { id: 0, kind: 'noop' },
      { id: 2, kind: 'jump' },
    ] };
    assert.throws(() => buildActionMap(d), /dense/);
  });

  it('rejects duplicate ids', () => {
    const d = { action: [
      { id: 0, kind: 'noop' },
      { id: 0, kind: 'jump' },
    ] };
    assert.throws(() => buildActionMap(d), /duplicate/);
  });

  it('rejects unknown kind', () => {
    assert.throws(() => buildActionMap({ action: [{ id: 0, kind: 'nuke' }] }));
  });

  it('rejects invalid move direction', () => {
    assert.throws(() => buildActionMap({ action: [
      { id: 0, kind: 'move', direction: 'diagonal', ticks: 1 },
    ] }));
  });

  it('rejects out-of-range hotbar_slot', () => {
    assert.throws(() => buildActionMap({ action: [
      { id: 0, kind: 'place', hotbar_slot: 99 },
    ] }));
  });

  it('rejects non-finite look angles', () => {
    assert.throws(() => buildActionMap({ action: [
      { id: 0, kind: 'look', yaw_deg: NaN, pitch_deg: 0 },
    ] }));
  });

  it('schema_id is stable across runs', () => {
    const a = buildActionMap(sampleData()).schemaId;
    const b = buildActionMap(sampleData()).schemaId;
    assert.equal(a, b);
  });

  it('schema_id is invariant under entry reordering (sort by id)', () => {
    const d1 = sampleData();
    const d2 = { ...sampleData(), action: [...sampleData().action].reverse() };
    assert.equal(buildActionMap(d1).schemaId, buildActionMap(d2).schemaId);
  });
});
