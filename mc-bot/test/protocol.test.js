import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import {
  SCHEMA_VERSION,
  helloMsg,
  observationMsg,
  errorMsg,
  parseClientMsg,
} from '../src/protocol.js';

describe('protocol — helloMsg', () => {
  it('builds a well-formed Hello with pinned schema_version', () => {
    const m = helloMsg({ actionCount: 12, obsDim: 960, schemaId: 'abc' });
    assert.equal(m.type, 'hello');
    assert.equal(m.schema_version, SCHEMA_VERSION);
    assert.equal(m.action_count, 12);
    assert.equal(m.obs_dim, 960);
    assert.equal(m.schema_id, 'abc');
  });

  it('rejects non-positive actionCount', () => {
    assert.throws(() => helloMsg({ actionCount: 0, obsDim: 10, schemaId: 'a' }));
    assert.throws(() => helloMsg({ actionCount: -1, obsDim: 10, schemaId: 'a' }));
  });

  it('rejects empty schema_id', () => {
    assert.throws(() => helloMsg({ actionCount: 1, obsDim: 10, schemaId: '' }));
  });
});

describe('protocol — observationMsg', () => {
  it('builds a well-formed Observation', () => {
    const m = observationMsg({
      tick: 7,
      obs: [0.1, 0.2],
      reward: 1.5,
      terminated: false,
      truncated: false,
      info: { event: 'spawned' },
    });
    assert.equal(m.type, 'observation');
    assert.equal(m.tick, 7);
    assert.deepEqual(m.obs, [0.1, 0.2]);
    assert.equal(m.reward, 1.5);
    assert.equal(m.terminated, false);
    assert.deepEqual(m.info, { event: 'spawned' });
  });

  it('rejects non-finite reward', () => {
    assert.throws(() =>
      observationMsg({ tick: 0, obs: [], reward: NaN, terminated: false, truncated: false }),
    );
  });

  it('defaults info to {}', () => {
    const m = observationMsg({ tick: 0, obs: [], reward: 0, terminated: false, truncated: false });
    assert.deepEqual(m.info, {});
  });
});

describe('protocol — errorMsg', () => {
  it('coerces non-string code/message to strings', () => {
    const m = errorMsg(404, 'not found');
    assert.equal(m.type, 'error');
    assert.equal(m.code, '404');
    assert.equal(m.message, 'not found');
  });
});

describe('protocol — parseClientMsg', () => {
  it('parses reset with numeric seed', () => {
    const m = parseClientMsg({ type: 'reset', seed: 42 });
    assert.deepEqual(m, { type: 'reset', seed: 42 });
  });

  it('parses reset with null seed', () => {
    const m = parseClientMsg({ type: 'reset', seed: null });
    assert.deepEqual(m, { type: 'reset', seed: null });
  });

  it('parses step', () => {
    const m = parseClientMsg({ type: 'step', action_id: 5 });
    assert.deepEqual(m, { type: 'step', action_id: 5 });
  });

  it('parses close', () => {
    const m = parseClientMsg({ type: 'close' });
    assert.deepEqual(m, { type: 'close' });
  });

  it('accepts JSON string input', () => {
    const m = parseClientMsg('{"type":"step","action_id":3}');
    assert.equal(m.type, 'step');
    assert.equal(m.action_id, 3);
  });

  it('rejects unknown type', () => {
    assert.throws(() => parseClientMsg({ type: 'nuke' }));
  });

  it('rejects negative or non-integer action_id', () => {
    assert.throws(() => parseClientMsg({ type: 'step', action_id: -1 }));
    assert.throws(() => parseClientMsg({ type: 'step', action_id: 1.5 }));
  });

  it('rejects non-object input', () => {
    assert.throws(() => parseClientMsg(null));
    assert.throws(() => parseClientMsg('not json'));
  });

  it('rejects unsafe-integer or non-integer seeds', () => {
    // Float seed — would lose precision on Rust's u64.
    assert.throws(() => parseClientMsg({ type: 'reset', seed: 1.5 }));
    // Negative seed — u64 can't represent it.
    assert.throws(() => parseClientMsg({ type: 'reset', seed: -1 }));
    // Above Number.MAX_SAFE_INTEGER — silently corrupted on the wire.
    assert.throws(() =>
      parseClientMsg({ type: 'reset', seed: Number.MAX_SAFE_INTEGER + 2 }),
    );
    // NaN / Infinity.
    assert.throws(() => parseClientMsg({ type: 'reset', seed: Number.NaN }));
    assert.throws(() => parseClientMsg({ type: 'reset', seed: Infinity }));
  });

  it('accepts seed at the safe-integer boundary', () => {
    const m = parseClientMsg({ type: 'reset', seed: Number.MAX_SAFE_INTEGER });
    assert.equal(m.seed, Number.MAX_SAFE_INTEGER);
  });
});

describe('protocol — xlang regression', () => {
  // Pinned-fixture cross-language gate. This exact value MUST equal
  // the Rust constant in:
  //   crates/forge-env-mc/src/protocol.rs::SCHEMA_VERSION
  // verified by Rust-side xlang_schema_version_pinned_to_known_good.
  //
  // If you bump the protocol, change BOTH constants in the same PR;
  // otherwise both this test and the Rust counterpart will fail.
  const PINNED_SCHEMA_VERSION = 1;

  it('xlang SCHEMA_VERSION matches Rust', () => {
    assert.equal(
      SCHEMA_VERSION,
      PINNED_SCHEMA_VERSION,
      'protocol SCHEMA_VERSION drift — Rust ' +
        'xlang_schema_version_pinned_to_known_good will also fail',
    );
  });
});
