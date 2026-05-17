import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import {
  buildRewardConfig,
  canonicalRewardsSha256,
  combinedSchemaId,
} from '../src/reward_config.js';

// Same fixture as the Rust `sample_toml()` in
// crates/forge-env-mc/src/reward_config.rs. The JSON-equivalent object
// is constructed manually here so we don't take a TOML parser dep on
// the test path; the byte-for-byte match with Rust is enforced by the
// xlang pin below.
function sampleData() {
  return {
    schema_version: 1,
    reward: [
      { kind: 'survival', value: 0.01 },
      {
        kind: 'distance_to_goal',
        clip: 100.0,
        target: { x: 0, y: 64, z: 0 },
      },
    ],
  };
}

describe('reward_config — buildRewardConfig', () => {
  it('builds a config with entries and a stable schemaSha256', () => {
    const cfg = buildRewardConfig(sampleData());
    assert.equal(cfg.entries.length, 2);
    const h = cfg.canonicalSha256();
    assert.equal(h.length, 64);
    assert.equal(cfg.canonicalSha256(), h);
  });

  it('rejects empty reward list', () => {
    assert.throws(() => buildRewardConfig({ schema_version: 1, reward: [] }));
    assert.throws(() => buildRewardConfig({ schema_version: 1 }));
  });

  it('rejects entry without kind', () => {
    assert.throws(() => buildRewardConfig({ reward: [{ value: 1 }] }), /missing string "kind"/);
  });

  it('rejects entry that is not an object', () => {
    assert.throws(() => buildRewardConfig({ reward: ['nope'] }), /must be an object/);
  });

  it('rejects non-object input', () => {
    assert.throws(() => buildRewardConfig(null));
    assert.throws(() => buildRewardConfig('text'));
  });
});

describe('reward_config — canonicalRewardsSha256', () => {
  it('changes when a numeric field changes', () => {
    const a = canonicalRewardsSha256(sampleData().reward);
    const modified = sampleData();
    modified.reward[1].clip = 50.0;
    const b = canonicalRewardsSha256(modified.reward);
    assert.notEqual(a, b);
  });

  it('rejects non-array input', () => {
    assert.throws(() => canonicalRewardsSha256({}));
  });
});

describe('reward_config — combinedSchemaId', () => {
  it('is deterministic and argument-order-sensitive', () => {
    const a = combinedSchemaId('aaa', 'bbb');
    const b = combinedSchemaId('aaa', 'bbb');
    const c = combinedSchemaId('bbb', 'aaa');
    assert.equal(a, b);
    assert.notEqual(a, c);
    assert.equal(a.length, 64);
  });
});

describe('reward_config — xlang regression', () => {
  // Pinned-fixture cross-language gate. MUST equal the Rust value in
  // crates/forge-env-mc/src/reward_config.rs::xlang_rewards_schema_id_pinned_to_known_good.
  // If both fail simultaneously after a TOML serialiser update, debug
  // both sides before bumping.
  it('hash matches Rust pinned constant', () => {
    const cfg = buildRewardConfig(sampleData());
    assert.equal(
      cfg.canonicalSha256(),
      '451b10f995371924a374633e5c42deab35c137fbbc65bc8f551bf2bd7844b478',
      'rewards schema_id drift — Rust xlang_rewards_schema_id_pinned_to_known_good will also fail',
    );
  });
});
