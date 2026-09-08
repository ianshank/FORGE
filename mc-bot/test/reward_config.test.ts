import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import { mkdtemp, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parse } from 'smol-toml';

import {
  buildRewardConfig,
  canonicalRewardsSha256,
  combinedSchemaId,
  loadRewardConfig,
  NESTED_REWARD_PATH_KEY_CONFIG,
  NESTED_REWARD_PATH_KEY_CRAFTING,
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
    assert.throws(() => canonicalRewardsSha256({} as any));
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

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../..');

describe('reward_config — nested path folding', () => {
  it('fails closed when a nested path key is set and the file is missing', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'forge-rewards-'));
    const rewardsPath = join(dir, 'rewards.toml');
    await writeFile(
      rewardsPath,
      'schema_version = 1\n\n[[reward]]\nkind = "milestone"\nconfig_path = "missing.toml"\n',
    );
    await assert.rejects(
      () => loadRewardConfig(rewardsPath, parse),
      /nested reward file not found/,
    );
  });

  it('nested content change bumps hash; rename without content change does not', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'forge-rewards-'));
    const nestedA = join(dir, 'mil_a.toml');
    await writeFile(nestedA, '[milestones]\nfirst_wood = { reward = 10.0, once = true }\n');
    const rewardsA = join(dir, 'rewards.toml');
    await writeFile(
      rewardsA,
      'schema_version = 1\n\n[[reward]]\nkind = "milestone"\nconfig_path = "mil_a.toml"\n',
    );
    const hashA = (await loadRewardConfig(rewardsA, parse)).canonicalSha256();

    const nestedB = join(dir, 'mil_b.toml');
    await writeFile(nestedB, '[milestones]\nfirst_wood = { reward = 10.0, once = true }\n');
    const rewardsB = join(dir, 'rewards_b.toml');
    await writeFile(
      rewardsB,
      'schema_version = 1\n\n[[reward]]\nkind = "milestone"\nconfig_path = "mil_b.toml"\n',
    );
    const hashB = (await loadRewardConfig(rewardsB, parse)).canonicalSha256();
    assert.equal(hashA, hashB);

    await writeFile(nestedA, '[milestones]\nfirst_wood = { reward = 11.0, once = true }\n');
    const hashChanged = (await loadRewardConfig(rewardsA, parse)).canonicalSha256();
    assert.notEqual(hashA, hashChanged);
  });

  it('canonicalSha256 is memoized; nested FS rewrite does not change the same instance', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'forge-rewards-memo-'));
    const nested = join(dir, 'mil.toml');
    await writeFile(nested, '[milestones]\nfirst_wood = { reward = 10.0, once = true }\n');
    const rewards = join(dir, 'rewards.toml');
    await writeFile(
      rewards,
      'schema_version = 1\n\n[[reward]]\nkind = "milestone"\nconfig_path = "mil.toml"\n',
    );
    const cfg = await loadRewardConfig(rewards, parse);
    const handshake = cfg.canonicalSha256();
    await writeFile(nested, '[milestones]\nfirst_wood = { reward = 11.0, once = true }\n');
    assert.equal(
      cfg.canonicalSha256(),
      handshake,
      'load-time handshake hash must not track mid-run nested-file rewrites',
    );
    const reloaded = (await loadRewardConfig(rewards, parse)).canonicalSha256();
    assert.notEqual(reloaded, handshake);
  });

  it('shipped rewards.toml hash matches Rust pin (nested files folded)', async () => {
    const cfg = await loadRewardConfig(
      resolve(repoRoot, 'configs/minecraft/rewards.toml'),
      parse,
    );
    assert.equal(
      cfg.canonicalSha256(),
      '78f96c103767f3db7280175e92b8564937bcb5d505e4d75e8aab0c570d237f4b',
      'shipped rewards schema_id drift — Rust xlang_shipped_rewards_schema_id_folds_nested_files will also fail',
    );
  });

  it('hashes path strings when sourcePath is omitted (fixture pin)', () => {
    const cfg = buildRewardConfig({
      schema_version: 1,
      reward: [
        {
          kind: 'milestone',
          [NESTED_REWARD_PATH_KEY_CONFIG]: 'configs/minecraft/does-not-exist.toml',
        },
      ],
    });
    const h = cfg.canonicalSha256();
    assert.equal(h.length, 64);
  });

  it('fails closed on empty nested path', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'forge-rewards-'));
    const rewardsPath = join(dir, 'rewards.toml');
    await writeFile(
      rewardsPath,
      `schema_version = 1\n\n[[reward]]\nkind = "milestone"\n${NESTED_REWARD_PATH_KEY_CONFIG} = ""\n`,
    );
    await assert.rejects(() => loadRewardConfig(rewardsPath, parse), /must be a non-empty string/);
  });

  it('fails closed on non-string nested path', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'forge-rewards-'));
    const rewardsPath = join(dir, 'rewards.toml');
    await writeFile(
      rewardsPath,
      `schema_version = 1\n\n[[reward]]\nkind = "milestone"\n${NESTED_REWARD_PATH_KEY_CONFIG} = 1\n`,
    );
    await assert.rejects(() => loadRewardConfig(rewardsPath, parse), /must be a non-empty string/);
  });

  it('fails closed when crafting_config_path is set and missing', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'forge-rewards-'));
    const rewardsPath = join(dir, 'rewards.toml');
    await writeFile(
      rewardsPath,
      `schema_version = 1\n\n[[reward]]\nkind = "milestone"\n${NESTED_REWARD_PATH_KEY_CRAFTING} = "missing.toml"\n`,
    );
    await assert.rejects(() => loadRewardConfig(rewardsPath, parse), /nested reward file not found/);
  });

  it('repo-style nested path prefers sibling over cwd-relative shipped file', async () => {
    const unique = '[milestones]\nsibling_only = { reward = 99.0, once = true }\n';
    const dir = await mkdtemp(join(tmpdir(), 'forge-rewards-'));
    await writeFile(join(dir, 'milestone_rewards.toml'), unique);
    const rewardsPath = join(dir, 'rewards.toml');
    await writeFile(
      rewardsPath,
      `schema_version = 1\n\n[[reward]]\nkind = "milestone"\n${NESTED_REWARD_PATH_KEY_CONFIG} = "configs/minecraft/milestone_rewards.toml"\n`,
    );
    const fromRepoStyle = (await loadRewardConfig(rewardsPath, parse)).canonicalSha256();

    const dir2 = await mkdtemp(join(tmpdir(), 'forge-rewards-'));
    await writeFile(join(dir2, 'milestone_rewards.toml'), unique);
    const rewards2 = join(dir2, 'rewards.toml');
    await writeFile(
      rewards2,
      `schema_version = 1\n\n[[reward]]\nkind = "milestone"\n${NESTED_REWARD_PATH_KEY_CONFIG} = "milestone_rewards.toml"\n`,
    );
    const fromBasename = (await loadRewardConfig(rewards2, parse)).canonicalSha256();
    assert.equal(fromRepoStyle, fromBasename);
  });
});
