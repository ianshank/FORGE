import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { buildActionMap } from '../src/action_map.js';
import {
  buildConfigBundleFromObjects,
  normalizeEnvConfig,
  normalizeResetConfig,
  resolveConfigPath,
  loadEnvConfig,
  loadResetConfig,
  loadConfigBundle,
  DEFAULT_CONFIG_DIR
} from '../src/config.js';

describe('config', () => {
  it('normalizes env config with Rust-compatible root fields', () => {
    const cfg = normalizeEnvConfig({
      ws_url: 'ws://0.0.0.0:9000',
      observation: { include_inventory: false },
    });
    assert.equal(cfg.websocket!.host, '0.0.0.0');
    assert.equal(cfg.websocket!.port, 9000);
    assert.equal(cfg.observation.include_inventory, false);
    assert.equal(cfg.observation.include_position, true);
  });

  it('rejects non-WebSocket URLs', () => {
    assert.throws(() => normalizeEnvConfig({ ws_url: 'http://127.0.0.1:9000' }), /ws_url/);
  });
  
  it('rejects invalid or missing WebSocket ports', () => {
    assert.throws(() => normalizeEnvConfig({ ws_url: 'ws://127.0.0.1:0' }), /ws_url port/);
  });

  it('normalizes reset config with teleport defaults', () => {
    const cfg = normalizeResetConfig({ teleport: { spawn: { x: 1, y: 70, z: 2 } } });
    assert.equal(cfg.strategy, 'teleport');
    assert.deepEqual(cfg.teleport!.spawn, { x: 1, y: 70, z: 2 });
    assert.equal(cfg.teleport!.restore_health, true);
  });

  it('resolves relative config paths from repo root', () => {
    const root = resolve('repo-root');
    assert.equal(resolveConfigPath('configs/minecraft/env.toml', root), resolve(root, 'configs/minecraft/env.toml'));
  });

  it('rejects empty config paths', () => {
    assert.throws(() => resolveConfigPath(''), /config path must be a non-empty string/);
  });

  it('builds schema bundle from parsed objects', () => {
    const actionMap = buildActionMap({
      action: [{ id: 0, kind: 'noop', ticks: 1 }],
    });
    const bundle = buildConfigBundleFromObjects({
      env: { ws_url: 'ws://127.0.0.1:8765' },
      reset: {},
      actionMapData: actionMap,
      rewardData: { reward: [{ kind: 'survival', value: 0.25 }] },
    });
    assert.equal(bundle.actionMap.actionCount, 1);
    assert.equal(bundle.rewardFn({ prev: {}, curr: {} }), 0.25);
    assert.match(bundle.schemaId, /^[0-9a-f]{64}$/);
  });

  it('loadEnvConfig loads from toml', async () => {
    const tomlParse = (raw: string) => ({ ws_url: 'ws://localhost:9999' });
    const cfg = await loadEnvConfig('package.json', { tomlParse });
    assert.equal(cfg.websocket!.port, 9999);
  });

  it('loadResetConfig loads from toml', async () => {
    const tomlParse = (raw: string) => ({ strategy: 'teleport' });
    const cfg = await loadResetConfig('package.json', { tomlParse });
    assert.equal(cfg.strategy, 'teleport');
  });

  it('loadConfigBundle assembles all parts and handles missing block embeddings', async () => {
    let parseCalls = 0;
    const tomlParse = (raw: string) => {
      parseCalls++;
      if (parseCalls === 1) return { ws_url: 'ws://127.0.0.1:8000' }; // env
      if (parseCalls === 2) return { action: [{ id: 0, kind: 'noop', ticks: 1 }] }; // action map
      if (parseCalls === 3) return { reward: [{ kind: 'survival', value: 0.5 }] }; // rewards
      if (parseCalls === 4) return {}; // reset
      throw new Error('Block embeddings not found'); // block_embeddings
    };

    const bundle = await loadConfigBundle({ tomlParse });
    assert.equal(bundle.env.websocket!.port, 8000);
    assert.equal(bundle.actionMap.actionCount, 1);
    assert.equal(bundle.rewardFn({ prev: {}, curr: {} }), 0.5);
    assert.deepEqual(bundle.env.observation.block_embeddings, {});
    assert.equal(parseCalls, 5);
  });
});