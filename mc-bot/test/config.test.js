import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { resolve } from 'node:path';

import { buildActionMap } from '../src/action_map.js';
import {
  buildConfigBundleFromObjects,
  normalizeEnvConfig,
  normalizeResetConfig,
  resolveConfigPath,
} from '../src/config.js';

describe('config', () => {
  it('normalizes env config with Rust-compatible root fields', () => {
    const cfg = normalizeEnvConfig({
      ws_url: 'ws://0.0.0.0:9000',
      observation: { include_inventory: false },
    });
    assert.equal(cfg.websocket.host, '0.0.0.0');
    assert.equal(cfg.websocket.port, 9000);
    assert.equal(cfg.observation.include_inventory, false);
    assert.equal(cfg.observation.include_position, true);
  });

  it('rejects non-WebSocket URLs', () => {
    assert.throws(() => normalizeEnvConfig({ ws_url: 'http://127.0.0.1:9000' }), /ws_url/);
  });

  it('normalizes reset config with teleport defaults', () => {
    const cfg = normalizeResetConfig({ teleport: { spawn: { x: 1, y: 70, z: 2 } } });
    assert.equal(cfg.strategy, 'teleport');
    assert.deepEqual(cfg.teleport.spawn, { x: 1, y: 70, z: 2 });
    assert.equal(cfg.teleport.restore_health, true);
  });

  it('resolves relative config paths from repo root', () => {
    const root = resolve('repo-root');
    assert.equal(resolveConfigPath('configs/minecraft/env.toml', root), resolve(root, 'configs/minecraft/env.toml'));
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
    assert.equal(bundle.rewardFn({}), 0.25);
    assert.match(bundle.schemaId, /^[0-9a-f]{64}$/);
  });
});