import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import { startViewer } from '../src/viewer.js';

/**
 * Build a fake `prismarine-viewer` module shape. The real module
 * exposes `mineflayer(bot, opts)`. We need to test that the resolver
 * tolerates the ESM-interop variations we see in the wild.
 */
function makeMineflayerStart() {
  const calls: any[] = [];
  const fn = (bot: any, opts: any) => {
    calls.push({ bot, opts });
    return { dispose() { /* no-op */ } };
  };
  fn.calls = calls;
  return fn;
}

describe('startViewer', () => {
  it('returns null when viewer is not enabled (no module import attempted)', async () => {
    // `viewerModule` deliberately absent — would crash if startViewer tried to import.
    const out = await startViewer({}, { enabled: false });
    assert.equal(out, null);
  });

  it('resolves viewer when module exposes `mineflayer` directly', async () => {
    const start = makeMineflayerStart();
    const viewerModule = { mineflayer: start };
    const bot = { id: 'bot-direct' };
    const ret = await startViewer(
      bot,
      { enabled: true, port: 3007, host: '127.0.0.1', first_person: true, view_distance: 6 },
      { viewerModule },
    );
    assert.equal(start.calls.length, 1);
    assert.equal(start.calls[0].bot, bot);
    assert.equal(start.calls[0].opts.port, 3007);
    assert.equal(start.calls[0].opts.host, '127.0.0.1');
    assert.equal(start.calls[0].opts.firstPerson, true);
    assert.equal(start.calls[0].opts.viewDistance, 6);
    assert.ok(ret && typeof ret.dispose === 'function');
  });

  it('resolves viewer when module exposes `default.mineflayer` (ESM-interop default-export)', async () => {
    const start = makeMineflayerStart();
    const viewerModule = { default: { mineflayer: start } };
    const bot = { id: 'bot-default-mineflayer' };
    await startViewer(bot, { enabled: true, port: 3008 }, { viewerModule });
    assert.equal(start.calls.length, 1);
    assert.equal(start.calls[0].opts.port, 3008);
  });

  it('resolves viewer when module exposes `default` as the function (CJS-default fallthrough)', async () => {
    const start = makeMineflayerStart();
    const viewerModule = { default: start };
    const bot = { id: 'bot-default-fn' };
    await startViewer(bot, { enabled: true, port: 3009 }, { viewerModule });
    assert.equal(start.calls.length, 1);
    assert.equal(start.calls[0].bot, bot);
  });

  it('throws a precise error when no resolution path yields a function', async () => {
    // Module shape with neither `mineflayer` nor a callable default.
    const viewerModule = { default: { something: 'else' } };
    await assert.rejects(
      () => startViewer({}, { enabled: true }, { viewerModule }),
      /prismarine-viewer does not expose a mineflayer viewer function/,
    );
  });

  it('passes through all known viewer config knobs verbatim', async () => {
    const start = makeMineflayerStart();
    const viewerModule = { mineflayer: start };
    const config = {
      enabled: true,
      port: 4242,
      host: '0.0.0.0',
      first_person: false,
      view_distance: 12,
    };
    await startViewer({ tag: 'bot' }, config, { viewerModule });
    const [{ opts }] = start.calls;
    assert.equal(opts.port, 4242);
    assert.equal(opts.host, '0.0.0.0');
    assert.equal(opts.firstPerson, false);
    assert.equal(opts.viewDistance, 12);
  });

  it('does not import the real prismarine-viewer when viewerModule is injected', async () => {
    // If startViewer ever tried to dynamic-import "prismarine-viewer"
    // we'd see it propagate from the bare specifier; injecting bypass.
    const start = makeMineflayerStart();
    await startViewer({}, { enabled: true }, { viewerModule: { mineflayer: start } });
    assert.equal(start.calls.length, 1);
  });
});
