import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import { applyReset } from '../src/reset.js';

function stubBot() {
  const calls = [];
  return {
    chat(s) { calls.push(s); },
    _calls: calls,
  };
}

describe('reset — applyReset', () => {
  it('teleports to spawn + clears + restores by default', async () => {
    const bot = stubBot();
    await applyReset(bot, {
      strategy: 'teleport',
      teleport: {
        spawn: { x: 0, y: 64, z: 0 },
        clear_inventory: true,
        restore_health: true,
        restore_food: true,
      },
    });
    const cmds = bot._calls;
    assert.equal(cmds[0], '/tp @s 0 64 0 0 0');
    assert.equal(cmds[1], '/clear @s');
    assert.match(cmds[2], /instant_health/);
    assert.match(cmds[3], /saturation/);
  });

  it('respects clear/health/food toggles', async () => {
    const bot = stubBot();
    await applyReset(bot, {
      strategy: 'teleport',
      teleport: {
        spawn: { x: 1, y: 2, z: 3 },
        clear_inventory: false,
        restore_health: false,
        restore_food: false,
      },
    });
    assert.equal(bot._calls.length, 1);
    assert.equal(bot._calls[0], '/tp @s 1 2 3 0 0');
  });

  it('throws on arena strategy in v1', async () => {
    const bot = stubBot();
    await assert.rejects(
      applyReset(bot, { strategy: 'arena' }),
      /arena reset strategy not implemented/,
    );
  });

  it('throws on unknown strategy', async () => {
    const bot = stubBot();
    await assert.rejects(applyReset(bot, { strategy: 'magic' }), /unknown reset strategy/);
  });

  it('throws if bot.chat is missing', async () => {
    await assert.rejects(applyReset({}, { strategy: 'teleport' }), /bot must expose chat/);
  });

  it('honours a custom selector (username) over @s default', async () => {
    const bot = stubBot();
    await applyReset(bot, {
      strategy: 'teleport',
      teleport: {
        spawn: { x: 5, y: 10, z: 15 },
        selector: 'ForgeBot',
        clear_inventory: true,
      },
    });
    assert.equal(bot._calls[0], '/tp ForgeBot 5 10 15 0 0');
    assert.equal(bot._calls[1], '/clear ForgeBot');
  });

  it('falls back to @s on empty or non-string selector', async () => {
    for (const bad of ['', null, undefined, 42]) {
      const bot = stubBot();
      await applyReset(bot, {
        strategy: 'teleport',
        teleport: { spawn: { x: 0, y: 0, z: 0 }, selector: bad },
      });
      assert.match(bot._calls[0], /^\/tp @s /);
    }
  });
});
