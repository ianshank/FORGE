import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import { executeAction } from '../src/actions.js';

function stubBot() {
  return {
    controls: [],
    waits: [],
    quickBarSlot: null,
    activated: 0,
    swings: [],
    looks: [],
    entity: { yaw: 0, pitch: 0 },
    setControlState(control, state) {
      this.controls.push([control, state]);
    },
    async waitForTicks(ticks) {
      this.waits.push(ticks);
    },
    setQuickBarSlot(slot) {
      this.quickBarSlot = slot;
    },
    activateItem() {
      this.activated += 1;
    },
    swingArm(hand) {
      this.swings.push(hand);
    },
    async look(yaw, pitch, force) {
      this.looks.push({ yaw, pitch, force });
    },
  };
}

describe('executeAction', () => {
  it('holds and releases move controls for configured ticks', async () => {
    const bot = stubBot();
    const result = await executeAction(bot, { kind: 'move', direction: 'forward', ticks: 4 });
    assert.deepEqual(bot.controls, [['forward', true], ['forward', false]]);
    assert.deepEqual(bot.waits, [4]);
    assert.equal(result.ticks, 4);
  });

  it('clears jump control after waiting', async () => {
    const bot = stubBot();
    await executeAction(bot, { kind: 'jump' });
    assert.deepEqual(bot.controls, [['jump', true], ['jump', false]]);
  });

  it('selects hotbar and activates item for place', async () => {
    const bot = stubBot();
    await executeAction(bot, { kind: 'place', hotbar_slot: 3 });
    assert.equal(bot.quickBarSlot, 3);
    assert.equal(bot.activated, 1);
  });

  it('turns look deltas into absolute radians', async () => {
    const bot = stubBot();
    await executeAction(bot, { kind: 'look', yaw_deg: 90, pitch_deg: -45 });
    assert.equal(bot.looks.length, 1);
    assert.ok(Math.abs(bot.looks[0].yaw - Math.PI / 2) < 1e-9);
    assert.ok(Math.abs(bot.looks[0].pitch + Math.PI / 4) < 1e-9);
    assert.equal(bot.looks[0].force, true);
  });

  it('rejects invalid action kinds and slots', async () => {
    const bot = stubBot();
    await assert.rejects(() => executeAction(bot, { kind: 'magic' }), /unknown action kind/);
    await assert.rejects(() => executeAction(bot, { kind: 'select_slot', hotbar_slot: 99 }), /hotbar_slot/);
  });

  it('noop waits the requested ticks without touching controls', async () => {
    const bot = stubBot();
    const result = await executeAction(bot, { kind: 'noop', ticks: 3 });
    assert.deepEqual(bot.controls, []);
    assert.deepEqual(bot.waits, [3]);
    assert.equal(result.ticks, 3);
  });

  it('attack swings the arm when no nearby entity is exposed', async () => {
    const bot = stubBot();
    await executeAction(bot, { kind: 'attack' });
    assert.deepEqual(bot.swings, ['right']);
    assert.equal(bot.waits.length, 1);
  });

  it('attack hits nearestEntity when bot exposes both nearestEntity and attack', async () => {
    const bot = stubBot();
    const target = { id: 'mob-7' };
    bot.nearestEntity = () => target;
    let hit;
    bot.attack = (entity) => { hit = entity; };
    await executeAction(bot, { kind: 'attack' });
    assert.equal(hit, target);
    // No fallback swing when attack-on-target path succeeded.
    assert.deepEqual(bot.swings, []);
  });

  it('use activates the held item via bot.activateItem', async () => {
    const bot = stubBot();
    await executeAction(bot, { kind: 'use' });
    assert.equal(bot.activated, 1);
  });

  it('use throws when bot lacks activateItem', async () => {
    const bot = stubBot();
    delete bot.activateItem;
    await assert.rejects(
      () => executeAction(bot, { kind: 'use' }),
      /activateItem/,
    );
  });

  it('place throws when bot lacks activateItem after selecting slot', async () => {
    const bot = stubBot();
    delete bot.activateItem;
    await assert.rejects(
      () => executeAction(bot, { kind: 'place', hotbar_slot: 0 }),
      /activateItem/,
    );
  });

  it('select_slot moves quickbar without activating', async () => {
    const bot = stubBot();
    await executeAction(bot, { kind: 'select_slot', hotbar_slot: 7 });
    assert.equal(bot.quickBarSlot, 7);
    assert.equal(bot.activated, 0);
  });

  it('look throws when bot lacks look method', async () => {
    const bot = stubBot();
    delete bot.look;
    await assert.rejects(
      () => executeAction(bot, { kind: 'look', yaw_deg: 0, pitch_deg: 0 }),
      /look/,
    );
  });

  it('move rejects unknown direction with descriptive error', async () => {
    const bot = stubBot();
    await assert.rejects(
      () => executeAction(bot, { kind: 'move', direction: 'upward' }),
      /unknown move direction: upward/,
    );
  });

  it('move releases control even when waitForTicks throws', async () => {
    const bot = stubBot();
    bot.waitForTicks = async () => {
      throw new Error('tick scheduler offline');
    };
    await assert.rejects(
      () => executeAction(bot, { kind: 'move', direction: 'forward', ticks: 2 }),
      /tick scheduler offline/,
    );
    // Both press and release recorded — the finally{} guard fired.
    assert.deepEqual(bot.controls, [['forward', true], ['forward', false]]);
  });

  it('falls back to delay when bot lacks waitForTicks', async () => {
    const bot = stubBot();
    delete bot.waitForTicks;
    const t0 = Date.now();
    await executeAction(bot, { kind: 'noop', ticks: 2 }, { tickMs: 5 });
    const elapsed = Date.now() - t0;
    // 2 ticks * 5ms each = 10ms; allow generous slack on Windows timers.
    assert.ok(elapsed >= 9, `expected >=9ms elapsed, got ${elapsed}ms`);
  });

  it('rejects entirely missing bot or action', async () => {
    await assert.rejects(() => executeAction(null, { kind: 'noop' }), /bot is required/);
    await assert.rejects(() => executeAction({}, null), /action is required/);
  });

  it('hotbar slot lower bound rejection (negative)', async () => {
    const bot = stubBot();
    await assert.rejects(
      () => executeAction(bot, { kind: 'select_slot', hotbar_slot: -1 }),
      /hotbar_slot/,
    );
  });
});