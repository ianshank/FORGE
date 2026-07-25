import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const registry = require('prismarine-registry')('1.20.4');

import { executeAction as executeActionRaw } from '../src/actions/index.js';
const executeAction = executeActionRaw as any;

function stubBot(): any {
  return {
    controls: [] as any[],
    waits: [] as any[],
    quickBarSlot: null as any,
    activated: 0,
    swings: [] as any[],
    looks: [] as any[],
    entity: { yaw: 0, pitch: 0 },
    setControlState(control: any, state: any) {
      this.controls.push([control, state]);
    },
    async waitForTicks(ticks: any) {
      this.waits.push(ticks);
    },
    setQuickBarSlot(slot: any) {
      this.quickBarSlot = slot;
    },
    activateItem() {
      this.activated += 1;
    },
    swingArm(hand: any) {
      this.swings.push(hand);
    },
    async look(yaw: any, pitch: any, force: any) {
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
    bot.attack = (entity: any) => { hit = entity; };
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

  it('sprint sets sprint control state and waits', async () => {
    const bot = stubBot();
    const result = await executeAction(bot, { kind: 'sprint', ticks: 4 });
    assert.deepEqual(bot.controls, [['sprint', true], ['sprint', false]]);
    assert.deepEqual(bot.waits, [4]);
    assert.equal(result.ticks, 4);
  });

  it('sneak sets sneak control state and waits', async () => {
    const bot = stubBot();
    const result = await executeAction(bot, { kind: 'sneak', ticks: 4 });
    assert.deepEqual(bot.controls, [['sneak', true], ['sneak', false]]);
    assert.deepEqual(bot.waits, [4]);
    assert.equal(result.ticks, 4);
  });

  it('swim_up sets jump control state and waits', async () => {
    const bot = stubBot();
    const result = await executeAction(bot, { kind: 'swim_up', ticks: 4 });
    assert.deepEqual(bot.controls, [['jump', true], ['jump', false]]);
    assert.deepEqual(bot.waits, [4]);
    assert.equal(result.ticks, 4);
  });

  it('eat tries to equip and consume food', async () => {
    const bot = stubBot();
    bot.inventory = {
      items: () => [{ name: 'bread' }, { name: 'dirt' }]
    };
    let equipped = null;
    let consumedCount = 0;
    bot.equip = async (item: any, hand: any) => {
      equipped = { item, hand };
    };
    bot.consume = async () => {
      consumedCount++;
    };

    const result = await executeAction(bot, { kind: 'eat' });
    assert.equal(result.consumed, true);
    assert.deepEqual(equipped, { item: { name: 'bread' }, hand: 'hand' });
    assert.equal(consumedCount, 1);
  });

  it('eat returns consumed=false if no edible food in inventory', async () => {
    const bot = stubBot();
    bot.inventory = {
      items: () => [{ name: 'dirt' }]
    };
    const result = await executeAction(bot, { kind: 'eat' });
    assert.equal(result.consumed, false);
  });

  it('craft_planks finds wooden planks recipe and crafts them', async () => {
    const bot = stubBot();
    bot.version = '1.20.4';
    bot.registry = registry;
    bot.inventory = { items: () => [{ type: registry.itemsByName.oak_log.id, count: 1 }] };
    let craftedRecipe = null;
    let craftedAmount = 0;
    bot.recipesFor = (id: any, metadata: any, count: any, table: any) => {
      // Fake recipesFor to return the required recipe so bot.craft doesn't crash in test mock
      const recipe = { id: 'recipe-oak' };
      return [recipe];
    };
    bot.craft = async (recipe: any, count: any, table: any) => {
      craftedRecipe = recipe;
      craftedAmount = count;
    };

    const result = await executeAction(bot, { kind: 'craft_planks' });
    assert.equal(result.crafted, true);
    assert.equal(result.item, 'oak_planks');
    assert.deepEqual(craftedRecipe, { id: 'recipe-oak' });
    assert.equal(craftedAmount, 1);
  });

  it('craft_wooden_pickaxe crafts wooden pickaxe using table if present', async () => {
    const bot = stubBot();
    bot.version = '1.20.4';
    bot.registry = registry;
    bot.inventory = { 
      items: () => [
        { type: registry.itemsByName.oak_planks.id, count: 3 },
        { type: registry.itemsByName.stick.id, count: 2 }
      ] 
    };
    let tableFoundId = null;
    bot.findBlock = (options: any) => {
      tableFoundId = options.matching;
      return { position: { x: 0, y: 0, z: 0 } };
    };
    let recipesQueryTable = null;
    bot.recipesFor = (id: any, metadata: any, count: any, table: any) => {
      recipesQueryTable = table;
      return [{ id: 'recipe-pickaxe' }];
    };
    let craftedRecipe = null;
    bot.craft = async (recipe: any, count: any, table: any) => {
      craftedRecipe = recipe;
    };

    const result = await executeAction(bot, { kind: 'craft_wooden_pickaxe' });
    assert.equal(result.crafted, true);
    assert.equal(tableFoundId, registry.blocksByName.crafting_table.id);
    assert.ok(recipesQueryTable);
    assert.deepEqual(craftedRecipe, { id: 'recipe-pickaxe' });
  });

  it('equip_armor equips highest tier armor found in inventory', async () => {
    const bot = stubBot();
    bot.inventory = {
      items: () => [{ name: 'iron_chestplate' }, { name: 'leather_chestplate' }, { name: 'diamond_boots' }]
    };
    const equips: any[] = [];
    bot.equip = async (item: any, dest: any) => {
      equips.push({ itemName: item.name, dest });
    };

    const result = await executeAction(bot, { kind: 'equip_armor' });
    assert.equal(result.equipped, true);
    assert.equal(equips.length, 2);
    // Should prioritize iron over leather
    assert.deepEqual(equips.find(e => e.dest === 'torso'), { itemName: 'iron_chestplate', dest: 'torso' });
    assert.deepEqual(equips.find(e => e.dest === 'feet'), { itemName: 'diamond_boots', dest: 'feet' });
  });

  it('drop_item drops held item using tossStack', async () => {
    const bot = stubBot();
    bot.heldItem = { name: 'dirt', count: 64 };
    let tossedItem = null;
    bot.tossStack = async (item: any) => {
      tossedItem = item;
    };

    const result = await executeAction(bot, { kind: 'drop_item' });
    assert.equal(result.dropped, true);
    assert.deepEqual(tossedItem, { name: 'dirt', count: 64 });
  });

  it('drop_item returns false if hand is empty', async () => {
    const bot = stubBot();
    bot.heldItem = null;
    bot.tossStack = async () => { throw new Error('should not toss'); };

    const result = await executeAction(bot, { kind: 'drop_item' });
    assert.equal(result.dropped, false);
  });
});