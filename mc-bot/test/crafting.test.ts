import test from 'node:test';
import assert from 'node:assert';
import { CraftingPlanner } from '../src/crafting.js';
import { createRequire } from 'module';

const require = createRequire(import.meta.url);
const registry = require('prismarine-registry')('1.20.4');

test('CraftingPlanner resolves wooden pickaxe from logs', (t) => {
  // Mock bot with registry and inventory
  const bot = {
    version: '1.20.4',
    registry: registry,
    inventory: {
      items: () => [
        { type: registry.itemsByName['oak_log'].id, count: 3 }
      ]
    }
  };

  const planner = new CraftingPlanner(bot);
  const pickaxeId = registry.itemsByName['wooden_pickaxe'].id;
  
  const plan = planner.plan(pickaxeId, 1);
  assert.ok(plan !== null, 'Plan should not be null');
  
  // We expect: craft planks, craft sticks, craft pickaxe.
  // We have 3 logs. 
  // Pickaxe needs 3 planks and 2 sticks.
  // 1 log -> 4 planks. 1 plank -> ? wait, 2 planks -> 4 sticks.
  // So 1 log -> 4 planks. We have 4 planks.
  // Use 2 planks -> 4 sticks. We have 2 planks, 4 sticks.
  // Pickaxe needs 3 planks, 2 sticks. We only have 2 planks!
  // So we need another log -> 4 planks. Now we have 6 planks.
  // So the plan should be:
  // craft planks
  // craft planks
  // craft sticks
  // craft wooden_pickaxe

  const planNames = plan.map(id => registry.items[id].name);
  console.log("Plan steps:");
  for (const name of planNames) {
    console.log("- craft", name);
  }

  assert.strictEqual(planNames[planNames.length - 1], 'wooden_pickaxe');
});

test('CraftingPlanner fails if not enough logs', (t) => {
  const bot = {
    version: '1.20.4',
    registry: registry,
    inventory: {
      items: () => [
        { type: registry.itemsByName['oak_log'].id, count: 1 } // only 1 log = 4 planks -> 2 planks + 4 sticks -> pickaxe needs 3 planks, impossible.
      ]
    }
  };
  const planner = new CraftingPlanner(bot);
  const pickaxeId = registry.itemsByName['wooden_pickaxe'].id;
  const plan = planner.plan(pickaxeId, 1);
  assert.strictEqual(plan, null);
});
