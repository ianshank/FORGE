import { setTimeout as delay } from 'node:timers/promises';
import { DEFAULT_TICKS } from './utils.js';
import { CraftingPlanner } from '../crafting.js';
import type { ActionEntry } from '../action_map.js';

export const macroHandlers: Record<string, (bot: any, action: ActionEntry, actionOptions: { defaultTicks: number; tickMs: number }) => Promise<any>> = {
  craft_planks: async (bot, action, actionOptions) => {
    if (typeof bot.recipesFor !== 'function' || typeof bot.craft !== 'function') {
      throw new Error('bot must expose recipesFor() and craft() for crafting actions');
    }
    const registry = bot.registry;
    if (!registry) {
      throw new Error('bot must expose registry for item resolution');
    }
    const plankNames = Object.keys(registry.itemsByName).filter(name => name.endsWith('_planks'));
    for (const name of plankNames) {
      const targetItem = registry.itemsByName[name];
      if (!targetItem) continue;
      const planner = new CraftingPlanner(bot);
      const planIds = planner.plan(targetItem.id, 1);
      if (planIds) {
        try {
          for (const stepItemId of planIds) {
            const recipes = bot.recipesFor(stepItemId, null, 1, null);
            if (recipes.length === 0) throw new Error(`Cannot craft intermediate item ${stepItemId}`);
            await bot.craft(recipes[0], 1, null);
          }
          return { ticks: DEFAULT_TICKS, crafted: true, item: name };
        } catch (err) {
          continue;
        }
      }
    }
    return { ticks: DEFAULT_TICKS, crafted: false };
  },

  craft_sticks: craftItemHandler,
  craft_crafting_table: craftItemHandler,
  craft_wooden_pickaxe: craftItemHandler,
  craft_stone_pickaxe: craftItemHandler,
  craft_furnace: craftItemHandler,
  craft_iron_pickaxe: craftItemHandler,

  place_crafting_table: placeBlockHandler,
  place_furnace: placeBlockHandler,

  mine_stone: async (bot, action, actionOptions) => {
    if (typeof bot.findBlock !== 'function' || typeof bot.dig !== 'function') {
      throw new Error('bot must expose findBlock() and dig()');
    }
    const registry = bot.registry;
    const targetId = registry.blocksByName['stone']?.id;
    if (targetId === undefined) return { ticks: DEFAULT_TICKS, mined: false };
    const block = bot.findBlock({ matching: targetId, maxDistance: 4 });
    if (!block) return { ticks: DEFAULT_TICKS, mined: false };
    try {
      await bot.dig(block);
      return { ticks: DEFAULT_TICKS, mined: true };
    } catch (err: any) {
      return { ticks: DEFAULT_TICKS, mined: false, error: err.message };
    }
  },

  smelt_iron: async (bot, action, actionOptions) => {
    if (typeof bot.findBlock !== 'function' || typeof bot.openFurnace !== 'function') {
      throw new Error('bot must expose findBlock and openFurnace');
    }
    const registry = bot.registry;
    const furnaceId = registry.blocksByName['furnace']?.id;
    if (furnaceId === undefined) return { ticks: DEFAULT_TICKS, smelted: false };
    const furnaceBlock = bot.findBlock({ matching: furnaceId, maxDistance: 4 });
    if (!furnaceBlock) return { ticks: DEFAULT_TICKS, smelted: false };
    try {
      const furnace = await bot.openFurnace(furnaceBlock);
      
      // 1. If there's already output, take it immediately
      let output = typeof furnace.outputItem === 'function' ? furnace.outputItem() : null;
      if (output) {
        if (typeof furnace.takeOutput === 'function') {
          await furnace.takeOutput();
        }
        furnace.close();
        return { ticks: DEFAULT_TICKS, smelted: true };
      }
      
      // 2. Try to put raw iron and fuel in if the slots are empty
      const rawIron = bot.inventory.items().find((i: any) => i.name === 'raw_iron' || i.name === 'iron_ore');
      const fuel = bot.inventory.items().find((i: any) => i.name === 'coal' || i.name === 'charcoal' || i.name.endsWith('_planks'));
      
      let inputAdded = false;
      let fuelAdded = false;
      
      const currentInput = typeof furnace.inputItem === 'function' ? furnace.inputItem() : null;
      const currentFuel = typeof furnace.fuelItem === 'function' ? furnace.fuelItem() : null;
      
      if (rawIron && !currentInput) {
        await furnace.putInput(rawIron.type, null, 1);
        inputAdded = true;
      }
      if (fuel && !currentFuel) {
        await furnace.putFuel(fuel.type, null, 1);
        fuelAdded = true;
      }
      
      // 3. Short non-blocking poll of 100ms to see if smelting completes instantly or is active
      await delay(100);
      output = typeof furnace.outputItem === 'function' ? furnace.outputItem() : null;
      if (output) {
        if (typeof furnace.takeOutput === 'function') {
          await furnace.takeOutput();
        }
        furnace.close();
        return { ticks: DEFAULT_TICKS, smelted: true };
      }
      
      furnace.close();
      return { 
        ticks: DEFAULT_TICKS, 
        smelted: false, 
        status: (inputAdded || fuelAdded || currentInput) ? 'smelting_in_progress' : 'no_materials' 
      };
    } catch (err: any) {
      return { ticks: DEFAULT_TICKS, smelted: false, error: err.message };
    }
  }
};

async function craftItemHandler(bot: any, action: ActionEntry, actionOptions: any): Promise<any> {
  if (typeof bot.recipesFor !== 'function' || typeof bot.craft !== 'function') {
    throw new Error('bot must expose recipesFor() and craft() for crafting actions');
  }
  const registry = bot.registry;
  if (!registry) {
    throw new Error('bot must expose registry for item resolution');
  }
  const itemName = action.kind.replace('craft_', '');
  const actualName = itemName === 'sticks' ? 'stick' : itemName;
  const targetItem = registry.itemsByName[actualName];
  if (!targetItem) {
    return { ticks: DEFAULT_TICKS, crafted: false, error: `unknown item: ${actualName}` };
  }
  const planner = new CraftingPlanner(bot);
  const planIds = planner.plan(targetItem.id, 1);
  if (!planIds) {
    return { ticks: DEFAULT_TICKS, crafted: false };
  }
  try {
    for (const stepItemId of planIds) {
      // Determine if this specific step requires a crafting table
      let stepTableBlock = null;
      const stepItemName = registry.items[stepItemId]?.name || '';
      if (stepItemName.includes('pickaxe') || stepItemName === 'furnace') {
        const tableId = registry.blocksByName['crafting_table']?.id;
        if (tableId !== undefined && typeof bot.findBlock === 'function') {
          stepTableBlock = bot.findBlock({ matching: tableId, maxDistance: 4 });
        }
      }
      const recipes = bot.recipesFor(stepItemId, null, 1, stepTableBlock);
      if (recipes.length === 0) {
        throw new Error(`Missing ingredients or table for step: ${stepItemName}`);
      }
      await bot.craft(recipes[0], 1, stepTableBlock);
    }
    return { ticks: DEFAULT_TICKS, crafted: true };
  } catch (err: any) {
    return { ticks: DEFAULT_TICKS, crafted: false, error: err.message };
  }
}

async function placeBlockHandler(bot: any, action: ActionEntry, actionOptions: any): Promise<any> {
  if (typeof bot.inventory?.items !== 'function' || typeof bot.equip !== 'function' || typeof bot.placeBlock !== 'function') {
    throw new Error('bot must expose inventory, equip, and placeBlock');
  }
  const itemName = action.kind.replace('place_', '');
  const targetItem = bot.inventory.items().find((i: any) => i.name === itemName);
  if (!targetItem) return { ticks: DEFAULT_TICKS, placed: false };
  try {
    await bot.equip(targetItem, 'hand');
    let refBlock = null;
    if (typeof bot.blockAtCursor === 'function') {
      refBlock = bot.blockAtCursor(4);
    }
    if (!refBlock || refBlock.name === 'air') {
      const pos = bot.entity.position.floored();
      refBlock = bot.blockAt(pos.offset(1, -1, 0)) || bot.blockAt(pos.offset(0, -1, 1));
    }
    if (!refBlock || refBlock.name === 'air') return { ticks: DEFAULT_TICKS, placed: false };
    const faceVector = bot.entity.position.clone().set(0, 1, 0);
    await bot.placeBlock(refBlock, faceVector);
    return { ticks: DEFAULT_TICKS, placed: true };
  } catch (err: any) {
    return { ticks: DEFAULT_TICKS, placed: false, error: err.message };
  }
}
