import { DEFAULT_TICKS, waitTicks, setHotbarSlot } from './utils.js';
import type { ActionEntry } from '../action_map.js';

const EDIBLE_FOODS = new Set([
  'apple', 'baked_potato', 'beef', 'beetroot', 'bread', 'carrot', 'chicken',
  'cooked_beef', 'cooked_chicken', 'cooked_cod', 'cooked_mutton', 'cooked_porkchop',
  'cooked_rabbit', 'cooked_salmon', 'cookie', 'dried_kelp', 'golden_apple',
  'golden_carrot', 'melon_slice', 'mutton', 'poisonous_potato', 'porkchop',
  'potato', 'pumpkin_pie', 'rabbit', 'rotten_flesh', 'salmon', 'spider_eye',
  'sweet_berries', 'tropical_fish'
]);

export const inventoryHandlers: Record<string, (bot: any, action: ActionEntry, actionOptions: { defaultTicks: number; tickMs: number }) => Promise<any>> = {
  use: async (bot, action, actionOptions) => {
    if (typeof bot.activateItem !== 'function') {
      throw new Error('bot must expose activateItem() for use actions');
    }
    bot.activateItem();
    await waitTicks(bot, DEFAULT_TICKS, actionOptions);
    return { ticks: DEFAULT_TICKS };
  },

  place: async (bot, action, actionOptions) => {
    setHotbarSlot(bot, action.hotbar_slot ?? 0);
    if (typeof bot.activateItem !== 'function') {
      throw new Error('bot must expose activateItem() for place actions');
    }
    bot.activateItem();
    await waitTicks(bot, DEFAULT_TICKS, actionOptions);
    return { ticks: DEFAULT_TICKS };
  },

  select_slot: async (bot, action, actionOptions) => {
    setHotbarSlot(bot, action.hotbar_slot ?? 0);
    await waitTicks(bot, DEFAULT_TICKS, actionOptions);
    return { ticks: DEFAULT_TICKS };
  },

  eat: async (bot, action, actionOptions) => {
    if (typeof bot.inventory?.items !== 'function') {
      throw new Error('bot must expose inventory.items()');
    }
    const foodItem = bot.inventory.items().find((item: any) => EDIBLE_FOODS.has(item.name));
    if (!foodItem) {
      return { ticks: DEFAULT_TICKS, consumed: false };
    }
    if (typeof bot.equip !== 'function' || typeof bot.consume !== 'function') {
      throw new Error('bot must expose equip() and consume() for eat actions');
    }
    try {
      await bot.equip(foodItem, 'hand');
      await bot.consume();
      return { ticks: DEFAULT_TICKS, consumed: true };
    } catch (err: any) {
      return { ticks: DEFAULT_TICKS, consumed: false, error: err.message };
    }
  },

  equip_pickaxe: async (bot, action, actionOptions) => {
    if (typeof bot.inventory?.items !== 'function' || typeof bot.equip !== 'function') {
      throw new Error('bot must expose inventory.items() and equip()');
    }
    const items = bot.inventory.items();
    const pickaxe = items.find((i: any) => i.name === 'iron_pickaxe') 
                 || items.find((i: any) => i.name === 'stone_pickaxe') 
                 || items.find((i: any) => i.name === 'wooden_pickaxe');
    if (!pickaxe) return { ticks: DEFAULT_TICKS, equipped: false };
    try {
      await bot.equip(pickaxe, 'hand');
      return { ticks: DEFAULT_TICKS, equipped: true };
    } catch (err: any) {
      return { ticks: DEFAULT_TICKS, equipped: false, error: err.message };
    }
  },

  equip_armor: async (bot, action, actionOptions) => {
    if (typeof bot.inventory?.items !== 'function' || typeof bot.equip !== 'function') {
      throw new Error('bot must expose inventory.items() and equip()');
    }
    const items = bot.inventory.items();
    const armors: Record<string, string[]> = {
      head: ['netherite_helmet', 'diamond_helmet', 'iron_helmet', 'golden_helmet', 'chainmail_helmet', 'leather_helmet'],
      torso: ['netherite_chestplate', 'diamond_chestplate', 'iron_chestplate', 'golden_chestplate', 'chainmail_chestplate', 'leather_chestplate'],
      legs: ['netherite_leggings', 'diamond_leggings', 'iron_leggings', 'golden_leggings', 'chainmail_leggings', 'leather_leggings'],
      feet: ['netherite_boots', 'diamond_boots', 'iron_boots', 'golden_boots', 'chainmail_boots', 'leather_boots']
    };
    
    let equippedSomething = false;
    for (const dest of Object.keys(armors)) {
      const types = armors[dest];
      for (const type of types) {
        const item = items.find((i: any) => i.name === type);
        if (item) {
          try {
            await bot.equip(item, dest);
            equippedSomething = true;
          } catch (err) {}
          break;
        }
      }
    }
    return { ticks: DEFAULT_TICKS, equipped: equippedSomething };
  },

  drop_item: async (bot, action, actionOptions) => {
    if (typeof bot.tossStack !== 'function') {
      throw new Error('bot must expose tossStack()');
    }
    const handItem = bot.heldItem;
    if (handItem) {
      try {
        await bot.tossStack(handItem);
        return { ticks: DEFAULT_TICKS, dropped: true };
      } catch (err: any) {
        return { ticks: DEFAULT_TICKS, dropped: false, error: err.message };
      }
    }
    return { ticks: DEFAULT_TICKS, dropped: false };
  }
};
