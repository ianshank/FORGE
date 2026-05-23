// One-shot milestone achievement reward shaper.
// Loads achievement-based reward values from milestone_rewards.toml
// and grants one-shot rewards upon triggering milestones.

import { readFileSync } from 'node:fs';
import type { RewardContext } from '../index.js';

export const name = 'milestone';

interface MilestoneSpec {
  reward: number;
  once: boolean;
}

export function factory(params: any): (ctx: RewardContext) => number {
  // Default milestones in case the TOML fails to parse or is missing
  const milestoneConfig: Record<string, MilestoneSpec> = {
    first_wood: { reward: 10.0, once: true },
    first_stone_tool: { reward: 25.0, once: true },
    first_iron: { reward: 50.0, once: true },
    first_shelter: { reward: 100.0, once: true },
    first_food_cooked: { reward: 15.0, once: true },
  };

  const configPath = params.config_path || 'configs/minecraft/milestone_rewards.toml';
  const craftingConfigPath = params.crafting_config_path || 'configs/minecraft/crafting_rewards.toml';
  
  const parseTomlConfig = (filepath: string): Record<string, MilestoneSpec> => {
    const parsed: Record<string, MilestoneSpec> = {};
    try {
      const content = readFileSync(filepath, 'utf8');
      for (const line of content.split('\n')) {
        const trimmed = line.trim();
        if (trimmed.startsWith('#') || !trimmed.includes('=')) continue;
        const [key, value] = trimmed.split('=').map((s) => s.trim());
        if (value.startsWith('{') && value.endsWith('}')) {
          const inner = value.slice(1, -1);
          const rewardMatch = inner.match(/reward\s*=\s*([0-9.-]+)/);
          const onceMatch = inner.match(/once\s*=\s*(true|false)/);
          if (rewardMatch) {
            parsed[key] = {
              reward: parseFloat(rewardMatch[1]),
              once: onceMatch ? onceMatch[1] === 'true' : true,
            };
          }
        }
      }
    } catch (_error) {
      // Keep defaults
    }
    return parsed;
  };

  const parsedMilestones = parseTomlConfig(configPath);
  const parsedCrafting = parseTomlConfig(craftingConfigPath);
  
  if (Object.keys(parsedMilestones).length > 0) {
    Object.assign(milestoneConfig, parsedMilestones);
  }
  if (Object.keys(parsedCrafting).length > 0) {
    Object.assign(milestoneConfig, parsedCrafting);
  }

  // Keep track of achieved milestones per episode
  const achieved = new Set<string>();

  const woodKeywords = ['log', 'planks', 'wood'];
  const stoneTools = ['stone_pickaxe', 'stone_axe', 'stone_shovel', 'stone_sword', 'stone_hoe'];
  const ironItems = ['iron_ore', 'raw_iron', 'iron_ingot', 'iron_block'];
  const shelterItems = ['bed', 'chest', 'furnace'];
  const shelterBlocks = ['chest', 'furnace', 'bed', 'torch', 'crafting_table'];
  const cookedFoods = [
    'cooked_porkchop',
    'cooked_beef',
    'cooked_chicken',
    'cooked_mutton',
    'cooked_rabbit',
    'cooked_cod',
    'cooked_salmon',
    'bread',
    'cookie',
    'pumpkin_pie',
  ];

  return ({ prev, curr }) => {
    if (!curr) return 0;

    // Reset achieved set on episode reset
    if (curr.tick === 0 || (prev && curr.tick < prev.tick)) {
      achieved.clear();
    }

    const currentInventory = curr.inventory ?? {};
    let totalReward = 0;

    // 1. first_wood
    if (milestoneConfig.first_wood && !achieved.has('first_wood')) {
      const hasWood = Object.entries(currentInventory).some(
        ([key, val]) => woodKeywords.some((keyword) => key.includes(keyword)) && (val as number) > 0,
      );
      if (hasWood) {
        achieved.add('first_wood');
        totalReward += milestoneConfig.first_wood.reward;
      }
    }

    // 2. first_stone_tool
    if (milestoneConfig.first_stone_tool && !achieved.has('first_stone_tool')) {
      const hasStoneTool = stoneTools.some((tool) => (currentInventory[tool] ?? 0) > 0);
      if (hasStoneTool) {
        achieved.add('first_stone_tool');
        totalReward += milestoneConfig.first_stone_tool.reward;
      }
    }

    // 3. first_iron
    if (milestoneConfig.first_iron && !achieved.has('first_iron')) {
      const hasIron = ironItems.some((item) => (currentInventory[item] ?? 0) > 0);
      if (hasIron) {
        achieved.add('first_iron');
        totalReward += milestoneConfig.first_iron.reward;
      }
    }

    // 4. first_shelter
    if (milestoneConfig.first_shelter && !achieved.has('first_shelter')) {
      const hasShelterItem = shelterItems.some((item) => (currentInventory[item] ?? 0) > 0);
      const hasShelterBlock = (curr.gridTopBlockTypes as Array<[string, number]>)?.some(([name]) =>
        shelterBlocks.some((block) => name.includes(block)),
      );
      if (hasShelterItem || hasShelterBlock) {
        achieved.add('first_shelter');
        totalReward += milestoneConfig.first_shelter.reward;
      }
    }

    // 5. first_food_cooked
    if (milestoneConfig.first_food_cooked && !achieved.has('first_food_cooked')) {
      const hasCookedFood = cookedFoods.some((food) => (currentInventory[food] ?? 0) > 0);
      if (hasCookedFood) {
        achieved.add('first_food_cooked');
        totalReward += milestoneConfig.first_food_cooked.reward;
      }
    }

    // 6. Crafting Milestones
    const checkCraftingMilestone = (key: string, itemMatches: string[]) => {
      if (milestoneConfig[key] && !achieved.has(key)) {
        const hasItem = itemMatches.some((item) => (currentInventory[item] ?? 0) > 0);
        const hasBlock = (curr.gridTopBlockTypes as Array<[string, number]>)?.some(([name]) => itemMatches.some((b) => name.includes(b)));
        if (hasItem || hasBlock) {
          achieved.add(key);
          totalReward += milestoneConfig[key].reward;
        }
      }
    };

    checkCraftingMilestone('first_crafting_table', ['crafting_table']);
    checkCraftingMilestone('first_wooden_pickaxe', ['wooden_pickaxe']);
    checkCraftingMilestone('first_stone_pickaxe', ['stone_pickaxe']);
    checkCraftingMilestone('first_furnace', ['furnace']);
    checkCraftingMilestone('first_iron_pickaxe', ['iron_pickaxe']);

    return totalReward;
  };
}
