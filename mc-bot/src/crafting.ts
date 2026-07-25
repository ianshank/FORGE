import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);

const recipeCache = new Map<string, any>();

export class CraftingPlanner {
  private bot: any;
  private Recipe: any;
  private registry: any;

  constructor(bot: any) {
    this.bot = bot;
    if (!recipeCache.has(bot.version)) {
      recipeCache.set(bot.version, require('prismarine-recipe')(bot.version));
    }
    const recipeModule = recipeCache.get(bot.version);
    this.Recipe = recipeModule.Recipe;
    this.registry = bot.registry;
  }

  // Gets the current items mapping id -> count
  getInventoryState(): Record<number, number> {
    const state: Record<number, number> = {};
    if (!this.bot.inventory) return state;
    for (const item of this.bot.inventory.items()) {
      state[item.type] = (state[item.type] || 0) + item.count;
    }
    return state;
  }

  // Basic DFS for crafting plan. Returns an array of Recipe objects to execute in order.
  // Returns null if it cannot be crafted with current inventory.
  plan(targetItemId: number, count = 1): number[] | null {
    const inventory = this.getInventoryState();
    const plan: number[] = [];

    // recursively find a way to get 'count' of 'itemId'
    const resolve = (itemId: number, neededCount: number): boolean => {
      // 1. Can we just satisfy it from inventory?
      const available = inventory[itemId] || 0;
      if (available >= neededCount) {
        inventory[itemId] -= neededCount;
        return true;
      }
      
      const stillNeeded = neededCount - available;
      // We will consume all available
      inventory[itemId] = 0;

      // 2. Try to craft the remainder
      const recipes = this.Recipe.find(itemId);
      if (recipes.length === 0) {
        return false; // Can't be crafted and not enough in inv
      }

      // Try recipes (usually the first one is fine for basics)
      for (const recipe of recipes) {
        // How many times do we need to execute this recipe?
        const yields = recipe.result.count;
        const executions = Math.ceil(stillNeeded / yields);
        
        // Save inventory state before trying this recipe
        const stateSnapshot = { ...inventory };
        const planSnapshot = [...plan];
        
        let possible = true;
        
        // Accumulate required ingredients
        // shape or shapeless
        const requirements: Record<number, number> = {};
        if (recipe.inShape) {
          for (const row of recipe.inShape) {
            for (const cell of row) {
              if (cell.id !== -1) {
                requirements[cell.id] = (requirements[cell.id] || 0) + 1;
              }
            }
          }
        } else if (recipe.ingredients) {
          for (const ing of recipe.ingredients) {
            if (ing.id !== -1) {
              requirements[ing.id] = (requirements[ing.id] || 0) + 1;
            }
          }
        }

        // Try to resolve each requirement
        for (const [reqIdStr, reqCount] of Object.entries(requirements)) {
          const reqId = Number.parseInt(reqIdStr, 10);
          if (!resolve(reqId, reqCount * executions)) {
            possible = false;
            break;
          }
        }

        if (possible) {
          // If we successfully resolved all requirements, record this step
          for (let i = 0; i < executions; i++) {
             plan.push(itemId);
          }
          // We got `executions * yields` items, we only needed `stillNeeded`.
          // The remainder goes into inventory!
          const remainder = (executions * yields) - stillNeeded;
          inventory[itemId] = (inventory[itemId] || 0) + remainder;
          return true;
        }
        // Revert state and try next recipe
        Object.assign(inventory, stateSnapshot);
        for (const keyStr of Object.keys(inventory)) {
          const key = Number.parseInt(keyStr, 10);
          if (!(key in stateSnapshot)) delete inventory[key];
        }
        plan.length = planSnapshot.length;
        for (let i = 0; i < planSnapshot.length; i++) plan[i] = planSnapshot[i];
      }
      return false; // All recipes failed
    };

    if (resolve(targetItemId, count)) {
      return plan;
    }
    return null;
  }
}
