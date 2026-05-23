// +R the first time each configured item appears in inventory.

import type { RewardContext } from '../index.js';

export const name = 'inventory_acquired';

export function factory(params: any): (ctx: RewardContext) => number {
  const items = Array.isArray(params.items) ? params.items : [];
  const valuePerItem = Number.isFinite(params.value) ? params.value : 1.0;
  const seen = new Set<string>();
  return ({ curr }) => {
    if (!curr || !curr.inventory) return 0;
    let bonus = 0;
    for (const item of items) {
      if (seen.has(item)) continue;
      const count = curr.inventory[item] ?? 0;
      if (count > 0) {
        seen.add(item);
        bonus += valuePerItem;
      }
    }
    return bonus;
  };
}
