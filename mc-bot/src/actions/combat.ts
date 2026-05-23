import { DEFAULT_TICKS, waitTicks } from './utils.js';
import type { ActionEntry } from '../action_map.js';

export const combatHandlers: Record<string, (bot: any, action: ActionEntry, actionOptions: { defaultTicks: number; tickMs: number }) => Promise<{ ticks: number }>> = {
  attack: async (bot, action, actionOptions) => {
    const filter = (entity: any) => entity.type === 'mob' || entity.type === 'player' || entity.type === 'animal' || entity.type === 'hostile';
    const target = typeof bot.nearestEntity === 'function' ? bot.nearestEntity(filter) : null;
    if (target) {
      if (bot.pvp && typeof bot.pvp.attack === 'function') {
        // Tell mineflayer-pvp to engage the target. It manages weapons and pathfinding automatically.
        bot.pvp.attack(target);
      } else if (typeof bot.attack === 'function') {
        bot.attack(target);
      } else if (typeof bot.swingArm === 'function') {
        bot.swingArm('right');
      }
    } else if (typeof bot.swingArm === 'function') {
      bot.swingArm('right');
    }
    await waitTicks(bot, actionOptions.defaultTicks, actionOptions);
    return { ticks: actionOptions.defaultTicks };
  }
};
