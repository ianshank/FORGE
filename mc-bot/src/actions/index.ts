import { DEFAULT_TICK_MS, positiveTicks } from './utils.js';
import { movementHandlers } from './movement.js';
import { combatHandlers } from './combat.js';
import { inventoryHandlers } from './inventory.js';
import { macroHandlers } from './macros.js';
import type { ActionEntry } from '../action_map.js';

export interface ActionOptions {
  defaultTicks?: number;
  tickMs?: number;
}

export type ActionHandler = (
  bot: any,
  action: ActionEntry,
  options: { defaultTicks: number; tickMs: number }
) => Promise<any>;

const actionRegistry: Record<string, ActionHandler> = {
  ...movementHandlers,
  ...combatHandlers,
  ...inventoryHandlers,
  ...macroHandlers,
};

export async function executeAction(
  bot: any,
  action: ActionEntry,
  options: ActionOptions = {}
): Promise<any> {
  if (!bot || typeof bot !== 'object') {
    throw new Error('bot is required');
  }
  if (!action || typeof action !== 'object') {
    throw new Error('action is required');
  }
  const actionOptions = {
    defaultTicks: positiveTicks(options.defaultTicks),
    tickMs: options.tickMs ?? DEFAULT_TICK_MS,
  };

  if (action.kind !== 'attack' && bot.pvp && bot.pvp.target) {
    bot.pvp.stop();
  }

  const handler = actionRegistry[action.kind];
  if (!handler) {
    throw new Error(`unknown action kind: ${action.kind}`);
  }

  return handler(bot, action, actionOptions);
}
