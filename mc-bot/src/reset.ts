// Episode reset strategy. v1 = teleport + state restore (no world regen).
// Mirrors the design documented in
// docs/plans/minecraft_rl_integration_plan_v2.md §3.3.2.

import type { ResetConfig } from './config.js';

const DEFAULT_SELECTOR = '@s';
const DEFAULT_SPAWN = Object.freeze({ x: 0, y: 64, z: 0 });

/**
 * Reset the bot per the supplied config. Pure side-effects on the bot;
 * returns a Promise that resolves when state-restoration commands have
 * been issued (NOT when ack'd by the server — that's a follow-up
 * tightening once mineflayer is wired up end-to-end).
 *
 * The function takes a bot-like interface so it can be unit-tested
 * with a stub.
 */
export async function applyReset(
  bot: { chat(msg: string): void; username?: string; [key: string]: any },
  cfg: ResetConfig
): Promise<void> {
  if (!bot || typeof bot.chat !== 'function') {
    throw new Error('bot must expose chat(string)');
  }
  if (!cfg || typeof cfg !== 'object') {
    throw new Error('reset config required');
  }
  if (cfg.strategy === 'arena') {
    // Arena mode is deferred to v3 (plan §7 out-of-scope).
    throw new Error('arena reset strategy not implemented in protocol v1');
  }
  if (cfg.strategy && cfg.strategy !== 'teleport') {
    throw new Error(`unknown reset strategy: ${cfg.strategy}`);
  }

  const t = cfg.teleport ?? {};
  const spawn = t.spawn ?? DEFAULT_SPAWN;
  const yaw = Number.isFinite(t.yaw) ? t.yaw : 0;
  const pitch = Number.isFinite(t.pitch) ? t.pitch : 0;
  const selector = (typeof t.selector === 'string' && t.selector.length > 0)
    ? t.selector
    : DEFAULT_SELECTOR;

  // Mineflayer uses chat commands for ops actions; the bot (or whoever
  // `selector` resolves to) must be op'd on the server.
  bot.chat(`/tp ${selector} ${spawn.x} ${spawn.y} ${spawn.z} ${yaw} ${pitch}`);
  if (t.clear_inventory !== false) {
    bot.chat(`/clear ${selector}`);
  }
  if (t.restore_health !== false) {
    bot.chat(`/effect give ${selector} minecraft:instant_health 1 10`);
  }
  if (t.restore_food !== false) {
    bot.chat(`/effect give ${selector} minecraft:saturation 1 10`);
  }
}
