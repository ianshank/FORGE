// Episode reset strategy. v1 = teleport + state restore (no world regen).
// Mirrors the design documented in
// docs/plans/minecraft_rl_integration_plan_v2.md §3.3.2.

/**
 * @typedef {Object} ResetConfig
 * @property {'teleport'|'arena'} strategy
 * @property {Object} teleport
 * @property {{x:number,y:number,z:number}} teleport.spawn
 * @property {number} [teleport.yaw]
 * @property {number} [teleport.pitch]
 * @property {boolean} teleport.clear_inventory
 * @property {boolean} teleport.restore_health
 * @property {boolean} teleport.restore_food
 */

/**
 * Reset the bot per the supplied config. Pure side-effects on the bot;
 * returns a Promise that resolves when state-restoration commands have
 * been issued (NOT when ack'd by the server — that's a follow-up
 * tightening once mineflayer is wired up end-to-end).
 *
 * The function takes a bot-like interface so it can be unit-tested
 * with a stub.
 *
 * @param {object} bot   mineflayer bot or stub exposing `chat(string)`
 *                       and `entity.position`
 * @param {ResetConfig} cfg
 */
export async function applyReset(bot, cfg) {
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
  const spawn = t.spawn ?? { x: 0, y: 64, z: 0 };
  const yaw = Number.isFinite(t.yaw) ? t.yaw : 0;
  const pitch = Number.isFinite(t.pitch) ? t.pitch : 0;

  // Mineflayer uses chat commands for ops actions; the bot must be op
  // on the server. Each command is awaited only synchronously in this
  // stub — production wiring should listen for ack events.
  bot.chat(`/tp @s ${spawn.x} ${spawn.y} ${spawn.z} ${yaw} ${pitch}`);
  if (t.clear_inventory !== false) {
    bot.chat('/clear @s');
  }
  if (t.restore_health !== false) {
    bot.chat('/effect give @s minecraft:instant_health 1 10');
  }
  if (t.restore_food !== false) {
    bot.chat('/effect give @s minecraft:saturation 1 10');
  }
}
