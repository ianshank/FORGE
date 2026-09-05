// Episode reset strategy. v1 = teleport + state restore (no world regen).
// Mirrors the design documented in
// docs/plans/minecraft_rl_integration_plan_v2.md §3.3.2.

import type { ResetConfig } from './config.js';

const DEFAULT_SELECTOR = '@s';
const DEFAULT_SPAWN = Object.freeze({ x: 0, y: 64, z: 0 });

/**
 * Strict allowlist for the reset selector.
 *
 * `applyReset` interpolates the selector into op-level chat commands
 * (`/tp`, `/clear`, `/effect give`). Minecraft's chat transport treats a
 * newline as a command boundary, so an unvalidated selector such as
 * `"@s\n/op attacker"` smuggles a second privileged command onto the server.
 * Only these three shapes are accepted:
 *
 * 1. A base target selector — `@p`, `@r`, `@a`, `@e`, `@s` — optionally
 *    followed by a bracketed argument list restricted to
 *    `A-Z a-z 0-9 _ . : = , ! -` (no whitespace, quotes, or `/`).
 * 2. A vanilla player name: 1–16 characters of `[A-Za-z0-9_]`.
 * 3. A player UUID in canonical 8-4-4-4-12 hex form.
 *
 * JavaScript's `$` (without the `m` flag) anchors at end-of-input rather than
 * before a trailing newline, so `"@s\n..."` cannot match.
 */
export const SELECTOR_PATTERN =
  /^(?:@[parse](?:\[[A-Za-z0-9_.:=,!-]{1,256}\])?|[A-Za-z0-9_]{1,16}|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})$/;

/**
 * Validate a configured reset selector against {@link SELECTOR_PATTERN}.
 *
 * @throws Error when the selector is not an allowlisted target selector,
 *   player name, or UUID.
 */
export function validateSelector(selector: string): string {
  if (typeof selector !== 'string' || !SELECTOR_PATTERN.test(selector)) {
    throw new Error(
      `reset.teleport.selector must be a Minecraft target selector (@p/@r/@a/@e/@s, optionally with a [key=value] argument list), a 1-16 character player name, or a player UUID; got ${JSON.stringify(selector)}`,
    );
  }
  return selector;
}

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
  const rawSpawn = t.spawn ?? DEFAULT_SPAWN;
  // Coordinates reach the same chat sink as `selector`, and Minecraft's chat
  // transport treats a newline as a command boundary. `yaw`/`pitch` were
  // already guarded with Number.isFinite; `spawn.x/y/z` were not, so a
  // reset.toml with `x = "0\n/op attacker"` smuggled a privileged command
  // through the exact mechanism validateSelector() closes. configs/minecraft
  // is a bind-mount in the compose stack, so that file is not trusted input.
  const spawn = {
    x: Number.isFinite(rawSpawn?.x) ? rawSpawn.x : DEFAULT_SPAWN.x,
    y: Number.isFinite(rawSpawn?.y) ? rawSpawn.y : DEFAULT_SPAWN.y,
    z: Number.isFinite(rawSpawn?.z) ? rawSpawn.z : DEFAULT_SPAWN.z,
  };
  const yaw = Number.isFinite(t.yaw) ? t.yaw : 0;
  const pitch = Number.isFinite(t.pitch) ? t.pitch : 0;
  // An absent / non-string / empty selector falls back to the default; anything
  // else must survive the allowlist before it reaches a chat command.
  const selector = (typeof t.selector === 'string' && t.selector.length > 0)
    ? validateSelector(t.selector)
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
