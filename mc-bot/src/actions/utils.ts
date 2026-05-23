import { setTimeout as delay } from 'node:timers/promises';

export const DEFAULT_TICKS = 1;
export const DEFAULT_TICK_MS = 50;

export function positiveTicks(value: any, fallback: number = DEFAULT_TICKS): number {
  return Number.isInteger(value) && value > 0 ? value : fallback;
}

export async function waitTicks(
  bot: any,
  ticks: any,
  actionOptions: { defaultTicks: number; tickMs: number }
): Promise<void> {
  const count = positiveTicks(ticks);
  if (typeof bot.waitForTicks === 'function') {
    await bot.waitForTicks(count);
    return;
  }
  const tickMs = Number.isFinite(actionOptions.tickMs) && actionOptions.tickMs > 0
    ? actionOptions.tickMs
    : DEFAULT_TICK_MS;
  await delay(count * tickMs);
}

export function setControl(bot: any, control: string, state: boolean): void {
  if (typeof bot.setControlState !== 'function') {
    throw new Error('bot must expose setControlState(control, state)');
  }
  bot.setControlState(control, state);
}

export function setHotbarSlot(bot: any, hotbarSlot: number): void {
  if (!Number.isInteger(hotbarSlot) || hotbarSlot < 0 || hotbarSlot > 8) {
    throw new Error(`hotbar_slot must be 0..=8, got ${hotbarSlot}`);
  }
  if (typeof bot.setQuickBarSlot === 'function') {
    bot.setQuickBarSlot(hotbarSlot);
  } else {
    bot.quickBarSlot = hotbarSlot;
  }
}

export function radians(degrees: number): number {
  return (degrees * Math.PI) / 180;
}
