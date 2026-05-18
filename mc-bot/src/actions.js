import { setTimeout as delay } from 'node:timers/promises';

const DEFAULT_TICKS = 1;
const DEFAULT_TICK_MS = 50;
const MOVE_CONTROLS = Object.freeze({
  forward: 'forward',
  back: 'back',
  left: 'left',
  right: 'right',
});

function positiveTicks(value, fallback = DEFAULT_TICKS) {
  return Number.isInteger(value) && value > 0 ? value : fallback;
}

async function waitTicks(bot, ticks, actionOptions) {
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

function setControl(bot, control, state) {
  if (typeof bot.setControlState !== 'function') {
    throw new Error('bot must expose setControlState(control, state)');
  }
  bot.setControlState(control, state);
}

function setHotbarSlot(bot, hotbarSlot) {
  if (!Number.isInteger(hotbarSlot) || hotbarSlot < 0 || hotbarSlot > 8) {
    throw new Error(`hotbar_slot must be 0..=8, got ${hotbarSlot}`);
  }
  if (typeof bot.setQuickBarSlot === 'function') {
    bot.setQuickBarSlot(hotbarSlot);
  } else {
    bot.quickBarSlot = hotbarSlot;
  }
}

function radians(degrees) {
  return (degrees * Math.PI) / 180;
}

async function executeMove(bot, action, actionOptions) {
  const control = MOVE_CONTROLS[action.direction];
  if (!control) {
    throw new Error(`unknown move direction: ${action.direction}`);
  }
  const ticks = positiveTicks(action.ticks, actionOptions.defaultTicks);
  setControl(bot, control, true);
  try {
    await waitTicks(bot, ticks, actionOptions);
  } finally {
    setControl(bot, control, false);
  }
  return { ticks };
}

export async function executeAction(bot, action, options = {}) {
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

  switch (action.kind) {
    case 'noop': {
      const ticks = positiveTicks(action.ticks, actionOptions.defaultTicks);
      await waitTicks(bot, ticks, actionOptions);
      return { ticks };
    }
    case 'move':
      return executeMove(bot, action, actionOptions);
    case 'jump':
      setControl(bot, 'jump', true);
      try {
        await waitTicks(bot, DEFAULT_TICKS, actionOptions);
      } finally {
        setControl(bot, 'jump', false);
      }
      return { ticks: DEFAULT_TICKS };
    case 'attack': {
      const target = typeof bot.nearestEntity === 'function' ? bot.nearestEntity() : null;
      if (target && typeof bot.attack === 'function') {
        bot.attack(target);
      } else if (typeof bot.swingArm === 'function') {
        bot.swingArm('right');
      }
      await waitTicks(bot, DEFAULT_TICKS, actionOptions);
      return { ticks: DEFAULT_TICKS };
    }
    case 'use':
      if (typeof bot.activateItem !== 'function') {
        throw new Error('bot must expose activateItem() for use actions');
      }
      bot.activateItem();
      await waitTicks(bot, DEFAULT_TICKS, actionOptions);
      return { ticks: DEFAULT_TICKS };
    case 'place':
      setHotbarSlot(bot, action.hotbar_slot);
      if (typeof bot.activateItem !== 'function') {
        throw new Error('bot must expose activateItem() for place actions');
      }
      bot.activateItem();
      await waitTicks(bot, DEFAULT_TICKS, actionOptions);
      return { ticks: DEFAULT_TICKS };
    case 'select_slot':
      setHotbarSlot(bot, action.hotbar_slot);
      await waitTicks(bot, DEFAULT_TICKS, actionOptions);
      return { ticks: DEFAULT_TICKS };
    case 'look': {
      if (typeof bot.look !== 'function') {
        throw new Error('bot must expose look(yaw, pitch, force)');
      }
      const yaw = Number(bot.entity?.yaw ?? 0) + radians(action.yaw_deg);
      const pitch = Number(bot.entity?.pitch ?? 0) + radians(action.pitch_deg);
      await bot.look(yaw, pitch, true);
      await waitTicks(bot, DEFAULT_TICKS, actionOptions);
      return { ticks: DEFAULT_TICKS };
    }
    default:
      throw new Error(`unknown action kind: ${action.kind}`);
  }
}