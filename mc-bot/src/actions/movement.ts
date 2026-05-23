import { DEFAULT_TICKS, positiveTicks, waitTicks, setControl, radians } from './utils.js';
import type { ActionEntry } from '../action_map.js';

const MOVE_CONTROLS: Record<string, string> = Object.freeze({
  forward: 'forward',
  back: 'back',
  left: 'left',
  right: 'right',
});

export const movementHandlers: Record<string, (bot: any, action: ActionEntry, actionOptions: { defaultTicks: number; tickMs: number }) => Promise<{ ticks: number }>> = {
  noop: async (bot, action, actionOptions) => {
    const ticks = positiveTicks(action.ticks, actionOptions.defaultTicks);
    await waitTicks(bot, ticks, actionOptions);
    return { ticks };
  },

  move: async (bot, action, actionOptions) => {
    const control = MOVE_CONTROLS[action.direction ?? ''];
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
  },

  jump: async (bot, action, actionOptions) => {
    setControl(bot, 'jump', true);
    try {
      await waitTicks(bot, DEFAULT_TICKS, actionOptions);
    } finally {
      setControl(bot, 'jump', false);
    }
    return { ticks: DEFAULT_TICKS };
  },

  look: async (bot, action, actionOptions) => {
    if (typeof bot.look !== 'function') {
      throw new Error('bot must expose look(yaw, pitch, force)');
    }
    const yaw = Number(bot.entity?.yaw ?? 0) + radians(action.yaw_deg ?? 0);
    const pitch = Number(bot.entity?.pitch ?? 0) + radians(action.pitch_deg ?? 0);
    await bot.look(yaw, pitch, true);
    await waitTicks(bot, DEFAULT_TICKS, actionOptions);
    return { ticks: DEFAULT_TICKS };
  },

  sprint: async (bot, action, actionOptions) => {
    const ticks = positiveTicks(action.ticks, actionOptions.defaultTicks);
    setControl(bot, 'sprint', true);
    try {
      await waitTicks(bot, ticks, actionOptions);
    } finally {
      setControl(bot, 'sprint', false);
    }
    return { ticks };
  },

  sneak: async (bot, action, actionOptions) => {
    const ticks = positiveTicks(action.ticks, actionOptions.defaultTicks);
    setControl(bot, 'sneak', true);
    try {
      await waitTicks(bot, ticks, actionOptions);
    } finally {
      setControl(bot, 'sneak', false);
    }
    return { ticks };
  },

  swim_up: async (bot, action, actionOptions) => {
    const ticks = positiveTicks(action.ticks, actionOptions.defaultTicks);
    setControl(bot, 'jump', true);
    try {
      await waitTicks(bot, ticks, actionOptions);
    } finally {
      setControl(bot, 'jump', false);
    }
    return { ticks };
  }
};
