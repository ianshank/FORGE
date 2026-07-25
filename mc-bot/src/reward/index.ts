// Pluggable reward registry. No auto-registration via side-effects —
// all builtins are imported as named factories below.

import * as survival from './builtins/survival.js';
import * as inventoryAcquired from './builtins/inventory_acquired.js';
import * as distanceToGoal from './builtins/distance_to_goal.js';
import * as healthDelta from './builtins/health_delta.js';
import * as composite from './builtins/composite.js';
import * as milestone from './builtins/milestone.js';

const _builtins = [survival, inventoryAcquired, distanceToGoal, healthDelta, composite, milestone];

export interface RewardContext {
  prev?: any;
  curr?: any;
  breakdown?: Record<string, number>;
}

export type RewardFn = (ctx: RewardContext) => number;

/**
 * Build a single RewardFn from a `{ kind, ...params }` config.
 */
export function buildOne(cfg: any): RewardFn {
  if (!cfg || typeof cfg !== 'object') {
    throw new Error('reward config must be an object');
  }
  const builtin = _builtins.find((b) => b.name === cfg.kind);
  if (!builtin) {
    throw new Error(`unknown reward kind: ${cfg.kind}`);
  }
  // `composite` needs `buildOne` injected so it can recurse without
  // creating an ES-module import cycle.
  return builtin.factory(cfg, { buildOne });
}

export function buildReward(cfg: any): RewardFn {
  if (!cfg || !Array.isArray(cfg.reward) || cfg.reward.length === 0) {
    throw new Error('rewards.toml must define at least one [[reward]]');
  }
  const fns = cfg.reward.map((rewardCfg: any) => {
    const fn = buildOne(rewardCfg);
    return { kind: rewardCfg.kind, fn };
  });

  return (ctx: RewardContext) => {
    let sum = 0;
    for (const { kind, fn } of fns) {
      const val = fn(ctx);
      sum += val;
      if (ctx?.breakdown) {
        ctx.breakdown[kind] = (ctx.breakdown[kind] || 0) + val;
      }
    }
    return sum;
  };
}

/** Names of all registered built-ins (for diagnostics + schema_id). */
export function listBuiltins(): string[] {
  return _builtins.map((b) => b.name);
}
