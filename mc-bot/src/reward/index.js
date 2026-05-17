// Pluggable reward registry. No auto-registration via side-effects —
// all builtins are imported as named factories below.

import * as survival from './builtins/survival.js';
import * as inventoryAcquired from './builtins/inventory_acquired.js';
import * as distanceToGoal from './builtins/distance_to_goal.js';
import * as healthDelta from './builtins/health_delta.js';
import * as composite from './builtins/composite.js';

const _builtins = [survival, inventoryAcquired, distanceToGoal, healthDelta, composite];

/**
 * Build a single RewardFn from a `{ kind, ...params }` config.
 */
export function buildOne(cfg) {
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

/**
 * Build the active reward function from a parsed rewards.toml object.
 */
export function buildReward(cfg) {
  if (!cfg || !Array.isArray(cfg.reward) || cfg.reward.length === 0) {
    throw new Error('rewards.toml must define at least one [[reward]]');
  }
  const fns = cfg.reward.map(buildOne);
  if (fns.length === 1) return fns[0];
  return (ctx) => fns.reduce((acc, fn) => acc + fn(ctx), 0);
}

/** Names of all registered built-ins (for diagnostics + schema_id). */
export function listBuiltins() {
  return _builtins.map((b) => b.name);
}
