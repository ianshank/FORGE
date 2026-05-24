// Weighted sum of named reward kinds.

import type { RewardContext, RewardFn } from '../index.js';

export const name = 'composite';

// `buildOne` is injected at construction time to avoid an ESM cycle.
// Each sub-reward is built inside its own try/catch so a bad sub-kind
// produces an error that names the offending child, not just the
// generic "unknown reward kind" from buildOne.
export function factory(
  params: any,
  { buildOne }: { buildOne: (cfg: any) => RewardFn }
): (ctx: RewardContext) => number {
  const weights = params.weights ?? {};
  if (!weights || typeof weights !== 'object') {
    throw new Error('composite.weights must be an object');
  }
  const names = Object.keys(weights);
  if (names.length === 0) {
    throw new Error('composite reward has no weights');
  }
  const subs: Array<{ weight: number; fn: RewardFn }> = [];
  for (const k of names) {
    const weight = weights[k];
    if (!Number.isFinite(weight)) {
      throw new Error(`composite weight for "${k}" must be finite, got ${weight}`);
    }
    if (k === 'composite') {
      throw new Error('composite cannot contain another composite');
    }
    const subParams = { kind: k, ...(params[k] ?? {}) };
    let fn: RewardFn;
    try {
      fn = buildOne(subParams);
    } catch (e: any) {
      throw new Error(`composite sub-reward "${k}": ${e.message}`);
    }
    subs.push({ weight, fn });
  }
  return (ctx: RewardContext) => subs.reduce((acc, { weight, fn }) => acc + weight * fn(ctx), 0);
}
