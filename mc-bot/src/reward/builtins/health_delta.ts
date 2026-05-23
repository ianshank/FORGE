// Sign-of-delta(health) * value. Negative when taking damage.

import type { RewardContext } from '../index.js';

export const name = 'health_delta';

export function factory(params: any): (ctx: RewardContext) => number {
  const value = Number.isFinite(params.value) ? params.value : 1.0;
  return ({ prev, curr }) => {
    if (!prev || !curr) return 0;
    const a = Number.isFinite(prev.health) ? prev.health : 0;
    const b = Number.isFinite(curr.health) ? curr.health : 0;
    if (b > a) return value;
    if (b < a) return -value;
    return 0;
  };
}
