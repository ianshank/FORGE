// Reward = -Δ(L1 distance to target). Lower distance ⇒ positive reward.

export const name = 'distance_to_goal';

function l1(a, b) {
  return Math.abs(a.x - b.x) + Math.abs(a.y - b.y) + Math.abs(a.z - b.z);
}

export function factory(params) {
  const target = params.target ?? { x: 0, y: 64, z: 0 };
  const clip = Number.isFinite(params.clip) ? params.clip : 1.0;
  return ({ prev, curr }) => {
    if (!prev || !curr || !prev.position || !curr.position) return 0;
    const dPrev = l1(prev.position, target);
    const dCurr = l1(curr.position, target);
    const delta = dPrev - dCurr;
    return Math.max(-clip, Math.min(clip, delta));
  };
}
