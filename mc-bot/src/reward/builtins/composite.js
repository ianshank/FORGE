// Weighted sum of named reward kinds.

export const name = 'composite';

// `buildOne` is injected at registration time to avoid an ESM cycle.
export function factory(params, { buildOne }) {
  const weights = params.weights ?? {};
  const subs = [];
  for (const [k, weight] of Object.entries(weights)) {
    if (!Number.isFinite(weight)) {
      throw new Error(`composite weight for ${k} must be finite, got ${weight}`);
    }
    const subParams = { kind: k, ...(params[k] ?? {}) };
    subs.push({ weight, fn: buildOne(subParams) });
  }
  if (subs.length === 0) {
    throw new Error('composite reward has no weights');
  }
  return (ctx) => subs.reduce((acc, { weight, fn }) => acc + weight * fn(ctx), 0);
}
