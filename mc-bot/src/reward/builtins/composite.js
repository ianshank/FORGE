// Weighted sum of named reward kinds.

export const name = 'composite';

// `buildOne` is injected at construction time to avoid an ESM cycle.
// Each sub-reward is built inside its own try/catch so a bad sub-kind
// produces an error that names the offending child, not just the
// generic "unknown reward kind" from buildOne.
export function factory(params, { buildOne }) {
  const weights = params.weights ?? {};
  if (!weights || typeof weights !== 'object') {
    throw new Error('composite.weights must be an object');
  }
  const names = Object.keys(weights);
  if (names.length === 0) {
    throw new Error('composite reward has no weights');
  }
  const subs = [];
  for (const k of names) {
    const weight = weights[k];
    if (!Number.isFinite(weight)) {
      throw new Error(`composite weight for "${k}" must be finite, got ${weight}`);
    }
    if (k === 'composite') {
      throw new Error('composite cannot contain another composite');
    }
    const subParams = { kind: k, ...(params[k] ?? {}) };
    let fn;
    try {
      fn = buildOne(subParams);
    } catch (e) {
      throw new Error(`composite sub-reward "${k}": ${e.message}`);
    }
    subs.push({ weight, fn });
  }
  return (ctx) => subs.reduce((acc, { weight, fn }) => acc + weight * fn(ctx), 0);
}
