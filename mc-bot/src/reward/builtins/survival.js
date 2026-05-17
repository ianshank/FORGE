// +R per tick alive. Trivial but useful as a baseline.

export const name = 'survival';

export function factory(params) {
  const value = Number.isFinite(params.value) ? params.value : 0.01;
  return () => value;
}
