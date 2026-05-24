// Tiny shared FNV-1a-style hash helper.  Kept in its own module so the
// observation.js / observation_grid.js pair can both consume it without
// introducing a circular import.

const DEFAULT_HASH_MOD = 4096;

/**
 * Coerce a value to a finite number, returning `fallback` when the
 * input is non-finite (NaN, ±Infinity, or non-numeric). Used at every
 * field-read boundary in the observation pipeline so a corrupt
 * mineflayer payload doesn't NaN-propagate through to the trainer.
 */
export function finiteNumber(value: unknown, fallback: number = 0): number {
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue : fallback;
}

/**
 * Deterministic 32-bit hash of an arbitrary stringifiable value, then
 * reduced modulo `modulus` (defaults to {@link DEFAULT_HASH_MOD}).
 *
 * Same algorithm as the Rust-side `stable_string_hash` (canonical
 * FNV-1a). Cross-language equivalence is pinned by the xlang protocol
 * tests; do not "optimise" the constants without bumping the schema.
 */
export function stableStringHash(text: any, modulus: number = DEFAULT_HASH_MOD): number {
  const hashMod = Number.isInteger(modulus) && modulus > 0 ? modulus : DEFAULT_HASH_MOD;
  let hash = 2166136261;
  for (const char of String(text)) {
    hash ^= char.charCodeAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0) % hashMod;
}

export { DEFAULT_HASH_MOD };
