/**
 * Recursively sort object keys before serialising. Rust's
 * `toml::Value::Table` is backed by a `BTreeMap` and therefore
 * always serialises keys in alphabetical order; JS must mirror that
 * so the two sides produce byte-identical canonical strings.
 *
 * Shared by `reward_config.ts` and `block_embeddings.ts`. Not the FNV
 * hasher in `hash.ts`.
 */
export function sortKeysDeep(v: unknown): unknown {
  if (Array.isArray(v)) return v.map(sortKeysDeep);
  if (v !== null && typeof v === 'object') {
    const record = v as Record<string, unknown>;
    const out: Record<string, unknown> = {};
    for (const k of Object.keys(record).sort()) {
      out[k] = sortKeysDeep(record[k]);
    }
    return out;
  }
  return v;
}
