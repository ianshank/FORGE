// SHA256 of a canonical-form action map — must agree byte-for-byte
// with the Rust side (`ActionMap::canonical_sha256` in
// crates/forge-env-mc/src/action_map.rs).
//
// Canonical form: serde_json string of the entries vector sorted by id.

import { createHash } from 'node:crypto';

/**
 * Canonicalise an action-map entry to match the Rust serde output.
 * Critical that field order matches `ActionEntry`'s declaration:
 *   { id, ...kind_fields }   (kind is #[serde(flatten)])
 *
 * @param {{ id: number, kind: string, [k: string]: any }} entry
 */
function canonicalEntry(entry) {
  // Rust serializes `id` first, then the flattened kind. We need to
  // produce the same JSON text. Build the object with keys in that
  // order so JSON.stringify preserves insertion order (Node has done
  // this for string keys since the language spec was tightened).
  const out = { id: entry.id };
  // `kind` is the discriminant tag, then kind-specific fields.
  out.kind = entry.kind;
  for (const [k, v] of Object.entries(entry)) {
    if (k === 'id' || k === 'kind') continue;
    out[k] = v;
  }
  return out;
}

/**
 * Compute the canonical SHA256 (hex) for an action map's entries.
 *
 * Must match `forge_env_mc::action_map::ActionMap::canonical_sha256`.
 *
 * @param {Array<{id: number, kind: string}>} entries
 * @returns {string}
 */
export function canonicalSha256(entries) {
  if (!Array.isArray(entries)) {
    throw new Error('entries must be an array');
  }
  const sorted = [...entries].sort((a, b) => a.id - b.id);
  const canonical = sorted.map(canonicalEntry);
  const text = JSON.stringify(canonical);
  return createHash('sha256').update(text).digest('hex');
}
