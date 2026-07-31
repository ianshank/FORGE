// SHA256 of a canonical-form action map — must agree byte-for-byte
// with the Rust side (`ActionMap::canonical_sha256` in
// crates/forge-env-mc/src/action_map.rs).
//
// Canonical form: serde_json string of the entries vector sorted by id.

import { createHash } from 'node:crypto';

/**
 * Canonical field order per `ActionKind` variant. Keys are the
 * `kind` discriminant (snake_case, matching serde `rename_all`),
 * values are the kind-specific field order from the Rust enum
 * declaration in `crates/forge-env-mc/src/action_map.rs`.
 *
 * `id` and `kind` are always emitted first (in that order); the
 * arrays below list ONLY the additional fields. Variants with no
 * extra fields (`jump`, `attack`, `use`) map to an empty array.
 *
 * If you add or reorder a Rust `ActionKind` variant field, update
 * the corresponding entry here AND re-run
 * `mc-bot/test/schema_id.test.ts` — the pinned xlang hash will
 * surface drift on either side.
 */
const CANONICAL_FIELD_ORDER: Record<string, string[]> = Object.freeze({
  noop: ['ticks'],
  move: ['direction', 'ticks'],
  jump: [],
  attack: [],
  use: [],
  place: ['hotbar_slot'],
  select_slot: ['hotbar_slot'],
  look: ['yaw_deg', 'pitch_deg'],
  eat: [],
  craft_planks: [],
  craft_sticks: [],
  craft_crafting_table: [],
  place_crafting_table: [],
  craft_wooden_pickaxe: [],
  mine_stone: [],
  craft_stone_pickaxe: [],
  craft_furnace: [],
  place_furnace: [],
  smelt_iron: [],
  craft_iron_pickaxe: [],
  equip_pickaxe: [],
  sprint: ['ticks'],
  sneak: ['ticks'],
  swim_up: ['ticks'],
});

/**
 * Canonicalise an action-map entry to match the Rust serde output.
 *
 * Rust emits `id` first, then `kind` (the `#[serde(tag = "kind")]`
 * discriminant), then the kind-specific fields in struct-declaration
 * order. We replicate that explicitly here — Node preserves
 * insertion order for string keys, so the resulting `JSON.stringify`
 * produces byte-identical output to serde_json on the Rust side.
 *
 * Unknown `kind` values fall back to insertion order from
 * `Object.entries`, preserving forward-compat for additive variants
 * the Node side hasn't been taught about yet. The Rust side will
 * still reject anything truly unknown, so this only matters for
 * future-version tolerance.
 */
function canonicalEntry(entry: { id: number; kind: string; [k: string]: any }): any {
  const out: Record<string, any> = { id: entry.id, kind: entry.kind };
  const order = CANONICAL_FIELD_ORDER[entry.kind];
  if (order) {
    for (const field of order) {
      if (Object.prototype.hasOwnProperty.call(entry, field)) {
        out[field] = entry[field];
      }
    }
  } else {
    // Forward-compat fallback for unknown kinds. Logs a warning so
    // drift is visible during development. Rust will reject the
    // entry on the other side if it truly isn't a known variant.
    for (const [k, v] of Object.entries(entry)) {
      if (k === 'id' || k === 'kind') continue;
      out[k] = v;
    }
  }
  return out;
}

/**
 * Compute the canonical SHA256 (hex) for an action map's entries.
 *
 * Must match `forge_env_mc::action_map::ActionMap::canonical_sha256`.
 */
export function canonicalSha256(entries: Array<{ id: number; kind: string; [k: string]: any }>): string {
  if (!Array.isArray(entries)) {
    throw new Error('entries must be an array');
  }
  const sorted = [...entries].sort((a, b) => a.id - b.id);
  const canonical = sorted.map(canonicalEntry);
  const text = JSON.stringify(canonical);
  return createHash('sha256').update(text).digest('hex');
}
