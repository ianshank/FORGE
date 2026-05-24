// Action map loader and validator. Mirrors
// `crates/forge-env-mc/src/action_map.rs` but is dep-free at parse
// time — TOML parsing is delegated to a caller-supplied parser
// (smol-toml in production, hand-built fixture object in tests).

import { canonicalSha256 } from './schema_id.js';

const VALID_KINDS = new Set([
  'noop',
  'move',
  'jump',
  'attack',
  'use',
  'place',
  'select_slot',
  'look',
  'eat',
  'craft_planks',
  'craft_sticks',
  'craft_crafting_table',
  'place_crafting_table',
  'craft_wooden_pickaxe',
  'mine_stone',
  'craft_stone_pickaxe',
  'craft_furnace',
  'place_furnace',
  'smelt_iron',
  'craft_iron_pickaxe',
  'equip_pickaxe',
  'sprint',
  'sneak',
  'swim_up',
]);

const VALID_MOVE_DIRECTIONS = new Set(['forward', 'back', 'left', 'right']);

export interface ActionEntry {
  id: number;
  kind: string;
  ticks?: number;
  direction?: 'forward' | 'back' | 'left' | 'right' | string;
  hotbar_slot?: number;
  yaw_deg?: number;
  pitch_deg?: number;
  [key: string]: any;
}

export interface ActionMap {
  schemaVersion: number;
  entries: ActionEntry[];
  actionCount: number;
  schemaId: string;
  get(id: number): ActionEntry | undefined;
}

/**
 * Validate one action entry's shape against its `kind`.
 */
function validateEntry(e: any): asserts e is ActionEntry {
  if (!e || typeof e !== 'object') {
    throw new Error(`entry is not an object: ${JSON.stringify(e)}`);
  }
  if (!Number.isInteger(e.id) || e.id < 0) {
    throw new Error(`entry has invalid id: ${JSON.stringify(e)}`);
  }
  if (typeof e.kind !== 'string' || !VALID_KINDS.has(e.kind)) {
    throw new Error(`entry has unknown kind: ${e.kind} (id=${e.id})`);
  }
  switch (e.kind) {
    case 'noop':
    case 'sprint':
    case 'sneak':
    case 'swim_up':
      if (e.ticks !== undefined && (!Number.isInteger(e.ticks) || e.ticks < 1)) {
        throw new Error(`${e.kind}.ticks must be >= 1 (id=${e.id})`);
      }
      break;
    case 'move':
      if (!VALID_MOVE_DIRECTIONS.has(e.direction)) {
        throw new Error(`move.direction invalid: ${e.direction} (id=${e.id})`);
      }
      break;
    case 'place':
    case 'select_slot': {
      const slot = e.hotbar_slot;
      if (!Number.isInteger(slot) || slot < 0 || slot > 8) {
        throw new Error(`hotbar_slot must be 0..=8 (id=${e.id})`);
      }
      break;
    }
    case 'look':
      if (!Number.isFinite(e.yaw_deg) || !Number.isFinite(e.pitch_deg)) {
        throw new Error(`look angles must be finite (id=${e.id})`);
      }
      break;
    default:
      // Other kinds (eat, craft_planks, craft_wooden_pickaxe) have no extra fields.
      break;
  }
}

/**
 * Build and validate an ActionMap from parsed TOML data.
 */
export function buildActionMap(data: any): ActionMap {
  if (!data || typeof data !== 'object') {
    throw new Error('action map must be an object');
  }
  const entries = data.action ?? [];
  if (!Array.isArray(entries) || entries.length === 0) {
    throw new Error('action map has no entries');
  }
  const seen = new Set<number>();
  for (const e of entries) {
    validateEntry(e);
    if (seen.has(e.id)) {
      throw new Error(`duplicate action id: ${e.id}`);
    }
    seen.add(e.id);
  }
  // Dense check
  for (let i = 0; i < entries.length; i++) {
    if (!seen.has(i)) {
      throw new Error(`action ids must be dense 0..${entries.length}, missing ${i}`);
    }
  }
  return {
    schemaVersion: data.schema_version ?? 1,
    entries,
    actionCount: entries.length,
    schemaId: canonicalSha256(entries),
    get(id: number) {
      return entries.find((e) => e.id === id);
    },
  };
}

/**
 * Load and validate an action map from a TOML file using a
 * caller-supplied parser. Keeps this module dep-free.
 */
export async function loadActionMap(
  path: string,
  tomlParse: (text: string) => any
): Promise<ActionMap> {
  const { readFile } = await import('node:fs/promises');
  const raw = await readFile(path, 'utf8');
  const data = tomlParse(raw);
  return buildActionMap(data);
}
