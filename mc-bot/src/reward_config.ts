import { createHash } from 'node:crypto';

export interface RewardEntry {
  kind: string;
  [key: string]: any;
}

export interface RewardConfig {
  schemaVersion: number;
  entries: RewardEntry[];
  canonicalSha256(): string;
  validate(): void;
}

/**
 * Validate a parsed rewards-toml object.
 */
export function buildRewardConfig(data: any): RewardConfig {
  if (!data || typeof data !== 'object') {
    throw new Error('rewards config must be an object');
  }
  const entries = data.reward ?? [];
  if (!Array.isArray(entries) || entries.length === 0) {
    throw new Error('rewards config has no [[reward]] entries');
  }
  for (let i = 0; i < entries.length; i++) {
    const e = entries[i];
    if (!e || typeof e !== 'object') {
      throw new Error(`reward[${i}] must be an object`);
    }
    if (typeof e.kind !== 'string' || e.kind.length === 0) {
      throw new Error(`reward[${i}] missing string "kind"`);
    }
  }
  const schemaVersion = data.schema_version ?? 1;
  return {
    schemaVersion,
    entries,
    canonicalSha256() {
      return canonicalRewardsSha256(entries);
    },
    validate() {
      // Already validated at build time; provided for symmetry.
    },
  };
}

/**
 * Recursively sort object keys before serialising. Rust's
 * `toml::Value::Table` is backed by a `BTreeMap` and therefore
 * always serialises keys in alphabetical order; we mirror that here
 * so the two sides produce byte-identical canonical strings.
 */
function sortKeysDeep(v: any): any {
  if (Array.isArray(v)) return v.map(sortKeysDeep);
  if (v !== null && typeof v === 'object') {
    const out: Record<string, any> = {};
    for (const k of Object.keys(v).sort()) {
      out[k] = sortKeysDeep(v[k]);
    }
    return out;
  }
  return v;
}

/**
 * Compute the canonical SHA256 of a rewards-entries array. Matches
 * `RewardConfig::canonical_sha256` in Rust by emitting keys in
 * alphabetical order at every depth.
 */
export function canonicalRewardsSha256(entries: RewardEntry[]): string {
  if (!Array.isArray(entries)) {
    throw new Error('entries must be an array');
  }
  const canonical = JSON.stringify(sortKeysDeep(entries));
  return createHash('sha256').update(canonical).digest('hex');
}

/**
 * Combine action map + rewards hashes into the global schema_id.
 * Order matters: `sha256(action_map_hash + ":" + rewards_hash)`.
 *
 * Mirrors `forge_env_mc::reward_config::combined_schema_id`.
 */
export function combinedSchemaId(actionMapHash: string, rewardsHash: string): string {
  const combined = `${actionMapHash}:${rewardsHash}`;
  return createHash('sha256').update(combined).digest('hex');
}

/**
 * Load and validate a rewards config from a TOML file using a
 * caller-supplied parser (keeps this module dep-free).
 */
export async function loadRewardConfig(
  path: string,
  tomlParse: (text: string) => any
): Promise<RewardConfig> {
  const { readFile } = await import('node:fs/promises');
  const raw = await readFile(path, 'utf8');
  return buildRewardConfig(tomlParse(raw));
}
