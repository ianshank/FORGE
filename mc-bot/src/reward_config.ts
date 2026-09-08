import { createHash } from 'node:crypto';
import { readFileSync, statSync } from 'node:fs';
import { basename, dirname, isAbsolute, join } from 'node:path';
import { parse as parseToml } from 'smol-toml';

import { sortKeysDeep } from './canonical_json.js';

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
 * Nested path keys whose file contents (not the path string) fold into
 * the rewards canonical SHA when the config was loaded from disk.
 * Twin of Rust `NESTED_REWARD_PATH_KEYS` and Python `NESTED_REWARD_PATH_KEYS`.
 */
export const NESTED_REWARD_PATH_KEY_CONFIG = 'config_path';
export const NESTED_REWARD_PATH_KEY_CRAFTING = 'crafting_config_path';
export const NESTED_REWARD_PATH_KEYS: readonly string[] = Object.freeze([
  NESTED_REWARD_PATH_KEY_CONFIG,
  NESTED_REWARD_PATH_KEY_CRAFTING,
]);

export function isNestedRewardPathKey(key: string): boolean {
  return (NESTED_REWARD_PATH_KEYS as readonly string[]).includes(key);
}

function isExistingFile(path: string): boolean {
  try {
    return statSync(path).isFile();
  } catch {
    return false;
  }
}

export interface CanonicalRewardsOptions {
  /** Directory of the rewards.toml this was loaded from. */
  baseDir?: string;
  /** TOML parser for nested files. Defaults to smol-toml. */
  tomlParse?: (text: string) => any;
}

/**
 * Validate a parsed rewards-toml object.
 */
export function buildRewardConfig(
  data: any,
  options: { sourcePath?: string; tomlParse?: (text: string) => any } = {},
): RewardConfig {
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
  const baseDir = options.sourcePath !== undefined ? dirname(options.sourcePath) : undefined;
  const tomlParse = options.tomlParse ?? parseToml;
  return {
    schemaVersion,
    entries,
    canonicalSha256() {
      return canonicalRewardsSha256(entries, { baseDir, tomlParse });
    },
    validate() {
      // Already validated at build time; provided for symmetry.
    },
  };
}

/**
 * SHA256 of the canonical JSON form of a parsed TOML document.
 * Mirrors Rust `canonical_json_sha256`.
 */
export function documentCanonicalSha256(doc: unknown): string {
  const canonical = JSON.stringify(sortKeysDeep(doc));
  return createHash('sha256').update(canonical).digest('hex');
}

/**
 * Resolve a nested reward path relative to `baseDir`.
 *
 * Candidates, first existing file wins:
 * 1. `value` if it is an absolute path to a file
 * 2. `join(baseDir, filename)` — sibling lookup
 * 3. `join(baseDir, value)`
 * 4. cwd-relative `value`
 */
export function resolveNestedRewardPath(baseDir: string, key: string, value: string): string {
  if (typeof value !== 'string' || value.length === 0) {
    throw new Error(`nested reward path key "${key}" must be a non-empty string`);
  }
  const candidates: string[] = [];
  if (isAbsolute(value)) {
    candidates.push(value);
  }
  candidates.push(join(baseDir, basename(value)));
  candidates.push(join(baseDir, value));
  if (!isAbsolute(value)) {
    candidates.push(value);
  }
  for (const candidate of candidates) {
    if (isExistingFile(candidate)) {
      return candidate;
    }
  }
  throw new Error(
    `nested reward file not found for \`${key}\` = \`${value}\` (searched sibling, base_dir-relative, and cwd-relative; hashing fails closed rather than using hardcoded milestone defaults)`,
  );
}

function foldNestedPathKeys(v: any, options: CanonicalRewardsOptions): any {
  if (Array.isArray(v)) {
    return v.map((item) => foldNestedPathKeys(item, options));
  }
  if (v !== null && typeof v === 'object') {
    const out: Record<string, any> = {};
    for (const k of Object.keys(v)) {
      const val = v[k];
      if (isNestedRewardPathKey(k)) {
        if (typeof val !== 'string' || val.length === 0) {
          throw new Error(`nested reward path key "${k}" must be a non-empty string`);
        }
        if (options.baseDir === undefined) {
          out[k] = val;
        } else {
          const nestedPath = resolveNestedRewardPath(options.baseDir, k, val);
          const raw = readFileSync(nestedPath, 'utf8');
          const parser = options.tomlParse ?? parseToml;
          out[k] = documentCanonicalSha256(parser(raw));
        }
      } else {
        out[k] = foldNestedPathKeys(val, options);
      }
    }
    return out;
  }
  return v;
}

/**
 * Compute the canonical SHA256 of a rewards-entries array. Matches
 * `RewardConfig::canonical_sha256` in Rust by emitting keys in
 * alphabetical order at every depth. When `options.baseDir` is set,
 * nested path keys are replaced with the nested file's canonical SHA.
 */
export function canonicalRewardsSha256(
  entries: RewardEntry[],
  options: CanonicalRewardsOptions = {},
): string {
  if (!Array.isArray(entries)) {
    throw new Error('entries must be an array');
  }
  const folded = foldNestedPathKeys(entries, options);
  const canonical = JSON.stringify(sortKeysDeep(folded));
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
 * caller-supplied parser. Nested path keys are folded immediately so a
 * missing nested file fails closed at load.
 */
export async function loadRewardConfig(
  path: string,
  tomlParse: (text: string) => any,
): Promise<RewardConfig> {
  const { readFile } = await import('node:fs/promises');
  const raw = await readFile(path, 'utf8');
  const cfg = buildRewardConfig(tomlParse(raw), { sourcePath: path, tomlParse });
  cfg.canonicalSha256();
  return cfg;
}
