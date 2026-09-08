import { createHash } from 'node:crypto';

import { sortKeysDeep } from './canonical_json.js';

/**
 * Named fallback when the TOML is missing. Matches the shipped
 * `unknown = 35` vocab (`max(index) + 1`). Twin of Rust
 * `DEFAULT_NUM_BLOCK_EMBEDDINGS`.
 */
export const DEFAULT_NUM_BLOCK_EMBEDDINGS = 36;

/**
 * Vocab size: `max(index) + 1` so sparse tables still cover every
 * index the encoder can emit.
 */
export function numBlockEmbeddings(blocks: Record<string, number>): number {
  const values = Object.values(blocks);
  if (values.length === 0) return 0;
  return Math.max(...values) + 1;
}

/**
 * SHA256 of the canonical JSON form of the `[blocks]` table.
 * Mirrors `BlockEmbeddings::canonical_sha256` in Rust.
 */
export function canonicalBlockEmbeddingsSha256(blocks: Record<string, number>): string {
  if (!blocks || typeof blocks !== 'object' || Array.isArray(blocks)) {
    throw new Error('block embeddings [blocks] table must be an object');
  }
  const canonical = JSON.stringify(sortKeysDeep(blocks));
  return createHash('sha256').update(canonical).digest('hex');
}

/**
 * Load and validate a block-embeddings TOML file using a caller-supplied
 * parser (keeps this module free of a TOML runtime dep at import time).
 */
export async function loadBlockEmbeddings(
  path: string,
  tomlParse: (text: string) => any,
): Promise<{ blocks: Record<string, number> }> {
  const { readFile } = await import('node:fs/promises');
  const raw = await readFile(path, 'utf8');
  const parsed = tomlParse(raw);
  const blocks = parsed?.blocks ?? {};
  if (!blocks || typeof blocks !== 'object' || Object.keys(blocks).length === 0) {
    throw new Error('block embeddings config has no [blocks] entries');
  }
  return { blocks };
}
