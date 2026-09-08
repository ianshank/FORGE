import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parse } from 'smol-toml';

import {
  DEFAULT_NUM_BLOCK_EMBEDDINGS,
  canonicalBlockEmbeddingsSha256,
  loadBlockEmbeddings,
  numBlockEmbeddings,
} from '../src/block_embeddings.js';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../..');

describe('block_embeddings', () => {
  it('vocab size is max(index)+1', () => {
    assert.equal(numBlockEmbeddings({ air: 0, unknown: 7, stone: 1 }), 8);
  });

  it('hash is stable under key reorder', () => {
    const a = canonicalBlockEmbeddingsSha256({ air: 0, stone: 1, unknown: 2 });
    const b = canonicalBlockEmbeddingsSha256({ unknown: 2, air: 0, stone: 1 });
    assert.equal(a, b);
    assert.equal(a.length, 64);
  });

  it('hash changes when an index changes', () => {
    const a = canonicalBlockEmbeddingsSha256({ air: 0, stone: 1, unknown: 2 });
    const b = canonicalBlockEmbeddingsSha256({ air: 0, stone: 1, unknown: 3 });
    assert.notEqual(a, b);
  });

  it('rejects non-object input', () => {
    assert.throws(() => canonicalBlockEmbeddingsSha256([] as any));
    assert.throws(() => canonicalBlockEmbeddingsSha256(null as any));
  });

  it('shipped block_embeddings.toml hash matches Rust pin', async () => {
    const { blocks } = await loadBlockEmbeddings(
      resolve(repoRoot, 'configs/minecraft/block_embeddings.toml'),
      parse,
    );
    assert.equal(numBlockEmbeddings(blocks), DEFAULT_NUM_BLOCK_EMBEDDINGS);
    assert.equal(blocks.unknown, 35);
    assert.equal(blocks.air, 0);
    assert.equal(
      canonicalBlockEmbeddingsSha256(blocks),
      'b5aef9f434474c17ffbdee7fe894ae93ada0f4e6477cb51b0a8cf4fc0d7a7a7e',
      'block-embeddings obs-layout pin drift — update Rust/JS/Python pins together',
    );
  });
});
