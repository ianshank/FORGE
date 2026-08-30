// Fast, isolated unit tests for web/app.js's parseSeed() -- deliberately not
// Playwright specs.
//
// The BigInt-overflow guard in parseSeed() is only reachable with a digit
// string long enough to exceed V8's own internal BigInt size cap (confirmed
// empirically at roughly 323 million decimal digits, independent of MAX_SEED).
// Constructing that string and filling it into a real DOM input through a
// browser would be slow and memory-heavy for what is really a pure-function
// property; node:test exercises parseSeed() directly, in milliseconds, with no
// browser at all.
import assert from "node:assert/strict";
import { test } from "node:test";

import { parseSeed } from "../../../web/app.js";

const MAX_SEED = (1n << 64n) - 1n;

test("parseSeed accepts a valid seed within range", () => {
  assert.equal(parseSeed("42"), 42n);
  assert.equal(parseSeed("0"), 0n);
  assert.equal(parseSeed(String(MAX_SEED)), MAX_SEED);
});

test("parseSeed trims surrounding whitespace", () => {
  assert.equal(parseSeed("  42  "), 42n);
});

test("parseSeed rejects non-digit input", () => {
  assert.equal(parseSeed(""), null);
  assert.equal(parseSeed("abc"), null);
  assert.equal(parseSeed("12.5"), null);
  assert.equal(parseSeed("-1"), null);
  assert.equal(parseSeed("1e10"), null);
});

test("parseSeed rejects a seed one past MAX_SEED", () => {
  // The exact regression this test pins: an off-by-one here would silently
  // accept an unrepresentable u64 or reject the largest valid one.
  assert.equal(parseSeed(String(MAX_SEED + 1n)), null);
});

test("parseSeed rejects a digit string too large for BigInt to represent, without throwing", () => {
  // 350 million is a comfortable margin over V8's ~323 million digit cap. If
  // parseSeed's try/catch regressed, BigInt(trimmed) would throw here and
  // this assertion would never run -- node:test reports that as a failure
  // carrying the original engine error, which is diagnostic enough on its own.
  const hugeDigits = "9".repeat(350_000_000);
  assert.equal(parseSeed(hugeDigits), null);
});
