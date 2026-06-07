import { describe, expect, it } from "vitest";
import { cn, formatDuration, formatNumber } from "../lib/utils";

describe("cn", () => {
  it("merges class names", () => {
    expect(cn("a", "b")).toBe("a b");
  });

  it("resolves conflicting tailwind utilities (last wins)", () => {
    expect(cn("px-2", "px-4")).toBe("px-4");
  });

  it("drops falsy values", () => {
    expect(cn("a", false, null, undefined, "b")).toBe("a b");
  });
});

describe("formatNumber", () => {
  it("formats integers", () => {
    expect(formatNumber(1234)).toBe("1,234");
  });

  it("respects fraction digits", () => {
    expect(formatNumber(12.3456, 2)).toBe("12.35");
  });

  it("returns an em dash for non-finite input", () => {
    expect(formatNumber(Number.NaN)).toBe("—");
    expect(formatNumber(Number.POSITIVE_INFINITY)).toBe("—");
  });
});

describe("formatDuration", () => {
  it("formats seconds only", () => {
    expect(formatDuration(45)).toBe("45s");
  });

  it("formats minutes and seconds", () => {
    expect(formatDuration(125)).toBe("2m 5s");
  });

  it("formats hours, minutes and seconds", () => {
    expect(formatDuration(3661)).toBe("1h 1m 1s");
  });

  it("returns an em dash for negative input", () => {
    expect(formatDuration(-1)).toBe("—");
  });
});
