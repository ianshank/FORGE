import { describe, expect, it } from "vitest";
import {
  FACTION_COLORS,
  INACTIVE_AGENT_COLOR,
  TERRAIN_COLORS,
  factionColor,
} from "../lib/domainColors";

describe("TERRAIN_COLORS", () => {
  it("provides a hex color for every known terrain type", () => {
    const expected = [
      "Ground",
      "Water",
      "Wall",
      "Lava",
      "Ice",
      "Sand",
      "Forest",
      "Mountain",
    ];
    for (const key of expected) {
      expect(TERRAIN_COLORS[key]).toMatch(/^#[0-9A-Fa-f]{6}$/);
    }
  });
});

describe("factionColor", () => {
  it("returns palette colors by index", () => {
    expect(factionColor(0)).toBe(FACTION_COLORS[0]);
    expect(factionColor(2)).toBe(FACTION_COLORS[2]);
  });

  it("wraps around the palette for out-of-range indices", () => {
    expect(factionColor(FACTION_COLORS.length)).toBe(FACTION_COLORS[0]);
    expect(factionColor(FACTION_COLORS.length + 1)).toBe(FACTION_COLORS[1]);
  });

  it("handles negative indices by wrapping forward", () => {
    expect(factionColor(-1)).toBe(FACTION_COLORS[FACTION_COLORS.length - 1]);
  });

  it("truncates fractional indices", () => {
    expect(factionColor(1.9)).toBe(FACTION_COLORS[1]);
  });

  it("returns the inactive color for non-finite indices", () => {
    expect(factionColor(Number.NaN)).toBe(INACTIVE_AGENT_COLOR);
    expect(factionColor(Number.POSITIVE_INFINITY)).toBe(INACTIVE_AGENT_COLOR);
  });
});
