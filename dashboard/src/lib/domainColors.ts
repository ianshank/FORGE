/**
 * Single source of truth for FORGE domain colors used by the canvas renderer
 * (terrain tiles and agent factions). Kept in one place so the values can't
 * drift between the renderer and any future legend/inspector.
 */

/** Terrain type → hex color. Keys match `TerrainType` in `types/simulation`. */
export const TERRAIN_COLORS: Record<string, string> = {
  Ground: "#8B9556",
  Water: "#4A90D9",
  Wall: "#6B6B6B",
  Lava: "#D94A4A",
  Ice: "#B0D4E8",
  Sand: "#D4C07A",
  Forest: "#2D6B3F",
  Mountain: "#8B7355",
};

/** Ordered faction palette, indexed by team id. */
export const FACTION_COLORS: readonly string[] = [
  "#3B82F6",
  "#EF4444",
  "#10B981",
  "#F59E0B",
];

/** Color shown for dead/inactive agents. */
export const INACTIVE_AGENT_COLOR = "#666";

/**
 * Resolve a faction color by index, wrapping around the palette so any
 * team/agent id maps to a stable color.
 */
export function factionColor(index: number): string {
  if (!Number.isFinite(index) || FACTION_COLORS.length === 0) {
    return INACTIVE_AGENT_COLOR;
  }
  const wrapped = ((Math.trunc(index) % FACTION_COLORS.length) +
    FACTION_COLORS.length) %
    FACTION_COLORS.length;
  return FACTION_COLORS[wrapped];
}
