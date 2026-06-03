/**
 * Pure helpers and constants for the simulation canvas renderer.
 *
 * Keeping the geometry/colour values here (rather than as magic numbers inside
 * the component) makes them a single source of truth and lets the selection
 * maths be unit-tested without a DOM/canvas.
 */

import type { AgentState } from "../types/simulation";

/**
 * Fixed-point scale for agent health. The simulation encodes health as a
 * `fixed` value where `655360 == 10.0`, so dividing by this yields [0, 1].
 */
export const HEALTH_FIXED_POINT_SCALE = 655360;

/** Health fraction above which the health bar renders in the "healthy" colour. */
export const HEALTH_HIGH_FRACTION = 0.5;

/** Geometry factors (multiples of the configured cell size) for agent glyphs. */
export const RENDER_GEOMETRY = {
  agentRadiusFactor: 0.35,
  selectionRingPadding: 3,
  selectionLineWidth: 2,
  gridLineWidth: 0.5,
  healthBarWidthFactor: 0.8,
  healthBarHeight: 2,
  healthBarOffsetY: 3,
  /** Click is treated as a hit within this many cells of an agent centre. */
  selectRadiusCells: 2,
  /** Minimum glyph font size in px. */
  minFontSize: 8,
  fontSizeFactor: 0.6,
} as const;

/** Concrete colour strings for the 2D canvas context. */
export const CANVAS_COLORS = {
  background: "#0d1525",
  gridLine: "rgba(255,255,255,0.08)",
  selection: "#38bdf8",
  healthTrack: "#333333",
  healthHigh: "#22c55e",
  healthLow: "#ef4444",
  intentLabel: "#ffffff",
} as const;

/**
 * Normalise a raw fixed-point health value to a [0, 1] fraction.
 */
export function healthFraction(
  rawHealth: number,
  scale: number = HEALTH_FIXED_POINT_SCALE,
): number {
  if (!Number.isFinite(rawHealth) || rawHealth <= 0 || scale <= 0) return 0;
  return Math.min(rawHealth / scale, 1);
}

/** A point in canvas pixel space. */
export interface CanvasPoint {
  x: number;
  y: number;
}

/**
 * Find the agent whose glyph centre is nearest to `point`, within
 * `maxCells` cells. Returns `null` when no agent is close enough.
 *
 * Distances are compared squared to avoid needless `sqrt`.
 */
export function findNearestAgent(
  agents: readonly AgentState[],
  point: CanvasPoint,
  cellSize: number,
  maxCells: number = RENDER_GEOMETRY.selectRadiusCells,
): AgentState | null {
  if (agents.length === 0 || cellSize <= 0) return null;

  let nearest: AgentState | null = null;
  let bestDist = Number.POSITIVE_INFINITY;
  for (const agent of agents) {
    const cx = agent.x * cellSize + cellSize / 2;
    const cy = agent.y * cellSize + cellSize / 2;
    const dist = (cx - point.x) ** 2 + (cy - point.y) ** 2;
    if (dist < bestDist) {
      bestDist = dist;
      nearest = agent;
    }
  }

  const maxDist = (cellSize * maxCells) ** 2;
  return nearest && bestDist <= maxDist ? nearest : null;
}
