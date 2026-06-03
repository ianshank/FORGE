import { describe, expect, it } from "vitest";
import {
  CANVAS_COLORS,
  findNearestAgent,
  healthFraction,
  HEALTH_FIXED_POINT_SCALE,
  RENDER_GEOMETRY,
} from "../lib/canvasRender";
import type { AgentState } from "../types/simulation";

function agent(partial: Partial<AgentState> & Pick<AgentState, "id" | "x" | "y">): AgentState {
  return {
    health: HEALTH_FIXED_POINT_SCALE,
    alive: true,
    teamId: null,
    intent: null,
    visionRadius: 3,
    ...partial,
  };
}

describe("healthFraction", () => {
  it("normalises full health to 1", () => {
    expect(healthFraction(HEALTH_FIXED_POINT_SCALE)).toBe(1);
  });

  it("normalises half health to 0.5", () => {
    expect(healthFraction(HEALTH_FIXED_POINT_SCALE / 2)).toBe(0.5);
  });

  it("clamps overshoot to 1", () => {
    expect(healthFraction(HEALTH_FIXED_POINT_SCALE * 4)).toBe(1);
  });

  it("returns 0 for non-positive or non-finite input", () => {
    expect(healthFraction(0)).toBe(0);
    expect(healthFraction(-10)).toBe(0);
    expect(healthFraction(Number.NaN)).toBe(0);
  });

  it("returns 0 when the scale is invalid", () => {
    expect(healthFraction(100, 0)).toBe(0);
  });
});

describe("findNearestAgent", () => {
  const cellSize = 10;
  const agents = [
    agent({ id: 1, x: 0, y: 0 }),
    agent({ id: 2, x: 5, y: 5 }),
  ];

  it("returns the agent nearest to a click within range", () => {
    // Centre of agent 2's cell is (55, 55).
    const hit = findNearestAgent(agents, { x: 55, y: 55 }, cellSize);
    expect(hit?.id).toBe(2);
  });

  it("returns null when the click is outside the select radius", () => {
    const miss = findNearestAgent(agents, { x: 1000, y: 1000 }, cellSize);
    expect(miss).toBeNull();
  });

  it("returns null for an empty agent list", () => {
    expect(findNearestAgent([], { x: 0, y: 0 }, cellSize)).toBeNull();
  });

  it("returns null for a non-positive cell size", () => {
    expect(findNearestAgent(agents, { x: 0, y: 0 }, 0)).toBeNull();
  });

  it("honours a custom max-cell radius", () => {
    // Just outside 1 cell but inside the default 2 cells.
    const point = { x: 5 + cellSize * 1.5, y: 5 };
    expect(findNearestAgent(agents, point, cellSize, 1)).toBeNull();
    expect(findNearestAgent(agents, point, cellSize, 2)).not.toBeNull();
  });
});

describe("constants", () => {
  it("exposes colour and geometry tables", () => {
    expect(CANVAS_COLORS.background).toMatch(/^#/);
    expect(RENDER_GEOMETRY.agentRadiusFactor).toBeGreaterThan(0);
  });
});
