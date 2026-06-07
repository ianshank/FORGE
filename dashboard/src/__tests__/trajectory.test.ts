import { describe, expect, it } from "vitest";
import {
  cumulativeReward,
  parseTrajectory,
  type TrajectoryStep,
} from "../lib/trajectory";

const VALID_V2 = {
  format_version: 2,
  env_id: "minecraft",
  schema_id: "abc123",
  episode_id: "ep-1",
  seed: 42,
  obs_dim: 4,
  action_count: 3,
  final_reward: 1.5,
  started_at: "2026-01-01T00:00:00Z",
  ended_at: "2026-01-01T00:01:00Z",
  steps: [
    {
      tick: 0,
      obs: [0, 0, 0, 0],
      action_id: 1,
      policy_target: [0.1, 0.8, 0.1],
      value_target: 0.5,
      reward: 1.0,
      terminated: false,
      truncated: false,
    },
    {
      tick: 1,
      obs: [1, 0, 0, 0],
      action_id: 2,
      policy_target: [0.2, 0.2, 0.6],
      value_target: 0.25,
      reward: 0.5,
      terminated: true,
      truncated: false,
    },
  ],
};

describe("parseTrajectory", () => {
  it("parses a well-formed v2 trajectory", () => {
    const result = parseTrajectory(VALID_V2);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    const t = result.trajectory;
    expect(t.envId).toBe("minecraft");
    expect(t.obsDim).toBe(4);
    expect(t.actionCount).toBe(3);
    expect(t.steps).toHaveLength(2);
    expect(t.steps[0].actionId).toBe(1);
    expect(t.steps[0].policyTargetLen).toBe(3);
    expect(t.steps[1].terminated).toBe(true);
  });

  it("rejects non-object input", () => {
    const result = parseTrajectory(42);
    expect(result.ok).toBe(false);
  });

  it("rejects input without a steps array", () => {
    const result = parseTrajectory({ env_id: "x" });
    expect(result.ok).toBe(false);
  });

  it("defaults missing optional header fields", () => {
    const result = parseTrajectory({ steps: [] });
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.trajectory.envId).toBe("unknown");
    expect(result.trajectory.seed).toBeNull();
    expect(result.trajectory.endedAt).toBeNull();
  });

  it("coerces malformed step fields to safe defaults", () => {
    const result = parseTrajectory({
      steps: [{ tick: "nope", reward: null }],
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    const step = result.trajectory.steps[0];
    expect(step.tick).toBe(0);
    expect(step.reward).toBe(0);
    expect(step.policyTargetLen).toBe(0);
  });
});

describe("cumulativeReward", () => {
  it("accumulates rewards across steps", () => {
    const steps = [
      { reward: 1 },
      { reward: 0.5 },
      { reward: -0.25 },
    ] as TrajectoryStep[];
    expect(cumulativeReward(steps)).toEqual([1, 1.5, 1.25]);
  });

  it("returns an empty array for no steps", () => {
    expect(cumulativeReward([])).toEqual([]);
  });
});
