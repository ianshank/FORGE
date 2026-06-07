/**
 * Client-side parser for the `forge-replay` v2 trajectory JSON format.
 *
 * Mirrors `TrajectoryV2` / `StepV2` from `crates/forge-replay/src/v2.rs`.
 * Parsing is defensive: malformed input yields a descriptive error rather
 * than throwing, so the UI can surface it cleanly.
 */

/** One transition in a v2 trajectory. */
export interface TrajectoryStep {
  tick: number;
  actionId: number;
  valueTarget: number;
  reward: number;
  terminated: boolean;
  truncated: boolean;
  /** Length of the policy-target distribution (kept, not the full vector). */
  policyTargetLen: number;
}

/** A parsed v2 trajectory (observation vectors are dropped to save memory). */
export interface Trajectory {
  envId: string;
  episodeId: string;
  schemaId: string;
  seed: number | null;
  obsDim: number;
  actionCount: number;
  finalReward: number;
  startedAt: string;
  endedAt: string | null;
  steps: TrajectoryStep[];
}

/** Result of a parse attempt. */
export type ParseResult =
  | { ok: true; trajectory: Trajectory }
  | { ok: false; error: string };

function asNumber(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function asString(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

/**
 * Parse a raw object (from `JSON.parse`) into a {@link Trajectory}.
 */
export function parseTrajectory(raw: unknown): ParseResult {
  if (typeof raw !== "object" || raw === null) {
    return { ok: false, error: "Expected a JSON object at the top level." };
  }
  const obj = raw as Record<string, unknown>;

  if (!Array.isArray(obj.steps)) {
    return { ok: false, error: "Missing or invalid `steps` array." };
  }

  const steps: TrajectoryStep[] = obj.steps.map((s) => {
    const step = (s ?? {}) as Record<string, unknown>;
    const policy = step.policy_target;
    return {
      tick: asNumber(step.tick),
      actionId: asNumber(step.action_id),
      valueTarget: asNumber(step.value_target),
      reward: asNumber(step.reward),
      terminated: step.terminated === true,
      truncated: step.truncated === true,
      policyTargetLen: Array.isArray(policy) ? policy.length : 0,
    };
  });

  return {
    ok: true,
    trajectory: {
      envId: asString(obj.env_id, "unknown"),
      episodeId: asString(obj.episode_id, "—"),
      schemaId: asString(obj.schema_id),
      seed: typeof obj.seed === "number" ? obj.seed : null,
      obsDim: asNumber(obj.obs_dim),
      actionCount: asNumber(obj.action_count),
      finalReward: asNumber(obj.final_reward),
      startedAt: asString(obj.started_at),
      endedAt: typeof obj.ended_at === "string" ? obj.ended_at : null,
      steps,
    },
  };
}

/** Running cumulative reward up to and including each step. */
export function cumulativeReward(steps: TrajectoryStep[]): number[] {
  let acc = 0;
  return steps.map((s) => {
    acc += s.reward;
    return acc;
  });
}
