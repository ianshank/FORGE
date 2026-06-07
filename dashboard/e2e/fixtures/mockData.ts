/**
 * Canned backend payloads shared across E2E specs. Field names mirror the
 * dashboard's runtime contracts (see src/types/simulation.ts + messageParser).
 */

/** Fixed-point health where 655360 == 10.0 (matches the simulation encoding). */
export const FULL_HEALTH = 655360;

/** A representative WebSocket StateUpdate message. */
export const STATE_UPDATE = {
  type: "StateUpdate",
  payload: {
    tick: 42,
    agents: [
      {
        id: 0,
        x: 1,
        y: 1,
        health: FULL_HEALTH,
        alive: true,
        teamId: 0,
        intent: "move",
        visionRadius: 3,
      },
      {
        id: 1,
        x: 3,
        y: 3,
        health: 0,
        alive: false,
        teamId: 1,
        intent: null,
        visionRadius: 3,
      },
    ],
    gridWidth: 8,
    gridHeight: 8,
    events: [],
    schemaVersion: 1,
  },
} as const;

/** GET /api/metrics response. */
export const SERVER_METRICS = {
  simulationTicks: 500,
  stepsPerSecond: 25,
  wsConnections: 1,
  uptimeSeconds: 120,
} as const;

/** POST /api/scenario/remix success response. */
export const REMIX_OK = {
  success: true,
  seed: 42,
  gridWidth: 64,
  gridHeight: 64,
} as const;

/** Build an SSE body (text/event-stream) terminated by the end marker. */
export function sseBody(lines: string[]): string {
  const frames = lines.map((l) => `data: ${JSON.stringify(l)}\n\n`);
  frames.push(`data: ${JSON.stringify("__STREAM_END__")}\n\n`);
  return frames.join("");
}

/** Default demo output lines streamed by the mocked SSE endpoint. */
export const DEMO_LINES = [
  "Starting demo…",
  "Generating 8x8 world",
  "Spawning 2 agents",
  "Simulation complete",
];
