/** Core simulation types matching the Rust server's JSON schema. */

/** Terrain types matching forge-types TerrainType enum. */
export type TerrainType =
  | "Ground"
  | "Water"
  | "Wall"
  | "Lava"
  | "Ice"
  | "Sand"
  | "Forest"
  | "Mountain";

/** A single cell in the terrain grid. */
export interface TerrainCell {
  terrain: TerrainType;
  elevation: number;
  visible: boolean;
  agentId: number | null;
  resourceId: number | null;
}

/** Snapshot of an agent's state at a given tick. */
export interface AgentState {
  id: number;
  x: number;
  y: number;
  health: number;
  alive: boolean;
  teamId: number | null;
  intent: string | null;
  visionRadius: number;
}

/** Status of an objective. */
export type ObjectiveStatus = "InProgress" | "Completed" | "Failed";

/** Objective state for display. */
export interface ObjectiveState {
  id: string;
  description: string;
  status: ObjectiveStatus;
  progress: number;
}

/** Full simulation state broadcast per tick. */
export interface SimulationState {
  tick: number;
  agents: AgentState[];
  gridWidth: number;
  gridHeight: number;
  events: string[];
  schemaVersion: number;
}

/** Training metrics for the metrics dashboard. */
export interface TrainingMetrics {
  episode: number;
  reward: number;
  winRate: number;
  stepsPerSecond: number;
  lossPolicy: number;
  lossValue: number;
  entropy: number;
}

/** Decision trace entry for the trace panel. */
export interface DecisionTraceEntry {
  tick: number;
  agentId: number;
  action: number;
  confidence: number;
  searchDepth: number;
  ucb1Score: number;
  intentLabel: string;
}

/**
 * A persisted training-metrics record as returned by
 * `GET /api/training-metrics/history` (server `TrainingRecord` wire shape).
 */
export interface TrainingHistoryRecord {
  runId: string;
  recordedAtMs: number;
  episode: number;
  totalSteps: number;
  meanReward: number;
  winRate: number;
  curriculumDifficulty: number;
  stepsPerSecond: number;
  lossPolicy: number;
  lossValue: number;
  entropy: number;
}

/**
 * A persisted decision-trace record as returned by
 * `GET /api/decision-traces/history` (server `TraceRecord` wire shape).
 */
export interface TraceHistoryRecord {
  runId: string;
  recordedAtMs: number;
  agentId: number;
  tick: number;
  intentLabel: string;
  confidence: number;
  searchDepth: number;
  ucb1Score: number;
  alternativesConsidered: number;
}

/** Summary of a single run as returned by `GET /api/runs`. */
export interface RunSummary {
  runId: string;
  startedAtMs: number;
  lastSeenMs: number;
  episodes: number;
  latestMeanReward: number;
}

/** Server health response. */
export interface HealthResponse {
  status: string;
  uptimeSeconds: number;
}

/** Server metrics response. */
export interface ServerMetrics {
  simulationTicks: number;
  stepsPerSecond: number;
  wsConnections: number;
  uptimeSeconds: number;
}
