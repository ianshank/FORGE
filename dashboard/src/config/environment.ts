/**
 * Runtime environment configuration for the dashboard.
 *
 * All values have sensible defaults and can be overridden via Vite
 * environment variables (prefixed with `VITE_`).
 */

export interface DashboardConfig {
  /** WebSocket URL for state updates. */
  wsUrl: string;
  /** Base URL for REST API calls. */
  apiBaseUrl: string;
  /** Polling interval for metrics (ms). */
  metricsPollingInterval: number;
  /** Maximum trace entries to keep in memory. */
  maxTraceEntries: number;
  /** Whether to show grid lines on the canvas. */
  showGridLines: boolean;
  /** Canvas cell size in pixels. */
  cellSize: number;
  /** Maximum WebSocket reconnect attempts before giving up. */
  maxReconnectAttempts: number;
}

/** Minimum allowed polling interval in ms. */
const MIN_POLLING_INTERVAL = 500;
/** Minimum allowed cell size in pixels. */
const MIN_CELL_SIZE = 2;
/** Maximum allowed cell size in pixels. */
const MAX_CELL_SIZE = 64;

/**
 * Clamp a numeric value to the range [min, max].
 */
function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/**
 * Parse an integer from a string with a default fallback.
 */
function parseIntWithDefault(raw: string | undefined, fallback: number): number {
  if (raw === undefined) return fallback;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? Math.floor(parsed) : fallback;
}

/** Default configuration — overridable via environment variables. */
export const DEFAULT_CONFIG: DashboardConfig = {
  wsUrl: import.meta.env.VITE_WS_URL ?? "ws://localhost:8080/ws",
  apiBaseUrl: import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8080",
  metricsPollingInterval: clamp(
    parseIntWithDefault(import.meta.env.VITE_METRICS_INTERVAL, 2000),
    MIN_POLLING_INTERVAL,
    60000,
  ),
  maxTraceEntries: parseIntWithDefault(import.meta.env.VITE_MAX_TRACES, 500),
  showGridLines: import.meta.env.VITE_SHOW_GRID !== "false",
  cellSize: clamp(
    parseIntWithDefault(import.meta.env.VITE_CELL_SIZE, 8),
    MIN_CELL_SIZE,
    MAX_CELL_SIZE,
  ),
  maxReconnectAttempts: clamp(
    parseIntWithDefault(import.meta.env.VITE_MAX_RECONNECT_ATTEMPTS, 10),
    1,
    100,
  ),
};

/** Cached config instance. */
let _cachedConfig: DashboardConfig | null = null;

/** Returns the current dashboard config (reads env once, then caches). */
export function getConfig(): DashboardConfig {
  if (!_cachedConfig) {
    _cachedConfig = { ...DEFAULT_CONFIG };
  }
  return _cachedConfig;
}

/**
 * Reset the cached config (for testing only).
 * @internal
 */
export function _resetConfigCache(): void {
  _cachedConfig = null;
}
