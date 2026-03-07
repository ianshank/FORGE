/** Runtime environment configuration for the dashboard. */

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
}

/** Default configuration — overridable via environment variables. */
export const DEFAULT_CONFIG: DashboardConfig = {
  wsUrl: import.meta.env.VITE_WS_URL ?? "ws://localhost:8080/ws",
  apiBaseUrl: import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8080",
  metricsPollingInterval: Number(
    import.meta.env.VITE_METRICS_INTERVAL ?? 2000,
  ),
  maxTraceEntries: Number(import.meta.env.VITE_MAX_TRACES ?? 500),
  showGridLines: import.meta.env.VITE_SHOW_GRID !== "false",
  cellSize: Number(import.meta.env.VITE_CELL_SIZE ?? 8),
};

/** Returns the current dashboard config (reads env once). */
export function getConfig(): DashboardConfig {
  return { ...DEFAULT_CONFIG };
}
