/**
 * Structured logging utility for the FORGE dashboard.
 *
 * Provides a lightweight logger that can be configured via environment
 * variables and produces consistent, filterable log output.
 */

export type LogLevel = "debug" | "info" | "warn" | "error";

const LOG_LEVEL_PRIORITY: Record<LogLevel, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
};

/** Current minimum log level (configurable via VITE_LOG_LEVEL). */
const CURRENT_LEVEL: LogLevel = (() => {
  const env =
    typeof import.meta !== "undefined"
      ? (import.meta.env?.VITE_LOG_LEVEL as string | undefined)
      : undefined;
  const level = (env ?? "info").toLowerCase();
  return level in LOG_LEVEL_PRIORITY ? (level as LogLevel) : "info";
})();

/**
 * Creates a named logger instance for a specific module.
 *
 * @param name - Module or component name (e.g. "useWebSocket", "SimulationCanvas")
 */
export function createLogger(name: string) {
  const shouldLog = (level: LogLevel): boolean =>
    LOG_LEVEL_PRIORITY[level] >= LOG_LEVEL_PRIORITY[CURRENT_LEVEL];

  return {
    debug: (msg: string, ...args: unknown[]) => {
      if (shouldLog("debug")) console.debug(`[${name}]`, msg, ...args);
    },
    info: (msg: string, ...args: unknown[]) => {
      if (shouldLog("info")) console.info(`[${name}]`, msg, ...args);
    },
    warn: (msg: string, ...args: unknown[]) => {
      if (shouldLog("warn")) console.warn(`[${name}]`, msg, ...args);
    },
    error: (msg: string, ...args: unknown[]) => {
      if (shouldLog("error")) console.error(`[${name}]`, msg, ...args);
    },
  };
}
