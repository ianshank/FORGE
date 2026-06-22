/**
 * Structured logging factory for the mc-bot bridge.
 *
 * Call sites already use a `console`-shaped logger and pass either a string
 * message or a structured `{ event, ... }` object. `createLogger()` returns a
 * logger honouring that same shape but, when `FORGE_LOG_FORMAT=json`, emits one
 * JSON object per line (level + timestamp + payload) suited to log aggregation.
 * In `text` mode (the default) it delegates to `console` so existing output is
 * unchanged — backwards compatible.
 *
 * Mirrors the Rust `forge-observability` crate and the Python
 * `forge.utils.logging_config` helper: the same `FORGE_LOG_FORMAT` switch flips
 * structured logging across the whole stack. No static side-effects — the
 * factory is invoked explicitly at the wiring seam (per CLAUDE.md ESM rules).
 */

/** Environment variable selecting the log output format (`text` | `json`). */
export const LOG_FORMAT_ENV = 'FORGE_LOG_FORMAT';

/** Payload accepted by a log method: a plain string or a structured object. */
export type LogPayload = string | Record<string, unknown>;

/** Minimal logger surface used across the bridge. */
export interface Logger {
  info(payload?: LogPayload): void;
  warn(payload?: LogPayload): void;
  error(payload?: LogPayload): void;
  debug(payload?: LogPayload): void;
}

type Level = 'info' | 'warn' | 'error' | 'debug';

/** Resolve whether JSON output is requested from the environment. */
export function jsonFormatFromEnv(env: NodeJS.ProcessEnv = process.env): boolean {
  const raw = env[LOG_FORMAT_ENV];
  if (raw === undefined) {
    return false;
  }
  return raw.trim().toLowerCase() === 'json';
}

function emitJson(level: Level, payload: LogPayload | undefined, sink: (line: string) => void): void {
  const entry: Record<string, unknown> = {
    timestamp: new Date().toISOString(),
    level,
  };
  if (typeof payload === 'string') {
    entry.message = payload;
  } else if (payload && typeof payload === 'object') {
    Object.assign(entry, payload);
  }
  sink(JSON.stringify(entry));
}

/**
 * Build a {@link Logger}. When `json` is true (or `FORGE_LOG_FORMAT=json`),
 * emits structured JSON lines; otherwise delegates to the provided `console`.
 *
 * @param options.json Force JSON output, overriding the env var.
 * @param options.env Environment map (injectable for tests).
 * @param options.console Console-like sink (injectable for tests).
 */
export function createLogger(options?: {
  json?: boolean;
  env?: NodeJS.ProcessEnv;
  console?: Pick<Console, 'info' | 'warn' | 'error' | 'debug' | 'log'>;
}): Logger {
  const env = options?.env ?? process.env;
  const useJson = options?.json ?? jsonFormatFromEnv(env);
  const sinkConsole = options?.console ?? console;

  if (!useJson) {
    // Text mode: delegate to console, preserving prior behaviour exactly.
    return {
      info: (p) => sinkConsole.info(p),
      warn: (p) => sinkConsole.warn(p),
      error: (p) => sinkConsole.error(p),
      debug: (p) => (sinkConsole.debug ?? sinkConsole.log)(p),
    };
  }

  // JSON mode: stdout for info/debug, stderr for warn/error.
  const out = (line: string) => sinkConsole.log(line);
  const err = (line: string) => sinkConsole.error(line);
  return {
    info: (p) => emitJson('info', p, out),
    warn: (p) => emitJson('warn', p, err),
    error: (p) => emitJson('error', p, err),
    debug: (p) => emitJson('debug', p, out),
  };
}
