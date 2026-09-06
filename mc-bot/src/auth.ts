import { timingSafeEqual } from 'node:crypto';

import type { WebsocketLimits } from './config.js';
import { createLogger } from './logger.js';

/**
 * Base used to parse the relative request-target of an HTTP upgrade
 * (`GET /?token=... HTTP/1.1`) with the WHATWG `URL` parser. Never dialled.
 */
const AUTH_URL_BASE = 'ws://mc-bot.invalid';

/**
 * Pull the presented credential off an HTTP upgrade request.
 *
 * Two equivalent transports are accepted, in this order:
 *  1. `Authorization: Bearer <token>` — preferred; keeps the secret out of
 *     access logs and process listings.
 *  2. `?<auth_query_param>=<token>` on the request target — for clients (such
 *     as the Rust `forge-env-mc` client) that only configure a URL.
 *
 * @returns the presented token, or `null` when the request carries none.
 */
export function extractAuthToken(req: any, queryParam: string): string | null {
  const header = req?.headers?.authorization;
  if (typeof header === 'string') {
    const match = /^bearer\s+(\S+)$/i.exec(header.trim());
    if (match) return match[1];
  }
  const rawUrl = typeof req?.url === 'string' ? req.url : '';
  if (rawUrl.length > 0) {
    try {
      const value = new URL(rawUrl, AUTH_URL_BASE).searchParams.get(queryParam);
      if (typeof value === 'string' && value.length > 0) return value;
    } catch {
      return null;
    }
  }
  return null;
}

/** Constant-time string comparison; short-circuits only on length mismatch. */
export function timingSafeEquals(a: string, b: string): boolean {
  const left = Buffer.from(a, 'utf8');
  const right = Buffer.from(b, 'utf8');
  if (left.length !== right.length) return false;
  return timingSafeEqual(left, right);
}

/**
 * Decide whether an upgrade request may open the control channel.
 *
 * Returns `true` unconditionally when no `auth_token` is configured, which is
 * what preserves the historical (unauthenticated) behaviour.
 */
export function authorizeRequest(req: any, limits: Pick<WebsocketLimits, 'auth_token' | 'auth_query_param'>): boolean {
  if (!limits.auth_token) return true;
  const presented = extractAuthToken(req, limits.auth_query_param);
  if (presented === null) return false;
  return timingSafeEquals(presented, limits.auth_token);
}

/**
 * Build a `ws` `verifyClient` callback enforcing {@link authorizeRequest}.
 * A rejected handshake is answered by `ws` with `401 Unauthorized`.
 */
export function createVerifyClient(options: {
  limits: Pick<WebsocketLimits, 'auth_token' | 'auth_query_param'>;
  logger?: any;
}): (info: any) => boolean {
  const { limits, logger = createLogger() } = options;
  return function verifyClient(info: any): boolean {
    const allowed = authorizeRequest(info?.req, limits);
    if (!allowed) {
      logger.warn?.({
        event: 'websocket_auth_rejected',
        remote: info?.req?.socket?.remoteAddress ?? 'unknown',
        msg: 'rejected WebSocket handshake: missing or invalid auth token',
      });
    }
    return allowed;
  };
}
