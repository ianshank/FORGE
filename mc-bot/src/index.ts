import { timingSafeEqual } from 'node:crypto';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

import { executeAction } from './actions/index.js';
import { BotManager } from './bot_manager.js';
import {
  loadConfigBundle,
  normalizeWebsocketConfig,
  type ConfigBundle,
  type EnvConfig,
  type WebsocketLimits,
} from './config.js';
import { createLogger } from './logger.js';
import { snapshotObservation, computeObsDim, type Snapshot } from './observation.js';
import { gridShapePayload } from './observation_grid.js';
import { errorMsg, helloMsg, observationMsg, parseClientMsg } from './protocol.js';
import { applyReset } from './reset.js';
import { startViewer } from './viewer.js';

const WS_OPEN = 1;

/**
 * Client-facing text for any server-side failure.
 *
 * Raw `Error.message` strings routinely embed absolute filesystem paths,
 * hostnames, and dependency internals. The detail is logged server-side; the
 * peer — which may be unauthenticated — only learns that something failed.
 */
export const GENERIC_INTERNAL_MESSAGE = 'internal server error; see mc-bot logs';

/** Process exit code used by the fatal-error guards. */
export const FATAL_EXIT_CODE = 1;

/**
 * Base used to parse the relative request-target of an HTTP upgrade
 * (`GET /?token=... HTTP/1.1`) with the WHATWG `URL` parser. Never dialled.
 */
const AUTH_URL_BASE = 'ws://mc-bot.invalid';

/** Reduce an unknown thrown value to a loggable `{ error, stack }` pair. */
function describeError(value: any): { error: string; stack?: string } {
  if (value instanceof Error) {
    return { error: value.message, stack: value.stack };
  }
  return { error: String(value) };
}

function sendJson(socket: any, message: any): void {
  socket.send(JSON.stringify(message));
}

function asText(data: any): string {
  return Buffer.isBuffer(data) ? data.toString('utf8') : String(data);
}

function terminalFromSnapshot(snapshot: Snapshot, envConfig: EnvConfig): boolean {
  return Boolean(envConfig.episode.terminate_on_death && snapshot.health <= 0);
}

function actionTicks(action: any, executionResult: any, envConfig: EnvConfig): number {
  if (Number.isInteger(executionResult?.ticks) && executionResult.ticks > 0) {
    return executionResult.ticks;
  }
  if (Number.isInteger(action?.ticks) && action.ticks > 0) {
    return action.ticks;
  }
  return envConfig.episode.action_repeat;
}

export function validateObservationConfig(envConfig: EnvConfig, obsDim: number): void {
  const expectedDim = envConfig.observation.expected_dim;
  if (expectedDim !== null && expectedDim !== undefined && expectedDim !== obsDim) {
    throw new Error(`observation.expected_dim=${expectedDim} does not match computed obs_dim=${obsDim}`);
  }
}

// ---------------------------------------------------------------------------
// Handshake authentication
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Connection handling
// ---------------------------------------------------------------------------

export function createConnectionHandler(options: {
  bot?: any;
  botManager?: BotManager;
  bundle: ConfigBundle;
  logger?: any;
}): (socket: any) => void {
  const { bot, botManager, bundle, logger = createLogger() } = options;
  // Support both legacy `bot` param and new `botManager` param.
  // When botManager is provided, always dereference via getBot() for live reference.
  const resolveBot = botManager ? () => botManager.getBot() : () => bot;
  const envConfig = bundle.env;
  const limits = normalizeWebsocketConfig(envConfig.websocket);
  const obsDim = computeObsDim(envConfig.observation);
  validateObservationConfig(envConfig, obsDim);
  let activeSocket: any = null;
  let episodeTicks = 0;
  let previousSnapshot: Snapshot | null = null;

  // When the botManager emits 'reconnected', reset episode state so the
  // next step/reset uses the fresh bot without stale cached data.
  if (botManager) {
    botManager.on('reconnected', () => {
      logger.info?.({ event: 'handler_reconnect_reset', msg: 'resetting episode state after reconnect' });
      episodeTicks = 0;
      previousSnapshot = null;
    });
  }

  return function handleConnection(socket: any) {
    // Register the error listener BEFORE the first send. `ws` reports a send on
    // a non-OPEN socket by emitting 'error', and an 'error' emit with no
    // listener throws out of the EventEmitter — which previously took down the
    // process on the two sends below (BUSY reply and hello handshake).
    socket.on('error', (error: any) => {
      logger.warn?.({ event: 'websocket_error', ...describeError(error) });
    });

    /** Send, swallowing (and logging) transport failures. */
    const safeSend = (message: any, event: string): void => {
      try {
        sendJson(socket, message);
      } catch (sendErr: any) {
        logger.warn?.({ event, ...describeError(sendErr) });
      }
    };

    if (activeSocket && activeSocket.readyState === WS_OPEN) {
      safeSend(errorMsg('BUSY', 'another client is already connected'), 'send_busy_error');
      socket.close();
      return;
    }
    activeSocket = socket;

    // --- keepalive / idle reclaim -----------------------------------------
    // `lastActivityMs` tracks *application* traffic only. Pongs deliberately do
    // not refresh it: a live-but-silent client answers pings forever, and it is
    // exactly that client which must not hold the single bot slot indefinitely.
    let lastActivityMs = Date.now();
    // Timestamp of the OLDEST still-unanswered ping, or null when the peer is
    // up to date. Deliberately not a boolean: the production client is a
    // *synchronous* Rust client (crates/forge-env-mc/src/client.rs) that reads
    // the socket only inside recv(), and tungstenite queues a pong only when
    // the ping frame is actually read -- there is no background pong thread.
    // Between episodes the runner writes the trajectory, re-hashes the whole
    // ONNX bundle (integrity::verify_bundle) and rebuilds three ORT sessions
    // without touching the socket. A boolean here reclaimed after a single
    // missed ping, i.e. one ping_interval_ms, killing a perfectly healthy
    // client mid-hot-reload. The deadline is now ping_timeout_ms.
    let unansweredPingSinceMs: number | null = null;
    let keepaliveTimer: ReturnType<typeof setInterval> | null = null;

    const stopKeepalive = (): void => {
      if (keepaliveTimer !== null) {
        clearInterval(keepaliveTimer);
        keepaliveTimer = null;
      }
    };
    const releaseSlot = (): void => {
      stopKeepalive();
      if (activeSocket === socket) activeSocket = null;
    };
    const reclaim = (reason: string, detail: Record<string, unknown>): void => {
      logger.warn?.({
        event: 'websocket_session_reclaimed',
        reason,
        ...detail,
        msg: 'reclaiming the control channel from an unresponsive client',
      });
      releaseSlot();
      try {
        if (typeof socket.terminate === 'function') socket.terminate();
        else socket.close();
      } catch (err: any) {
        logger.warn?.({ event: 'session_reclaim_error', ...describeError(err) });
      }
    };

    const keepaliveTick = (): void => {
      const idleMs = Date.now() - lastActivityMs;
      if (limits.idle_timeout_ms > 0 && idleMs >= limits.idle_timeout_ms) {
        reclaim('idle_timeout', { idle_ms: idleMs, idle_timeout_ms: limits.idle_timeout_ms });
        return;
      }
      if (limits.ping_interval_ms > 0 && typeof socket.ping === 'function') {
        if (unansweredPingSinceMs !== null) {
          const unansweredMs = Date.now() - unansweredPingSinceMs;
          if (limits.ping_timeout_ms > 0 && unansweredMs >= limits.ping_timeout_ms) {
            reclaim('ping_timeout', {
              unanswered_ms: unansweredMs,
              ping_timeout_ms: limits.ping_timeout_ms,
              ping_interval_ms: limits.ping_interval_ms,
            });
          }
          // Otherwise keep probing: a busy-but-alive client will drain the
          // queued pings the next time it reads the socket.
          return;
        }
        unansweredPingSinceMs = Date.now();
        try {
          socket.ping();
        } catch (err: any) {
          logger.warn?.({ event: 'websocket_ping_error', ...describeError(err) });
        }
      }
    };

    const tickMs = limits.ping_interval_ms > 0 ? limits.ping_interval_ms : limits.idle_timeout_ms;
    if (tickMs > 0) {
      keepaliveTimer = setInterval(keepaliveTick, tickMs);
      // Never keep the process alive purely for a keepalive probe.
      keepaliveTimer.unref?.();
    }
    socket.on('pong', () => {
      // The peer is reachable again. Note this still does NOT refresh
      // lastActivityMs -- a live-but-silent client must not hold the slot
      // forever; that is what idle_timeout_ms is for.
      unansweredPingSinceMs = null;
    });

    const gridShape = gridShapePayload(envConfig.observation);
    if (gridShape !== null) {
      logger.info?.(
        `[mc-bot] block-grid encoder enabled: h=${gridShape.height} w=${gridShape.width} d=${gridShape.depth} ch=${gridShape.channels} vector_dim=${gridShape.vector_dim} total=${obsDim}`,
      );
    } else {
      logger.info?.(`[mc-bot] flat observation: obs_dim=${obsDim}`);
    }
    safeSend(
      helloMsg({
        actionCount: bundle.actionMap.actionCount,
        obsDim,
        schemaId: bundle.schemaId,
        gridShape,
      }),
      'send_hello_error',
    );

    // --- bounded inbound queue --------------------------------------------
    // `step` awaits real game ticks, so a client can enqueue far faster than
    // the chain drains. Depth is capped; breaching it is answered with a
    // protocol error and a close rather than unbounded buffering.
    let queue: Promise<void> = Promise.resolve();
    let queueDepth = 0;
    let overflowed = false;

    const enqueue = (data: any) => {
      lastActivityMs = Date.now();
      if (overflowed) return;
      if (queueDepth >= limits.max_queue_depth) {
        overflowed = true;
        logger.warn?.({
          event: 'client_queue_overflow',
          queue_depth: queueDepth,
          max_queue_depth: limits.max_queue_depth,
          msg: 'client exceeded the in-flight message cap; closing the control channel',
        });
        safeSend(
          errorMsg('BACKPRESSURE', `message queue depth exceeded (max ${limits.max_queue_depth})`),
          'send_backpressure_error',
        );
        releaseSlot();
        socket.close();
        return;
      }
      queueDepth += 1;
      queue = queue
        .then(() => (overflowed ? undefined : handleClientMessage(socket, asText(data))))
        .catch((error: any) => {
          logger.warn?.({ event: 'protocol_handler_error', ...describeError(error) });
          safeSend(errorMsg('INTERNAL', GENERIC_INTERNAL_MESSAGE), 'send_internal_error');
        })
        .finally(() => {
          queueDepth -= 1;
        });
    };

    async function handleClientMessage(clientSocket: any, text: string): Promise<void> {
      let message: ReturnType<typeof parseClientMsg>;
      try {
        message = parseClientMsg(text);
      } catch (error: any) {
        // Parse failures echo only the client's own malformed input.
        safeSend(errorMsg('BAD_MESSAGE', error.message), 'send_bad_message_error');
        return;
      }

      if (message.type === 'close') {
        clientSocket.close();
        return;
      }

      // Guard: if the botManager is mid-reconnect, return RECONNECTING
      if (botManager?.isReconnecting()) {
        safeSend(errorMsg('RECONNECTING', 'mineflayer reconnecting, please retry'), 'send_reconnecting_error');
        return;
      }

      const currentBot = resolveBot();
      if (!currentBot) {
        safeSend(errorMsg('RECONNECTING', 'bot unavailable, reconnecting'), 'send_reconnecting_error');
        return;
      }

      if (message.type === 'reset') {
        let resultMsg: any;
        try {
          await applyReset(currentBot, bundle.reset);
          episodeTicks = 0;
          previousSnapshot = snapshotObservation(currentBot, envConfig.observation);
          if (botManager) botManager.updateTickAge(currentBot.time?.age ?? 0);
          resultMsg = observationMsg({
            tick: previousSnapshot.tick,
            obs: previousSnapshot.obs!,
            reward: 0,
            terminated: terminalFromSnapshot(previousSnapshot, envConfig),
            truncated: false,
            info: { event: 'reset', seed: message.seed ?? null },
          });
        } catch (err: any) {
          logger.warn?.({ event: 'reset_error', ...describeError(err) });
          if (botManager) {
            botManager.reconnect().catch((reconnectErr: any) => {
              logger.error?.({ event: 'reconnect_failed', origin: 'reset', ...describeError(reconnectErr) });
            });
            resultMsg = errorMsg('RECONNECTING', 'mineflayer reconnecting after reset error');
          } else {
            resultMsg = errorMsg('INTERNAL', GENERIC_INTERNAL_MESSAGE);
          }
        }
        safeSend(resultMsg, 'send_reset_reply_error');
        return;
      }

      if (message.type === 'step') {
        const action = bundle.actionMap.get(message.action_id);
        if (!action) {
          safeSend(
            errorMsg('INVALID_ACTION', `unknown action_id ${message.action_id}`),
            'send_invalid_action_reply_error',
          );
          return;
        }
        let resultMsg: any;
        try {
          const before = previousSnapshot ?? snapshotObservation(currentBot, envConfig.observation);
          const executionResult = await executeAction(currentBot, action, {
            defaultTicks: envConfig.episode.action_repeat,
          });
          const after = snapshotObservation(currentBot, envConfig.observation);
          if (botManager) botManager.updateTickAge(currentBot.time?.age ?? 0);
          const rewardCtx = { prev: before, curr: after, action, breakdown: {} };
          const reward = bundle.rewardFn(rewardCtx);
          episodeTicks += actionTicks(action, executionResult, envConfig);
          previousSnapshot = after;
          resultMsg = observationMsg({
            tick: after.tick,
            obs: after.obs!,
            reward,
            terminated: terminalFromSnapshot(after, envConfig),
            truncated: episodeTicks >= envConfig.episode.max_ticks,
            info: {
              action_id: message.action_id,
              action_kind: action.kind,
              episode_ticks: episodeTicks,
              reward_breakdown: rewardCtx.breakdown,
            },
          });
        } catch (err: any) {
          logger.warn?.({ event: 'step_error', ...describeError(err) });
          if (botManager) {
            botManager.reconnect().catch((reconnectErr: any) => {
              logger.error?.({ event: 'reconnect_failed', origin: 'step', ...describeError(reconnectErr) });
            });
            resultMsg = errorMsg('RECONNECTING', 'mineflayer reconnecting after step error');
          } else {
            resultMsg = errorMsg('INTERNAL', GENERIC_INTERNAL_MESSAGE);
          }
        }
        safeSend(resultMsg, 'send_step_reply_error');
      }
    }

    socket.on('message', enqueue);
    socket.on('close', releaseSlot);
  };
}

export async function startProtocolServer(options: {
  bot?: any;
  botManager?: BotManager;
  bundle: ConfigBundle;
  WebSocketServer: any;
  logger?: any;
}): Promise<any> {
  const { bot, botManager, bundle, WebSocketServer, logger = createLogger() } = options;
  const limits = normalizeWebsocketConfig(bundle.env.websocket);
  // NOT `websocket.host` -- that is the client's dial target. Binding to it
  // made the server listen on the container's own eth0 address under docker
  // (ws_url = "ws://mc-bot:8765"), so the healthcheck's connection to
  // 127.0.0.1 was refused and `runner` never cleared `service_healthy`.
  // See resolveBindHost() for the rule; `undefined` = every interface.
  const bindHost = bundle.env.websocket?.bind_host;
  const serverOptions: Record<string, any> = {
    ...(bindHost === undefined ? {} : { host: bindHost }),
    port: bundle.env.websocket?.port,
    // Replaces the `ws` default of 100 MiB per frame. See
    // DEFAULT_WEBSOCKET_LIMITS for the sizing rationale.
    maxPayload: limits.max_payload_bytes,
  };
  if (limits.auth_token) {
    serverOptions.verifyClient = createVerifyClient({ limits, logger });
    logger.info?.({
      event: 'websocket_auth_enabled',
      auth_query_param: limits.auth_query_param,
      msg: `control channel requires a shared secret via 'Authorization: Bearer <token>' or ?${limits.auth_query_param}=<token>`,
    });
  } else {
    logger.warn?.({
      event: 'websocket_auth_disabled',
      bind_host: bindHost ?? '0.0.0.0 (all interfaces)',
      msg: `mc-bot control channel is UNAUTHENTICATED: any process that can reach ${bindHost ?? 'any interface'}:${bundle.env.websocket?.port} can drive the bot. Set [websocket] auth_token in configs/minecraft/env.toml to require a shared secret.`,
    });
  }
  const server = new WebSocketServer(serverOptions);
  server.on('connection', createConnectionHandler({ bot, botManager, bundle, logger }));
  await new Promise((resolveListen, rejectListen) => {
    server.once('listening', resolveListen);
    server.once('error', rejectListen);
  });
  return server;
}

async function waitForSpawn(bot: any): Promise<void> {
  if (bot.entity) return;
  await new Promise<void>((resolveSpawn) => {
    bot.once('spawn', resolveSpawn);
  });
}

/**
 * Install process-level guards for errors that escape every `try`/`catch`.
 *
 * Both handlers log the full context (message + stack) through the injected
 * logger and then exit non-zero. Nothing is swallowed: a bot left running after
 * an unhandled rejection holds the single client slot while being unable to
 * serve it, and Node's default `unhandledRejection` behaviour (terminate) gives
 * no structured record of the cause.
 *
 * Not installed on import — call it explicitly at the entry point (per the
 * project's no-side-effects-on-import rule).
 */
export function installProcessGuards(options: {
  logger?: any;
  processRef?: any;
  exit?: (code: number) => void;
} = {}): void {
  const logger = options.logger ?? createLogger();
  const proc = options.processRef ?? process;
  const exit = options.exit ?? ((code: number) => proc.exit(code));

  proc.on('unhandledRejection', (reason: any) => {
    logger.error?.({
      event: 'unhandled_rejection',
      ...describeError(reason),
      msg: 'unhandled promise rejection; exiting',
    });
    exit(FATAL_EXIT_CODE);
  });

  proc.on('uncaughtException', (error: any, origin?: any) => {
    logger.error?.({
      event: 'uncaught_exception',
      origin: origin ?? 'uncaughtException',
      ...describeError(error),
      msg: 'uncaught exception; exiting',
    });
    exit(FATAL_EXIT_CODE);
  });
}

export async function main(options: any = {}): Promise<any> {
  const bundle = await loadConfigBundle(options);
  const mineflayerModule = options.mineflayerModule ?? await import('mineflayer');
  const wsModule = options.wsModule ?? await import('ws');
  const createBotFn = mineflayerModule.createBot ?? mineflayerModule.default?.createBot;
  const WebSocketServer = wsModule.WebSocketServer ?? wsModule.default?.WebSocketServer;
  if (typeof createBotFn !== 'function') {
    throw new Error('mineflayer module does not expose createBot');
  }
  if (typeof WebSocketServer !== 'function') {
    throw new Error('ws module does not expose WebSocketServer');
  }

  const logger = options.logger ?? createLogger();

  // When an explicit bot stub is provided (e.g. tests), skip BotManager.
  if (options.bot) {
    const bot = options.bot;
    await waitForSpawn(bot);
    await startViewer(bot, bundle.env.viewer, options);
    const server = await startProtocolServer({ bot, bundle, WebSocketServer, logger });
    logger.info?.(`mc-bot listening at ${bundle.env.ws_url}`);
    return { bot, server, bundle };
  }

  const req = createRequire(import.meta.url);
  const pvpPlugin = options.pvpPlugin ?? req('mineflayer-pvp').plugin;

  const pluginWrappedCreateBot = (config: any) => {
    const b = createBotFn(config);
    if (pvpPlugin) {
      b.loadPlugin(pvpPlugin);
    }
    return b;
  };

  // Production path: use BotManager for lifecycle + auto-reconnect.
  const botManager = new BotManager(bundle.env.bot, pluginWrappedCreateBot, {
    reconnectConfig: bundle.env.reconnect,
    logger,
  });
  const bot = await botManager.createInitialBot();
  await startViewer(bot, bundle.env.viewer, options);
  const server = await startProtocolServer({ botManager, bundle, WebSocketServer, logger });
  // Background stale-detection: reconnect if the server stops ticking while the
  // socket stays half-open. Interval is the config-driven env.heartbeat_ms.
  botManager.startHeartbeat(bundle.env.heartbeat_ms);
  logger.info?.(`mc-bot listening at ${bundle.env.ws_url}`);
  return { bot: botManager.getBot(), botManager, server, bundle };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const bootLogger = createLogger();
  installProcessGuards({ logger: bootLogger });
  main({ logger: bootLogger }).catch((error) => {
    bootLogger.error?.({ event: 'startup_failed', ...describeError(error) });
    process.exitCode = FATAL_EXIT_CODE;
  });
}
