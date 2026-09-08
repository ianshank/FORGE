import { executeAction } from './actions/index.js';
import type { BotManager } from './bot_manager.js';
import {
  normalizeWebsocketConfig,
  type ConfigBundle,
  type EnvConfig,
} from './config.js';
import { createLogger } from './logger.js';
import { snapshotObservation, computeObsDim, type Snapshot } from './observation.js';
import { gridShapePayload } from './observation_grid.js';
import {
  ERROR_CODE_BAD_MESSAGE,
  ERROR_CODE_BUSY,
  ERROR_CODE_INTERNAL,
  ERROR_CODE_INVALID_ACTION,
  ERROR_CODE_RECONNECTING,
  errorMsg,
  helloMsg,
  observationMsg,
  parseClientMsg,
} from './protocol.js';
import { applyReset } from './reset.js';

export const WS_OPEN = 1;

/**
 * Client-facing text for any server-side failure.
 *
 * Raw `Error.message` strings routinely embed absolute filesystem paths,
 * hostnames, and dependency internals. The detail is logged server-side; the
 * peer — which may be unauthenticated — only learns that something failed.
 */
export const GENERIC_INTERNAL_MESSAGE = 'internal server error; see mc-bot logs';

/** Reduce an unknown thrown value to a loggable `{ error, stack }` pair. */
export function describeError(value: any): { error: string; stack?: string } {
  if (value instanceof Error) {
    return { error: value.message, stack: value.stack };
  }
  return { error: String(value) };
}

export function sendJson(socket: any, message: any): void {
  socket.send(JSON.stringify(message));
}

export function asText(data: any): string {
  return Buffer.isBuffer(data) ? data.toString('utf8') : String(data);
}

export function terminalFromSnapshot(snapshot: Snapshot, envConfig: EnvConfig): boolean {
  return Boolean(envConfig.episode.terminate_on_death && snapshot.health <= 0);
}

export function actionTicks(action: any, executionResult: any, envConfig: EnvConfig): number {
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
      safeSend(errorMsg(ERROR_CODE_BUSY, 'another client is already connected'), 'send_busy_error');
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
          safeSend(errorMsg(ERROR_CODE_INTERNAL, GENERIC_INTERNAL_MESSAGE), 'send_internal_error');
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
        safeSend(errorMsg(ERROR_CODE_BAD_MESSAGE, error.message), 'send_bad_message_error');
        return;
      }

      if (message.type === 'close') {
        clientSocket.close();
        return;
      }

      // Guard: if the botManager is mid-reconnect, return RECONNECTING
      if (botManager?.isReconnecting()) {
        safeSend(errorMsg(ERROR_CODE_RECONNECTING, 'mineflayer reconnecting; discard this episode'), 'send_reconnecting_error');
        return;
      }

      const currentBot = resolveBot();
      if (!currentBot) {
        safeSend(errorMsg(ERROR_CODE_RECONNECTING, 'bot unavailable, reconnecting'), 'send_reconnecting_error');
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
            resultMsg = errorMsg(ERROR_CODE_RECONNECTING, 'mineflayer reconnecting after reset error');
          } else {
            resultMsg = errorMsg(ERROR_CODE_INTERNAL, GENERIC_INTERNAL_MESSAGE);
          }
        }
        safeSend(resultMsg, 'send_reset_reply_error');
        return;
      }

      if (message.type === 'step') {
        const action = bundle.actionMap.get(message.action_id);
        if (!action) {
          safeSend(
            errorMsg(ERROR_CODE_INVALID_ACTION, `unknown action_id ${message.action_id}`),
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
            resultMsg = errorMsg(ERROR_CODE_RECONNECTING, 'mineflayer reconnecting after step error');
          } else {
            resultMsg = errorMsg(ERROR_CODE_INTERNAL, GENERIC_INTERNAL_MESSAGE);
          }
        }
        safeSend(resultMsg, 'send_step_reply_error');
      }
    }

    socket.on('message', enqueue);
    socket.on('close', releaseSlot);
  };
}
