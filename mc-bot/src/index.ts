import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

import { executeAction } from './actions/index.js';
import { BotManager } from './bot_manager.js';
import { loadConfigBundle, type ConfigBundle, type EnvConfig } from './config.js';
import { createLogger } from './logger.js';
import { snapshotObservation, computeObsDim, type Snapshot } from './observation.js';
import { gridShapePayload } from './observation_grid.js';
import { errorMsg, helloMsg, observationMsg, parseClientMsg } from './protocol.js';
import { applyReset } from './reset.js';
import { startViewer } from './viewer.js';

const WS_OPEN = 1;

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
    if (activeSocket && activeSocket.readyState === WS_OPEN) {
      sendJson(socket, errorMsg('BUSY', 'another client is already connected'));
      socket.close();
      return;
    }
    activeSocket = socket;
    const gridShape = gridShapePayload(envConfig.observation);
    if (gridShape !== null) {
      logger.info?.(
        `[mc-bot] block-grid encoder enabled: ` +
          `h=${gridShape.height} w=${gridShape.width} ` +
          `d=${gridShape.depth} ch=${gridShape.channels} ` +
          `vector_dim=${gridShape.vector_dim} total=${obsDim}`,
      );
    } else {
      logger.info?.(`[mc-bot] flat observation: obs_dim=${obsDim}`);
    }
    sendJson(socket, helloMsg({
      actionCount: bundle.actionMap.actionCount,
      obsDim,
      schemaId: bundle.schemaId,
      gridShape,
    }));

    let queue = Promise.resolve();
    const enqueue = (data: any) => {
      queue = queue.then(() => handleClientMessage(socket, asText(data))).catch((error) => {
        logger.warn?.(`mc-bot protocol handler error: ${error.message}`);
        sendJson(socket, errorMsg('INTERNAL', error.message));
      });
    };

    async function handleClientMessage(clientSocket: any, text: string): Promise<void> {
      let message;
      try {
        message = parseClientMsg(text);
      } catch (error: any) {
        sendJson(clientSocket, errorMsg('BAD_MESSAGE', error.message));
        return;
      }

      if (message.type === 'close') {
        clientSocket.close();
        return;
      }

      // Guard: if the botManager is mid-reconnect, return RECONNECTING
      if (botManager?.isReconnecting()) {
        sendJson(clientSocket, errorMsg('RECONNECTING', 'mineflayer reconnecting, please retry'));
        return;
      }

      const currentBot = resolveBot();
      if (!currentBot) {
        sendJson(clientSocket, errorMsg('RECONNECTING', 'bot unavailable, reconnecting'));
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
          logger.warn?.({ event: 'reset_error', error: err.message });
          if (botManager) {
            botManager.reconnect().catch(() => {});
            resultMsg = errorMsg('RECONNECTING', 'mineflayer reconnecting after reset error');
          } else {
            resultMsg = errorMsg('INTERNAL', err.message);
          }
        }
        try {
          sendJson(clientSocket, resultMsg);
        } catch (sendErr: any) {
          logger.warn?.({ event: 'send_reset_reply_error', error: sendErr.message });
        }
        return;
      }

      if (message.type === 'step') {
        const action = bundle.actionMap.get(message.action_id);
        if (!action) {
          try {
            sendJson(clientSocket, errorMsg('INVALID_ACTION', `unknown action_id ${message.action_id}`));
          } catch (sendErr: any) {
            logger.warn?.({ event: 'send_invalid_action_reply_error', error: sendErr.message });
          }
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
          logger.warn?.({ event: 'step_error', error: err.message });
          if (botManager) {
            botManager.reconnect().catch(() => {});
            resultMsg = errorMsg('RECONNECTING', 'mineflayer reconnecting after step error');
          } else {
            resultMsg = errorMsg('INTERNAL', err.message);
          }
        }
        try {
          sendJson(clientSocket, resultMsg);
        } catch (sendErr: any) {
          logger.warn?.({ event: 'send_step_reply_error', error: sendErr.message });
        }
      }
    }

    socket.on('message', enqueue);
    socket.on('close', () => {
      if (activeSocket === socket) activeSocket = null;
    });
    socket.on('error', (error: any) => {
      logger.warn?.(`mc-bot websocket error: ${error.message}`);
    });
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
  const server = new WebSocketServer({
    host: bundle.env.websocket?.host,
    port: bundle.env.websocket?.port,
  });
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
  logger.info?.(`mc-bot listening at ${bundle.env.ws_url}`);
  return { bot: botManager.getBot(), botManager, server, bundle };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}