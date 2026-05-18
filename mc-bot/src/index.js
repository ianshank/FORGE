import { fileURLToPath } from 'node:url';

import { executeAction } from './actions.js';
import { loadConfigBundle } from './config.js';
import { snapshotObservation, computeObsDim } from './observation.js';
import { errorMsg, helloMsg, observationMsg, parseClientMsg } from './protocol.js';
import { applyReset } from './reset.js';
import { startViewer } from './viewer.js';

const WS_OPEN = 1;

function sendJson(socket, message) {
  socket.send(JSON.stringify(message));
}

function asText(data) {
  return Buffer.isBuffer(data) ? data.toString('utf8') : String(data);
}

function terminalFromSnapshot(snapshot, envConfig) {
  return Boolean(envConfig.episode.terminate_on_death && snapshot.health <= 0);
}

function actionTicks(action, executionResult, envConfig) {
  if (Number.isInteger(executionResult?.ticks) && executionResult.ticks > 0) {
    return executionResult.ticks;
  }
  if (Number.isInteger(action?.ticks) && action.ticks > 0) {
    return action.ticks;
  }
  return envConfig.episode.action_repeat;
}

export function validateObservationConfig(envConfig, obsDim) {
  const expectedDim = envConfig.observation.expected_dim;
  if (expectedDim !== null && expectedDim !== undefined && expectedDim !== obsDim) {
    throw new Error(`observation.expected_dim=${expectedDim} does not match computed obs_dim=${obsDim}`);
  }
}

export function createConnectionHandler({ bot, bundle, logger = console }) {
  const envConfig = bundle.env;
  const obsDim = computeObsDim(envConfig.observation);
  validateObservationConfig(envConfig, obsDim);
  let activeSocket = null;
  let episodeTicks = 0;
  let previousSnapshot = null;

  return function handleConnection(socket) {
    if (activeSocket && activeSocket.readyState === WS_OPEN) {
      sendJson(socket, errorMsg('BUSY', 'another client is already connected'));
      socket.close();
      return;
    }
    activeSocket = socket;
    sendJson(socket, helloMsg({
      actionCount: bundle.actionMap.actionCount,
      obsDim,
      schemaId: bundle.schemaId,
    }));

    let queue = Promise.resolve();
    const enqueue = (data) => {
      queue = queue.then(() => handleClientMessage(socket, asText(data))).catch((error) => {
        logger.warn?.(`mc-bot protocol handler error: ${error.message}`);
        sendJson(socket, errorMsg('INTERNAL', error.message));
      });
    };

    async function handleClientMessage(clientSocket, text) {
      let message;
      try {
        message = parseClientMsg(text);
      } catch (error) {
        sendJson(clientSocket, errorMsg('BAD_MESSAGE', error.message));
        return;
      }

      if (message.type === 'close') {
        clientSocket.close();
        return;
      }

      if (message.type === 'reset') {
        await applyReset(bot, bundle.reset);
        episodeTicks = 0;
        previousSnapshot = snapshotObservation(bot, envConfig.observation);
        sendJson(clientSocket, observationMsg({
          tick: previousSnapshot.tick,
          obs: previousSnapshot.obs,
          reward: 0,
          terminated: terminalFromSnapshot(previousSnapshot, envConfig),
          truncated: false,
          info: { event: 'reset', seed: message.seed },
        }));
        return;
      }

      if (message.type === 'step') {
        const action = bundle.actionMap.get(message.action_id);
        if (!action) {
          sendJson(clientSocket, errorMsg('INVALID_ACTION', `unknown action_id ${message.action_id}`));
          return;
        }
        const before = previousSnapshot ?? snapshotObservation(bot, envConfig.observation);
        const executionResult = await executeAction(bot, action, {
          defaultTicks: envConfig.episode.action_repeat,
        });
        const after = snapshotObservation(bot, envConfig.observation);
        const reward = bundle.rewardFn({ prev: before, curr: after, action });
        episodeTicks += actionTicks(action, executionResult, envConfig);
        previousSnapshot = after;
        sendJson(clientSocket, observationMsg({
          tick: after.tick,
          obs: after.obs,
          reward,
          terminated: terminalFromSnapshot(after, envConfig),
          truncated: episodeTicks >= envConfig.episode.max_ticks,
          info: {
            action_id: message.action_id,
            action_kind: action.kind,
            episode_ticks: episodeTicks,
          },
        }));
      }
    }

    socket.on('message', enqueue);
    socket.on('close', () => {
      if (activeSocket === socket) activeSocket = null;
    });
    socket.on('error', (error) => {
      logger.warn?.(`mc-bot websocket error: ${error.message}`);
    });
  };
}

export async function startProtocolServer({ bot, bundle, WebSocketServer, logger = console }) {
  const server = new WebSocketServer({
    host: bundle.env.websocket.host,
    port: bundle.env.websocket.port,
  });
  server.on('connection', createConnectionHandler({ bot, bundle, logger }));
  await new Promise((resolveListen, rejectListen) => {
    server.once('listening', resolveListen);
    server.once('error', rejectListen);
  });
  return server;
}

async function waitForSpawn(bot) {
  if (bot.entity) return;
  await new Promise((resolveSpawn) => {
    bot.once('spawn', resolveSpawn);
  });
}

export async function main(options = {}) {
  const bundle = await loadConfigBundle(options);
  const mineflayerModule = options.mineflayerModule ?? await import('mineflayer');
  const wsModule = options.wsModule ?? await import('ws');
  const createBot = mineflayerModule.createBot ?? mineflayerModule.default?.createBot;
  const WebSocketServer = wsModule.WebSocketServer ?? wsModule.default?.WebSocketServer;
  if (typeof createBot !== 'function') {
    throw new Error('mineflayer module does not expose createBot');
  }
  if (typeof WebSocketServer !== 'function') {
    throw new Error('ws module does not expose WebSocketServer');
  }

  const bot = options.bot ?? createBot(bundle.env.bot);
  await waitForSpawn(bot);
  await startViewer(bot, bundle.env.viewer, options);
  const server = await startProtocolServer({ bot, bundle, WebSocketServer, logger: options.logger ?? console });
  options.logger?.info?.(`mc-bot listening at ${bundle.env.ws_url}`);
  return { bot, server, bundle };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}