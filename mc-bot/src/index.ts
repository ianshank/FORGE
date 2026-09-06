import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

import { BotManager } from './bot_manager.js';
import { loadConfigBundle } from './config.js';
import { describeError } from './connection.js';
import { createLogger } from './logger.js';
import { startProtocolServer } from './server.js';
import { startViewer } from './viewer.js';

export {
  extractAuthToken,
  timingSafeEquals,
  authorizeRequest,
  createVerifyClient,
} from './auth.js';

export {
  WS_OPEN,
  GENERIC_INTERNAL_MESSAGE,
  describeError,
  sendJson,
  asText,
  terminalFromSnapshot,
  actionTicks,
  validateObservationConfig,
  createConnectionHandler,
} from './connection.js';

export { startProtocolServer } from './server.js';

/** Process exit code used by the fatal-error guards. */
export const FATAL_EXIT_CODE = 1;

async function waitForSpawn(bot: any): Promise<void> {
  if (bot.entity) return;
  await new Promise<void>((resolveSpawn) => {
    bot.once('spawn', resolveSpawn);
  });
}

/**
 * Install process-level guards for errors that escape every `try`/`catch`.
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
  const mineflayerModule = options.mineflayerModule ?? (await import('mineflayer'));
  const wsModule = options.wsModule ?? (await import('ws'));
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
