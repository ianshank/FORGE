import { createVerifyClient } from './auth.js';
import type { BotManager } from './bot_manager.js';
import {
  normalizeWebsocketConfig,
  type ConfigBundle,
} from './config.js';
import { createConnectionHandler } from './connection.js';
import { createLogger } from './logger.js';

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
