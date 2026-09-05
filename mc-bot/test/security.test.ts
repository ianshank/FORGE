import { EventEmitter } from 'node:events';
import { createServer } from 'node:net';
import type { AddressInfo } from 'node:net';
import { setTimeout as delay } from 'node:timers/promises';
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import { WebSocket, WebSocketServer } from 'ws';

import { buildActionMap } from '../src/action_map.js';
import { executeAction } from '../src/actions/index.js';
import {
  buildConfigBundleFromObjects,
  deepMerge,
  normalizeEnvConfig,
  normalizeWebsocketConfig,
  DEFAULT_WEBSOCKET_LIMITS,
} from '../src/config.js';
import {
  authorizeRequest,
  createConnectionHandler,
  extractAuthToken,
  installProcessGuards,
  startProtocolServer,
  timingSafeEquals,
  FATAL_EXIT_CODE,
  GENERIC_INTERNAL_MESSAGE,
} from '../src/index.js';
import { parseClientMsg } from '../src/protocol.js';
import { applyReset, validateSelector } from '../src/reset.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Reserve, then release, an ephemeral loopback port. */
async function freePort(): Promise<number> {
  const probe = createServer();
  await new Promise<void>((done) => probe.listen(0, '127.0.0.1', done));
  const { port } = probe.address() as AddressInfo;
  await new Promise<void>((done) => probe.close(() => done()));
  return port;
}

/** Minimal mineflayer stand-in; `gate` (when supplied) stalls every tick wait. */
function stubBot(gate?: Promise<void>) {
  return {
    username: 'ForgeBot',
    chats: [] as string[],
    controls: [] as [string, boolean][],
    time: { age: 0 },
    entity: {
      position: { x: 10, y: 64, z: 0 },
      velocity: { x: 0, y: 0, z: 0 },
      yaw: 0,
      pitch: 0,
    },
    health: 20,
    food: 20,
    oxygenLevel: 20,
    inventory: { slots: Array.from({ length: 45 }, () => null), items: () => [] as any[] },
    chat(command: string) {
      this.chats.push(command);
    },
    setControlState(control: string, state: boolean) {
      this.controls.push([control, state]);
    },
    async waitForTicks(ticks: number) {
      if (gate) await gate;
      this.time.age += ticks;
    },
  };
}

/** `[websocket]` overrides that switch the keepalive timers off entirely. */
const NO_TIMERS = Object.freeze({ idle_timeout_ms: 0, ping_interval_ms: 0 });

/** Poll granularity for {@link waitUntil}. */
const POLL_INTERVAL_MS = 10;

/**
 * Frame-size cap the exact-boundary tests configure. Small enough that a
 * cap that drifted by any factor lands outside the accept/reject pair.
 */
const PAYLOAD_CAP_BYTES = 256;

/** In-flight message cap the exact-boundary backpressure tests configure. */
const QUEUE_CAP = 2;

/**
 * How long gated `step` messages get to reach the server before the test
 * judges the resulting in-flight depth. Only ever weakens the at-the-cap
 * assertion if it is too short — the positive check that follows (every
 * message drains into an observation) is what actually decides the test.
 */
const ENQUEUE_SETTLE_MS = 250;

/** Bundle wired to `port`, with the `[websocket]` hardening table under test. */
function testBundle(port: number, websocket: Record<string, unknown> = {}) {
  const actionMap = buildActionMap({
    action: [
      { id: 0, kind: 'noop', ticks: 1 },
      { id: 1, kind: 'move', direction: 'forward', ticks: 1 },
    ],
  });
  return buildConfigBundleFromObjects({
    env: {
      ws_url: `ws://127.0.0.1:${port}`,
      episode: { max_ticks: 1000, action_repeat: 1 },
      observation: {
        include_velocity: false,
        include_orientation: false,
        include_vitals: false,
        include_inventory: false,
      },
      websocket,
    },
    reset: { teleport: { selector: 'ForgeBot' } },
    actionMapData: actionMap,
    rewardData: { reward: [{ kind: 'survival', value: 0.5 }] },
  });
}

/** Start a hardened protocol server on a fresh port and hand it to `body`. */
async function withServer(
  websocket: Record<string, unknown>,
  body: (ctx: { url: string; bot: ReturnType<typeof stubBot> }) => Promise<void>,
  options: { gate?: Promise<void> } = {},
): Promise<void> {
  const port = await freePort();
  const bot = stubBot(options.gate);
  const server = await startProtocolServer({
    bot,
    bundle: testBundle(port, websocket),
    WebSocketServer,
    logger: { info() {}, warn() {}, error() {}, debug() {} },
  });
  try {
    await body({ url: `ws://127.0.0.1:${port}`, bot });
  } finally {
    for (const client of server.clients) client.terminate();
    await new Promise<void>((done) => server.close(() => done()));
  }
}

/** Await the next JSON message on a client socket. */
function nextMessage(socket: WebSocket, timeoutMs = 4000): Promise<any> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('timed out waiting for a message')), timeoutMs);
    socket.once('message', (raw) => {
      clearTimeout(timer);
      resolve(JSON.parse(raw.toString()));
    });
  });
}

/** Await a client socket's close, resolving with the close code. */
function nextClose(socket: WebSocket, timeoutMs = 4000): Promise<number> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('timed out waiting for close')), timeoutMs);
    socket.once('close', (code) => {
      clearTimeout(timer);
      resolve(code);
    });
  });
}

/** Connect and consume the `hello` handshake. */
async function connectAndHello(url: string, options?: any): Promise<WebSocket> {
  const socket = new WebSocket(url, options);
  const hello = await nextMessage(socket);
  assert.equal(hello.type, 'hello');
  return socket;
}

/**
 * Poll `predicate` until it holds, or fail with `message` after
 * `timeoutMs`. Preferred over a fixed sleep wherever the test can name
 * the condition it is actually waiting for.
 */
async function waitUntil(predicate: () => boolean, message: string, timeoutMs = 4000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!predicate()) {
    if (Date.now() > deadline) throw new Error(`timed out waiting for ${message}`);
    await delay(POLL_INTERVAL_MS);
  }
}

/**
 * A valid `step` message padded to exactly `bytes` bytes on the wire.
 *
 * The pad is ASCII `A`, which JSON never escapes, so one character costs
 * exactly one byte and the arithmetic is exact. Both the floor and the
 * resulting length are asserted rather than assumed: a later edit to the
 * envelope would otherwise produce an off-length frame and silently
 * weaken every cap test built on this helper.
 */
function stepFrameOfExactly(bytes: number): string {
  const envelopeBytes = JSON.stringify({ type: 'step', action_id: 0, pad: '' }).length;
  assert.ok(
    bytes >= envelopeBytes,
    `cannot build a ${bytes}-byte frame; the envelope alone is ${envelopeBytes} bytes`,
  );
  const frame = JSON.stringify({ type: 'step', action_id: 0, pad: 'A'.repeat(bytes - envelopeBytes) });
  assert.equal(Buffer.byteLength(frame), bytes, 'frame builder must produce an exact byte length');
  return frame;
}

/** Recording logger with the `createLogger` shape. */
function recordingLogger() {
  const entries: any[] = [];
  const push = (level: string) => (payload: any) => entries.push({ level, payload });
  return { entries, info: push('info'), warn: push('warn'), error: push('error'), debug: push('debug') };
}

class FakeSocket extends EventEmitter {
  readyState = 1;
  sent: any[] = [];
  pings = 0;
  terminated = false;
  closed = false;

  send(text: string) {
    this.sent.push(JSON.parse(text));
  }
  ping() {
    this.pings += 1;
  }
  terminate() {
    this.terminated = true;
    this.readyState = 3;
    this.emit('close');
  }
  close() {
    this.closed = true;
    this.readyState = 3;
    this.emit('close');
  }
}

// ---------------------------------------------------------------------------

describe('Security and edge cases', () => {
  describe('protocol parseClientMsg', () => {
    it('rejects prototype pollution in message type', () => {
      const payload = JSON.parse('{"__proto__": {"type": "step", "action_id": 1}}');
      assert.throws(() => parseClientMsg(payload), /unknown client msg type/);
    });

    it('rejects prototype pollution in action_id', () => {
      const payload = JSON.parse('{"type": "step", "__proto__": {"action_id": 1}}');
      assert.throws(() => parseClientMsg(payload), /step.action_id must be a non-negative integer/);
    });
  });

  // -- defect 8: command injection via the reset selector --------------------
  describe('reset applyReset selector allowlist', () => {
    it('rejects a newline-smuggled second command instead of relaying it', async () => {
      const bot = { chatLogs: [] as string[], chat(msg: string) { this.chatLogs.push(msg); } };
      await assert.rejects(
        applyReset(bot, {
          strategy: 'teleport',
          teleport: { selector: '@s\n/op attacker\n' },
        } as any),
        /reset\.teleport\.selector must be a Minecraft target selector/,
      );
      // Nothing reached the chat sink: the selector is validated before the
      // first `/tp` is issued.
      assert.deepEqual(bot.chatLogs, []);
    });

    it('does not relay a command smuggled through spawn coordinates', async () => {
      // Same sink and same newline-as-command-boundary problem as the
      // selector, one field over. yaw/pitch were already Number.isFinite
      // guarded; x/y/z were not.
      const bot = { chatLogs: [] as string[], chat(msg: string) { this.chatLogs.push(msg); } };
      await applyReset(bot, {
        strategy: 'teleport',
        teleport: { spawn: { x: '0\n/op attacker', y: 64, z: 0 } },
      } as any);
      const relayed = bot.chatLogs.join('\n');
      assert.ok(!relayed.includes('/op attacker'), `smuggled command relayed: ${relayed}`);
      assert.ok(!relayed.includes('\n/op'), 'newline-prefixed command survived');
      // The non-finite coordinate falls back to the default rather than
      // aborting the reset, matching how yaw/pitch already behave.
      assert.ok(bot.chatLogs[0].startsWith('/tp '), `got: ${bot.chatLogs[0]}`);
    });

    it('rejects every shell/command metacharacter payload', () => {
      const payloads = [
        '@s\n/op attacker',
        '@s\r\n/op attacker',
        '@s /op attacker',
        '@s;/op attacker',
        '@a[tag=x] /kill @e',
        '/op attacker',
        '@s\u0000',
        '@z',
        '@',
        'name-with-dash',
        'way_too_long_player_name',
        '"@s"',
        '@e[type=zombie] extra',
      ];
      for (const payload of payloads) {
        assert.throws(
          () => validateSelector(payload),
          /must be a Minecraft target selector/,
          `expected rejection for ${JSON.stringify(payload)}`,
        );
      }
    });

    it('accepts the allowlisted selector shapes', () => {
      const allowed = [
        '@s',
        '@p',
        '@r',
        '@a',
        '@e',
        '@e[type=minecraft:zombie,limit=1]',
        'ForgeBot',
        'a',
        'sixteen_char_nam',
        '069a79f4-44e9-4726-a5be-fca90e38aaf5',
      ];
      for (const selector of allowed) {
        assert.equal(validateSelector(selector), selector);
      }
    });
  });

  // -- defect 10 replacement: real coverage for the action lookup path -------
  describe('actions executeAction', () => {
    it('does not resolve a handler through Object.prototype', async () => {
      // The deleted test asserted `true` and explained in comments that the
      // action map is TOML-sourced. That is a reason to test the *dispatcher's*
      // own-property discipline, not a reason to skip testing: a plain
      // `registry[kind]` index resolves these names to inherited functions and
      // would call them with the live bot.
      for (const kind of ['constructor', 'toString', 'valueOf', 'hasOwnProperty', '__proto__']) {
        await assert.rejects(
          executeAction(stubBot(), { id: 0, kind, ticks: 1 } as any, { defaultTicks: 1 }),
          /unknown action kind/,
          `kind ${JSON.stringify(kind)} was dispatched`,
        );
      }
    });

    it('rejects an unknown action kind outright', async () => {
      await assert.rejects(
        executeAction(stubBot(), { id: 9, kind: 'rm -rf', ticks: 1 } as any, { defaultTicks: 1 }),
        /unknown action kind/,
      );
    });
  });

  // -- defect 7: prototype pollution in deepMerge ----------------------------
  describe('config deepMerge', () => {
    it('does not let a __proto__ key reach Object.prototype', () => {
      const override = JSON.parse('{"__proto__": {"polluted": "yes"}}');
      const merged = deepMerge({ safe: 1 }, override);
      assert.equal(({} as any).polluted, undefined);
      assert.equal((merged as any).polluted, undefined);
      assert.equal(Object.getPrototypeOf(merged), Object.prototype);
      assert.equal(merged.safe, 1);
    });

    it('drops constructor/prototype overrides instead of merging them', () => {
      const override = JSON.parse('{"constructor": {"prototype": {"polluted": "yes"}}, "keep": 2}');
      const merged = deepMerge({ keep: 1 }, override);
      assert.equal(({} as any).polluted, undefined);
      assert.equal(Object.prototype.hasOwnProperty.call(merged, 'constructor'), false);
      assert.equal(merged.keep, 2);
    });

    it('sanitises a polluted base as well as the override', () => {
      const base = JSON.parse('{"__proto__": {"polluted": "yes"}, "kept": true}');
      const merged = deepMerge(base, { other: 1 });
      assert.equal(Object.prototype.hasOwnProperty.call(merged, '__proto__'), false);
      assert.equal(merged.kept, true);
      assert.equal(merged.other, 1);
    });

    it('survives a hostile env.toml without polluting Object.prototype', () => {
      const cfg = normalizeEnvConfig(
        JSON.parse('{"ws_url": "ws://127.0.0.1:8765", "observation": {"__proto__": {"radius": 9999}}}'),
      );
      assert.equal(({} as any).radius, undefined);
      assert.equal(cfg.observation.radius, 8);
    });
  });

  // -- defect 1/2/3/4 config surface ----------------------------------------
  describe('websocket hardening config', () => {
    it('defaults every limit when no [websocket] table is present', () => {
      const cfg = normalizeEnvConfig({ ws_url: 'ws://127.0.0.1:8765' });
      assert.deepEqual(
        {
          max_payload_bytes: cfg.websocket!.max_payload_bytes,
          max_queue_depth: cfg.websocket!.max_queue_depth,
          idle_timeout_ms: cfg.websocket!.idle_timeout_ms,
          ping_interval_ms: cfg.websocket!.ping_interval_ms,
          ping_timeout_ms: cfg.websocket!.ping_timeout_ms,
          auth_token: cfg.websocket!.auth_token,
          auth_query_param: cfg.websocket!.auth_query_param,
        },
        { ...DEFAULT_WEBSOCKET_LIMITS },
      );
      // Bind fields stay derived from ws_url.
      assert.equal(cfg.websocket!.host, '127.0.0.1');
      assert.equal(cfg.websocket!.port, 8765);
      // A loopback ws_url keeps the listener loopback-only.
      assert.equal(cfg.websocket!.bind_host, '127.0.0.1');
    });

    it('never lets the ping deadline undercut idle_timeout_ms', () => {
      // Regression guard for the defaults, not just the plumbing. The
      // production client (crates/forge-env-mc) is synchronous and cannot
      // answer a ping while it is writing a trajectory or re-hashing an ONNX
      // bundle between episodes. If ping_timeout_ms ever drops below
      // idle_timeout_ms, the control channel grows a second, tighter deadline
      // that reclaims healthy clients -- which is exactly the bug this pins.
      assert.ok(
        DEFAULT_WEBSOCKET_LIMITS.ping_timeout_ms >= DEFAULT_WEBSOCKET_LIMITS.idle_timeout_ms,
        `ping_timeout_ms (${DEFAULT_WEBSOCKET_LIMITS.ping_timeout_ms}) must be >= ` +
          `idle_timeout_ms (${DEFAULT_WEBSOCKET_LIMITS.idle_timeout_ms})`,
      );
      assert.ok(
        DEFAULT_WEBSOCKET_LIMITS.ping_timeout_ms > DEFAULT_WEBSOCKET_LIMITS.ping_interval_ms,
        'a single missed ping must not reclaim the session',
      );
    });

    it('derives bind_host from ws_url, and lets an explicit override win', () => {
      // The docker overlay uses ws://mc-bot:8765. Binding that hostname made
      // the server listen on the container's own eth0 address, so the
      // healthcheck's connection to 127.0.0.1 was refused and `runner` never
      // cleared `service_healthy`. Non-loopback => listen on all interfaces.
      assert.equal(
        normalizeEnvConfig({ ws_url: 'ws://mc-bot:8765' }).websocket!.bind_host,
        undefined,
      );
      // A loopback ws_url binds a literal IP, never the name. Passing
      // "localhost" to Node's listen() goes through DNS, and on a dual-stack
      // host that commonly resolves to ::1 first -- so the server would be
      // IPv6-loopback-only while the Dockerfile healthcheck dials the literal
      // 127.0.0.1. Same unhealthy-container/stack-down failure this function
      // exists to prevent, reached by a different route. (Copilot review.)
      assert.equal(
        normalizeEnvConfig({ ws_url: 'ws://localhost:8765' }).websocket!.bind_host,
        '127.0.0.1',
        'localhost must resolve to the IPv4 literal the healthcheck probes',
      );
      assert.equal(
        normalizeEnvConfig({ ws_url: 'ws://127.0.0.1:8765' }).websocket!.bind_host,
        '127.0.0.1',
      );
      // An explicitly IPv6 ws_url is a deliberate choice and stays IPv6.
      // WHATWG `URL` keeps the brackets (hostname === '[::1]'), which the
      // original loopback table did not account for -- so this input matched
      // nothing and bound every interface. Node's listen() wants it bare.
      assert.equal(
        normalizeEnvConfig({ ws_url: 'ws://[::1]:8765' }).websocket!.bind_host,
        '::1',
        'an IPv6-loopback ws_url must bind IPv6 loopback, not every interface',
      );
      assert.equal(
        normalizeEnvConfig({
          ws_url: 'ws://mc-bot:8765',
          websocket: { bind_host: '10.1.2.3' },
        }).websocket!.bind_host,
        '10.1.2.3',
      );
      // An explicit override is never rewritten, including to a name.
      assert.equal(
        normalizeEnvConfig({
          ws_url: 'ws://mc-bot:8765',
          websocket: { bind_host: 'localhost' },
        }).websocket!.bind_host,
        'localhost',
      );
    });

    it('rejects non-integer and out-of-range limits', () => {
      assert.throws(() => normalizeWebsocketConfig({ max_payload_bytes: 0 }), /max_payload_bytes/);
      assert.throws(() => normalizeWebsocketConfig({ max_queue_depth: -1 }), /max_queue_depth/);
      assert.throws(() => normalizeWebsocketConfig({ idle_timeout_ms: 1.5 }), /idle_timeout_ms/);
      assert.throws(() => normalizeWebsocketConfig({ ping_interval_ms: 'soon' }), /ping_interval_ms/);
      assert.throws(() => normalizeWebsocketConfig({ auth_token: 42 }), /auth_token/);
      assert.throws(() => normalizeWebsocketConfig({ auth_query_param: '' }), /auth_query_param/);
    });

    it('treats an empty auth_token as unset', () => {
      assert.equal(normalizeWebsocketConfig({ auth_token: '' }).auth_token, null);
      assert.equal(normalizeWebsocketConfig({ auth_token: 's3cret' }).auth_token, 's3cret');
    });
  });

  // -- defect 3: handshake auth ---------------------------------------------
  describe('handshake authorization', () => {
    it('extracts the token from a Bearer header or the query string', () => {
      assert.equal(extractAuthToken({ headers: { authorization: 'Bearer abc123' } }, 'token'), 'abc123');
      assert.equal(extractAuthToken({ headers: {}, url: '/?token=abc123' }, 'token'), 'abc123');
      assert.equal(extractAuthToken({ headers: {}, url: '/?other=abc123' }, 'token'), null);
      assert.equal(extractAuthToken({ headers: {}, url: 'not a url' }, 'token'), null);
      assert.equal(extractAuthToken({}, 'token'), null);
    });

    it('compares tokens without an early-exit on content', () => {
      assert.equal(timingSafeEquals('abc', 'abc'), true);
      assert.equal(timingSafeEquals('abc', 'abd'), false);
      assert.equal(timingSafeEquals('abc', 'abcd'), false);
      assert.equal(timingSafeEquals('', ''), true);
    });

    it('allows everything when no token is configured (backwards compatible)', () => {
      const limits = { auth_token: null, auth_query_param: 'token' };
      assert.equal(authorizeRequest({ headers: {}, url: '/' }, limits), true);
    });

    it('rejects a missing or wrong token when one is configured', () => {
      const limits = { auth_token: 'right', auth_query_param: 'token' };
      assert.equal(authorizeRequest({ headers: {}, url: '/' }, limits), false);
      assert.equal(authorizeRequest({ headers: {}, url: '/?token=wrong' }, limits), false);
      assert.equal(authorizeRequest({ headers: { authorization: 'Bearer wrong' } }, limits), false);
      assert.equal(authorizeRequest({ headers: {}, url: '/?token=right' }, limits), true);
      assert.equal(authorizeRequest({ headers: { authorization: 'Bearer right' } }, limits), true);
    });
  });

  // -- defect 5: error listener must precede the first send ------------------
  describe('socket error listener ordering', () => {
    it('does not throw when the hello send emits an error on a closed socket', () => {
      class ErrorOnSend extends EventEmitter {
        readyState = 3;
        send() {
          // Mirrors `ws`: a send on a non-OPEN socket surfaces as an 'error'
          // emit, which throws out of EventEmitter when unhandled.
          this.emit('error', new Error('WebSocket is not open: readyState 3 (CLOSED)'));
        }
        close() {}
      }
      const logger = recordingLogger();
      const handle = createConnectionHandler({ bot: stubBot(), bundle: testBundle(8765, NO_TIMERS), logger });
      const socket = new ErrorOnSend();
      assert.doesNotThrow(() => handle(socket as any));
      assert.ok(logger.entries.some((e) => e.payload?.event === 'websocket_error'));
    });

    it('does not throw when the BUSY reply emits an error', () => {
      class ErrorOnSend extends EventEmitter {
        readyState = 1;
        send() {
          this.emit('error', new Error('WebSocket is not open: readyState 3 (CLOSED)'));
        }
        close() {}
      }
      const logger = recordingLogger();
      const handle = createConnectionHandler({ bot: stubBot(), bundle: testBundle(8765, NO_TIMERS), logger });
      handle(new FakeSocket() as any); // occupies the single client slot
      assert.doesNotThrow(() => handle(new ErrorOnSend() as any));
    });
  });

  // -- defect 9: information disclosure --------------------------------------
  describe('internal error redaction', () => {
    it('returns a generic message to the peer and logs the detail', async () => {
      const secret = '/home/operator/secrets/model.onnx';
      const bot = stubBot();
      bot.waitForTicks = async () => {
        throw new Error(`ENOENT: no such file or directory, open '${secret}'`);
      };
      const logger = recordingLogger();
      const socket = new FakeSocket();
      const handle = createConnectionHandler({ bot, bundle: testBundle(8765, NO_TIMERS), logger });
      handle(socket as any);

      socket.emit('message', JSON.stringify({ type: 'step', action_id: 1 }));
      await delay(10);

      const reply = socket.sent.at(-1);
      assert.equal(reply.type, 'error');
      assert.equal(reply.code, 'INTERNAL');
      assert.equal(reply.message, GENERIC_INTERNAL_MESSAGE);
      assert.ok(!JSON.stringify(socket.sent).includes(secret), 'path leaked to the client');
      assert.ok(
        logger.entries.some((e) => e.payload?.event === 'step_error' && String(e.payload.error).includes(secret)),
        'detail was not logged server-side',
      );
    });
  });

  // -- defect 6: process-level crash handlers --------------------------------
  describe('process guards', () => {
    it('logs and exits non-zero on an unhandled rejection', () => {
      const logger = recordingLogger();
      const proc = new EventEmitter();
      const exits: number[] = [];
      installProcessGuards({ logger, processRef: proc, exit: (code) => exits.push(code) });

      proc.emit('unhandledRejection', new Error('boom-rejection'));
      const entry = logger.entries.find((e) => e.payload?.event === 'unhandled_rejection');
      assert.ok(entry, 'no unhandled_rejection log');
      assert.equal(entry.level, 'error');
      assert.equal(entry.payload.error, 'boom-rejection');
      assert.ok(entry.payload.stack, 'stack not captured');
      assert.deepEqual(exits, [FATAL_EXIT_CODE]);
    });

    it('logs and exits non-zero on an uncaught exception', () => {
      const logger = recordingLogger();
      const proc = new EventEmitter();
      const exits: number[] = [];
      installProcessGuards({ logger, processRef: proc, exit: (code) => exits.push(code) });

      proc.emit('uncaughtException', new Error('boom-exception'), 'uncaughtException');
      const entry = logger.entries.find((e) => e.payload?.event === 'uncaught_exception');
      assert.ok(entry, 'no uncaught_exception log');
      assert.equal(entry.payload.error, 'boom-exception');
      assert.deepEqual(exits, [FATAL_EXIT_CODE]);
    });

    it('handles a non-Error rejection reason', () => {
      const logger = recordingLogger();
      const proc = new EventEmitter();
      const exits: number[] = [];
      installProcessGuards({ logger, processRef: proc, exit: (code) => exits.push(code) });
      proc.emit('unhandledRejection', 'plain string reason');
      assert.equal(logger.entries.at(-1).payload.error, 'plain string reason');
      assert.deepEqual(exits, [FATAL_EXIT_CODE]);
    });
  });

  // -- defect 4: keepalive reclaim of a dead peer ----------------------------
  describe('keepalive', () => {
    it('terminates a peer that never answers a ping and frees the slot', async () => {
      const logger = recordingLogger();
      const handle = createConnectionHandler({
        bot: stubBot(),
        // idle timeout disabled so this exercises the ping path alone.
        // ping_timeout_ms is what decides reclaim; ping_interval_ms only sets
        // the probe cadence.
        bundle: testBundle(8765, {
          idle_timeout_ms: 0,
          ping_interval_ms: 25,
          ping_timeout_ms: 60,
        }),
        logger,
      });
      const dead = new FakeSocket();
      handle(dead as any);
      assert.equal(dead.sent[0].type, 'hello');

      // A single missed ping must NOT reclaim: the production client is
      // synchronous and cannot pong while it is writing a trajectory or
      // re-hashing an ONNX bundle between episodes.
      await delay(40);
      assert.ok(dead.pings >= 1, 'no ping was sent');
      assert.equal(
        dead.terminated,
        false,
        'reclaimed after one missed ping — a busy-but-alive client would be killed',
      );

      // Past ping_timeout_ms with still no pong, the peer is genuinely dead.
      await delay(120);
      assert.equal(dead.terminated, true);
      assert.ok(
        logger.entries.some(
          (e) => e.payload?.event === 'websocket_session_reclaimed' && e.payload.reason === 'ping_timeout',
        ),
      );

      // The single-client slot is free again.
      const fresh = new FakeSocket();
      handle(fresh as any);
      assert.equal(fresh.sent[0].type, 'hello');
      fresh.close();
    });
  });

  // =========================================================================
  // Adversarial tests against a real WebSocketServer
  // =========================================================================
  describe('live WebSocketServer hardening', () => {
    // -- defect 1 -----------------------------------------------------------
    it('closes the connection with 1009 when a frame exceeds max_payload_bytes', async () => {
      await withServer({ max_payload_bytes: 512, idle_timeout_ms: 0, ping_interval_ms: 0 }, async ({ url }) => {
        const socket = await connectAndHello(url);
        const closed = nextClose(socket);
        socket.send(JSON.stringify({ type: 'step', action_id: 0, pad: 'A'.repeat(4096) }));
        assert.equal(await closed, 1009, 'expected a 1009 "message too big" close');
      });
    });

    it('still accepts a normal-sized protocol message under the same cap', async () => {
      await withServer({ max_payload_bytes: 512, idle_timeout_ms: 0, ping_interval_ms: 0 }, async ({ url }) => {
        const socket = await connectAndHello(url);
        socket.send(JSON.stringify({ type: 'step', action_id: 0 }));
        const reply = await nextMessage(socket);
        assert.equal(reply.type, 'observation');
        socket.close();
      });
    });

    // The pair above tests the cap at 8x over and at ~30 bytes under, so a
    // cap that drifted by any factor -- `max_payload_bytes * 8`, or the
    // default substituted for the configured value -- satisfies both and
    // survives. These two pin the exact byte at which behaviour changes.
    it('accepts a frame of exactly max_payload_bytes', async () => {
      await withServer({ ...NO_TIMERS, max_payload_bytes: PAYLOAD_CAP_BYTES }, async ({ url }) => {
        const socket = await connectAndHello(url);
        socket.send(stepFrameOfExactly(PAYLOAD_CAP_BYTES));
        const reply = await nextMessage(socket);
        assert.equal(reply.type, 'observation', 'a frame exactly at the cap is within it');
        socket.close();
      });
    });

    it('closes with 1009 on a frame one byte over max_payload_bytes', async () => {
      await withServer({ ...NO_TIMERS, max_payload_bytes: PAYLOAD_CAP_BYTES }, async ({ url }) => {
        const socket = await connectAndHello(url);
        const closed = nextClose(socket);
        socket.send(stepFrameOfExactly(PAYLOAD_CAP_BYTES + 1));
        assert.equal(await closed, 1009, 'one byte over the cap is over the cap');
      });
    });

    // -- defect 2 -----------------------------------------------------------
    it('answers a flood that outruns the drain with BACKPRESSURE and closes', async () => {
      let release: () => void = () => {};
      const gate = new Promise<void>((done) => {
        release = done;
      });
      try {
        await withServer(
          { max_queue_depth: 2, idle_timeout_ms: 0, ping_interval_ms: 0 },
          async ({ url }) => {
            const socket = await connectAndHello(url);
            const errorSeen = new Promise<any>((resolve, reject) => {
              const timer = setTimeout(() => reject(new Error('no BACKPRESSURE error')), 4000);
              socket.on('message', (raw) => {
                const msg = JSON.parse(raw.toString());
                if (msg.type === 'error') {
                  clearTimeout(timer);
                  resolve(msg);
                }
              });
            });
            const closed = nextClose(socket);
            // Every `step` blocks on the gated tick wait, so nothing drains.
            for (let i = 0; i < 8; i += 1) {
              socket.send(JSON.stringify({ type: 'step', action_id: 1 }));
            }
            const err = await errorSeen;
            assert.equal(err.code, 'BACKPRESSURE');
            assert.match(err.message, /queue depth exceeded \(max 2\)/);
            await closed;
          },
          { gate },
        );
      } finally {
        release();
      }
    });

    it('sustains sequential traffic well past the queue cap', async () => {
      await withServer({ max_queue_depth: 2, idle_timeout_ms: 0, ping_interval_ms: 0 }, async ({ url }) => {
        const socket = await connectAndHello(url);
        for (let i = 0; i < 10; i += 1) {
          socket.send(JSON.stringify({ type: 'step', action_id: 0 }));
          const reply = await nextMessage(socket);
          assert.equal(reply.type, 'observation');
        }
        socket.close();
      });
    });

    // The flood above sends 8 against a cap of 2, and the sequential test
    // never lets depth exceed 1, so `queueDepth > max` in place of
    // `queueDepth >= max` passes both: it just rejects on the 4th message
    // instead of the 3rd. These two bracket the transition exactly.
    it('admits exactly max_queue_depth messages in flight', async () => {
      let release: () => void = () => {};
      const gate = new Promise<void>((done) => {
        release = done;
      });
      try {
        await withServer(
          { ...NO_TIMERS, max_queue_depth: QUEUE_CAP },
          async ({ url }) => {
            const socket = await connectAndHello(url);
            const seen: any[] = [];
            socket.on('message', (raw) => seen.push(JSON.parse(raw.toString())));

            // Each `step` blocks on the gated tick wait, so these pile up
            // rather than draining: in-flight depth climbs to exactly the cap.
            for (let i = 0; i < QUEUE_CAP; i += 1) {
              socket.send(JSON.stringify({ type: 'step', action_id: 0 }));
            }
            await delay(ENQUEUE_SETTLE_MS);
            release();

            // Positive evidence that every one of them was admitted: each
            // drains into an observation. A cap enforced one slot early
            // would have answered the last with BACKPRESSURE and closed.
            await waitUntil(() => seen.length >= QUEUE_CAP, `${QUEUE_CAP} replies`);
            assert.equal(
              seen.filter((m) => m.type === 'observation').length,
              QUEUE_CAP,
              'every message up to the cap must be served',
            );
            assert.deepEqual(
              seen.filter((m) => m.type === 'error'),
              [],
              'nothing at or under the cap may be rejected',
            );
            socket.close();
          },
          { gate },
        );
      } finally {
        release();
      }
    });

    it('rejects the first message past max_queue_depth', async () => {
      let release: () => void = () => {};
      const gate = new Promise<void>((done) => {
        release = done;
      });
      try {
        await withServer(
          { ...NO_TIMERS, max_queue_depth: QUEUE_CAP },
          async ({ url }) => {
            const socket = await connectAndHello(url);
            const errors: any[] = [];
            socket.on('message', (raw) => {
              const msg = JSON.parse(raw.toString());
              if (msg.type === 'error') errors.push(msg);
            });
            const closed = nextClose(socket);

            // Exactly one more than the cap, and no more: an off-by-one in
            // the guard has nowhere to hide behind a flood.
            for (let i = 0; i < QUEUE_CAP + 1; i += 1) {
              socket.send(JSON.stringify({ type: 'step', action_id: 0 }));
            }
            await waitUntil(() => errors.length > 0, 'a BACKPRESSURE error');
            assert.equal(errors[0].code, 'BACKPRESSURE');
            assert.match(errors[0].message, new RegExp(`queue depth exceeded \\(max ${QUEUE_CAP}\\)`));
            await closed;
          },
          { gate },
        );
      } finally {
        release();
      }
    });

    // -- defect 3 -----------------------------------------------------------
    it('rejects the handshake with 401 when the token is missing', async () => {
      await withServer({ auth_token: 'correct-horse', idle_timeout_ms: 0, ping_interval_ms: 0 }, async ({ url }) => {
        const socket = new WebSocket(url);
        const err = await new Promise<Error>((resolve, reject) => {
          const timer = setTimeout(() => reject(new Error('no handshake error')), 4000);
          socket.once('error', (e) => {
            clearTimeout(timer);
            resolve(e);
          });
        });
        assert.match(err.message, /401/);
      });
    });

    it('rejects the handshake when the token is wrong', async () => {
      await withServer({ auth_token: 'correct-horse', idle_timeout_ms: 0, ping_interval_ms: 0 }, async ({ url }) => {
        const socket = new WebSocket(`${url}/?token=wrong-horse`);
        const err = await new Promise<Error>((resolve, reject) => {
          const timer = setTimeout(() => reject(new Error('no handshake error')), 4000);
          socket.once('error', (e) => {
            clearTimeout(timer);
            resolve(e);
          });
        });
        assert.match(err.message, /401/);
      });
    });

    it('accepts the token via the query parameter', async () => {
      await withServer({ auth_token: 'correct-horse', idle_timeout_ms: 0, ping_interval_ms: 0 }, async ({ url }) => {
        const socket = await connectAndHello(`${url}/?token=correct-horse`);
        socket.close();
      });
    });

    it('accepts the token via an Authorization: Bearer header', async () => {
      await withServer({ auth_token: 'correct-horse', idle_timeout_ms: 0, ping_interval_ms: 0 }, async ({ url }) => {
        const socket = await connectAndHello(url, { headers: { Authorization: 'Bearer correct-horse' } });
        socket.close();
      });
    });

    // -- defect 4 -----------------------------------------------------------
    it('reclaims the bot from a connected-but-silent client after the idle timeout', async () => {
      await withServer({ idle_timeout_ms: 150, ping_interval_ms: 40 }, async ({ url }) => {
        const squatter = await connectAndHello(url);
        // The client answers pings automatically, so only the absence of
        // application traffic can reclaim the slot.
        await nextClose(squatter);

        // A fresh client is admitted rather than being told BUSY.
        const successor = new WebSocket(url);
        const hello = await nextMessage(successor);
        assert.equal(hello.type, 'hello');
        successor.close();
      });
    });

    it('holds the slot against a second client while the first is active', async () => {
      await withServer({ idle_timeout_ms: 0, ping_interval_ms: 0 }, async ({ url }) => {
        const first = await connectAndHello(url);
        const second = new WebSocket(url);
        const busy = await nextMessage(second);
        assert.equal(busy.type, 'error');
        assert.equal(busy.code, 'BUSY');
        second.close();
        first.close();
      });
    });
  });
});
