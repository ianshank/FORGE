import { readFile } from 'node:fs/promises';
import { dirname, isAbsolute, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { loadActionMap, type ActionMap } from './action_map.js';
import { DEFAULT_RECONNECT_CONFIG, DEFAULT_SPAWN_TIMEOUT_MS } from './bot_manager.js';
import { buildReward } from './reward/index.js';
import { buildRewardConfig, combinedSchemaId, loadRewardConfig, type RewardConfig } from './reward_config.js';

const MODULE_DIR = dirname(fileURLToPath(import.meta.url));
export const DEFAULT_REPO_ROOT = resolve(MODULE_DIR, '..', '..');
export const DEFAULT_CONFIG_DIR = resolve(DEFAULT_REPO_ROOT, 'configs', 'minecraft');

export interface EnvConfig {
  action_map_path: string;
  rewards_path: string;
  reset_path: string;
  ws_url: string;
  heartbeat_ms: number;
  bot: {
    host: string;
    port: number;
    username: string;
    version: string;
    auth: string;
    spawn_timeout_ms: number;
    [key: string]: any;
  };
  episode: {
    max_ticks: number;
    action_repeat: number;
    terminate_on_death: boolean;
    [key: string]: any;
  };
  observation: {
    radius: number;
    expected_dim: number | null;
    include_position: boolean;
    include_velocity: boolean;
    include_orientation: boolean;
    include_vitals: boolean;
    include_inventory: boolean;
    inventory_slots: number;
    inventory_features_per_slot: number;
    position_scale: number;
    hash_mod: number;
    max_stack_size: number;
    block_embeddings_path: string;
    use_raw_block_id: boolean;
    block_embeddings?: Record<string, any>;
    [key: string]: any;
  };
  viewer: {
    enabled: boolean;
    host: string;
    port: number;
    first_person: boolean;
    view_distance: number;
    [key: string]: any;
  };
  reconnect: {
    backoff_ms: number[];
    max_attempts: number;
    stale_timeout_ms: number;
  };
  websocket?: WebsocketConfig;
  [key: string]: any;
}

/**
 * Hardening limits for the control WebSocket. Every field is optional in
 * `env.toml` (`[websocket]` table) and falls back to
 * {@link DEFAULT_WEBSOCKET_LIMITS}, so pre-existing configs keep parsing
 * unchanged.
 */
export interface WebsocketLimits {
  /** Maximum accepted size of a single inbound WebSocket frame, in bytes. */
  max_payload_bytes: number;
  /** Maximum number of client messages that may be in flight concurrently. */
  max_queue_depth: number;
  /** Reclaim the single client slot after this many ms without a client message (0 disables). */
  idle_timeout_ms: number;
  /** Interval between server-initiated WebSocket pings, in ms (0 disables). */
  ping_interval_ms: number;
  /**
   * How long a ping may go unanswered before the peer is declared dead, in ms
   * (0 disables ping-based reclaim). Must exceed the longest legitimate gap in
   * which the client does not read its socket — see
   * {@link DEFAULT_WEBSOCKET_LIMITS} for why that is much longer than
   * {@link WebsocketLimits.ping_interval_ms}.
   */
  ping_timeout_ms: number;
  /** Optional shared secret required at handshake time; `null` = unauthenticated. */
  auth_token: string | null;
  /** Query-string parameter carrying {@link WebsocketLimits.auth_token}. */
  auth_query_param: string;
}

/** Derived bind information plus the {@link WebsocketLimits} knobs. */
export interface WebsocketConfig extends WebsocketLimits {
  /** Echo of `ws_url`. Derived — a `[websocket] url` override is ignored. */
  url: string;
  /** Hostname parsed out of `ws_url`. This is the *client's* dial target. */
  host: string;
  /** Bind port, derived from `ws_url`. */
  port: number;
  /**
   * Address the server actually listens on. See {@link resolveBindHost} —
   * this is deliberately NOT the same thing as {@link WebsocketConfig.host},
   * because a client dial target and a server bind address answer different
   * questions. `undefined` means "every interface".
   */
  bind_host: string | undefined;
}

/**
 * Hostnames that mean "this machine only".
 *
 * `URL` strips the brackets from an IPv6 authority, so `ws://[::1]:8765`
 * parses to a bare `::1` here.
 */
const LOOPBACK_HOSTNAMES: ReadonlySet<string> = new Set(['localhost', '127.0.0.1', '::1']);

/**
 * Decide what address the WebSocket server binds to.
 *
 * The bind address used to be taken straight from `ws_url`'s hostname, which
 * conflates two different things: where a *client* dials and where the
 * *server* listens. Under docker that conflation breaks the stack.
 * `configs/minecraft/env.docker.toml` sets `ws_url = "ws://mc-bot:8765"`, so
 * the server bound to the `mc-bot` service name — i.e. the container's own
 * eth0 address — and then refused the container healthcheck's connection to
 * `127.0.0.1`. Because `compose.minecraft.yml` gates the `runner` service on
 * `mc-bot: service_healthy`, that alone kept the whole v0.5 stack down.
 *
 * The rule:
 * - an explicit `[websocket] bind_host` always wins, so an operator can pin
 *   the address without touching `ws_url`;
 * - a loopback `ws_url` keeps the server loopback-only, which is what a local
 *   `cargo run` wants and matches forge-server's loopback-by-default posture;
 * - anything else listens on every interface. That is the right call *inside a
 *   container*, where the network namespace is the isolation boundary and the
 *   compose `ports:` entry (bound to `${BIND_HOST:-127.0.0.1}`) is what limits
 *   host exposure.
 *
 * @param wsUrlHostname Hostname parsed from `ws_url`.
 * @param override Explicit `[websocket] bind_host`, if the operator set one.
 * @returns The bind address, or `undefined` to listen on every interface.
 */
export function resolveBindHost(
  wsUrlHostname: string,
  override?: unknown,
): string | undefined {
  if (override !== undefined && override !== null && override !== '') {
    if (typeof override !== 'string') {
      throw new Error(`websocket.bind_host must be a string, got ${typeof override}`);
    }
    return override;
  }
  return LOOPBACK_HOSTNAMES.has(wsUrlHostname) ? wsUrlHostname : undefined;
}

export interface ResetConfig {
  strategy: string;
  teleport?: {
    spawn?: { x: number; y: number; z: number };
    yaw?: number;
    pitch?: number;
    selector?: string;
    clear_inventory?: boolean;
    restore_health?: boolean;
    restore_food?: boolean;
  };
  [key: string]: any;
}

export interface ConfigBundle {
  env: EnvConfig;
  reset: ResetConfig;
  actionMap: ActionMap;
  rewardConfig: RewardConfig;
  rewardFn: (event: { prev: any; curr: any }) => number;
  schemaId: string;
  paths?: {
    envPath: string;
    actionMapPath: string;
    rewardsPath: string;
    resetPath: string;
  };
}

/**
 * Defaults for the `[websocket]` hardening table.
 *
 * Rationale for each number (all overridable in `configs/minecraft/env.toml`):
 *
 * - `max_payload_bytes` (16 KiB): every client→server frame this server
 *   accepts is a tiny JSON control message — the largest is
 *   `{"type":"reset","seed":9007199254740991}` at ~40 bytes, and
 *   `{"type":"step","action_id":N}` / `{"type":"close"}` are smaller still.
 *   16 KiB leaves ~400x headroom for future protocol fields while replacing
 *   the `ws` library default of 100 MiB per frame, which lets one unauthorised
 *   peer pin 100 MiB of heap per frame.
 * - `max_queue_depth` (32): the wire protocol is strictly request/response —
 *   a well-behaved client has at most one message in flight. 32 tolerates
 *   pipelining and reordering bursts while bounding the promise chain.
 * - `idle_timeout_ms` (120_000): the single-client slot is held until the
 *   socket closes, so a peer that connects and goes silent otherwise owns the
 *   bot forever. Two minutes is comfortably longer than the longest legitimate
 *   client-side gap (an inter-episode ONNX bundle hot-reload) and short enough
 *   to reclaim a wedged session without operator action.
 * - `ping_interval_ms` (20_000): server-initiated keepalive. A ping left
 *   unanswered for a full interval marks a half-open TCP connection dead and
 *   reclaims it well before `idle_timeout_ms` elapses.
 * - `auth_token` (`null`): opt-in shared secret. Unset preserves the historical
 *   unauthenticated behaviour (a startup warning is logged instead).
 * - `auth_query_param` (`'token'`): name of the query-string parameter the
 *   client may use instead of an `Authorization: Bearer <token>` header.
 */
export const DEFAULT_WEBSOCKET_LIMITS: Readonly<WebsocketLimits> = Object.freeze({
  max_payload_bytes: 16 * 1024,
  max_queue_depth: 32,
  idle_timeout_ms: 120_000,
  ping_interval_ms: 20_000,
  // Equal to idle_timeout_ms on purpose, so the control channel has ONE
  // deadline rather than two, and the tighter one is not an accident.
  //
  // The production client is synchronous: crates/forge-env-mc/src/client.rs
  // reads the socket only inside recv(), which the runner calls from
  // reset_into/step_into. tungstenite queues a pong only when the ping frame
  // is read, so a client that is busy rather than dead cannot answer. Between
  // episodes the runner writes the trajectory, re-hashes the ONNX bundle and
  // rebuilds three ORT sessions without reading the socket. Reclaiming on a
  // single missed ping (20 s) would kill that client; reclaiming on
  // ping_timeout_ms (120 s) does not, while still evicting a genuinely dead
  // peer well before it matters.
  ping_timeout_ms: 120_000,
  auth_token: null,
  auth_query_param: 'token',
});

export const DEFAULT_ENV_CONFIG = Object.freeze({
  action_map_path: 'configs/minecraft/action_map.toml',
  rewards_path: 'configs/minecraft/rewards.toml',
  reset_path: 'configs/minecraft/reset.toml',
  ws_url: 'ws://127.0.0.1:8765',
  heartbeat_ms: 2000,
  bot: Object.freeze({
    host: '127.0.0.1',
    port: 25565,
    username: 'ForgeBot',
    version: '1.20.4',
    auth: 'offline',
    spawn_timeout_ms: DEFAULT_SPAWN_TIMEOUT_MS,
  }),
  episode: Object.freeze({
    max_ticks: 6000,
    action_repeat: 4,
    terminate_on_death: true,
  }),
  observation: Object.freeze({
    radius: 8,
    expected_dim: null,
    include_position: true,
    include_velocity: true,
    include_orientation: true,
    include_vitals: true,
    include_inventory: true,
    inventory_slots: 9,
    inventory_features_per_slot: 2,
    position_scale: 1024,
    hash_mod: 4096,
    max_stack_size: 64,
    block_embeddings_path: 'configs/minecraft/block_embeddings.toml',
    use_raw_block_id: true,
  }),
  viewer: Object.freeze({
    enabled: false,
    host: '127.0.0.1',
    port: 3007,
    first_person: true,
    view_distance: 6,
  }),
  reconnect: Object.freeze({ ...DEFAULT_RECONNECT_CONFIG }),
  websocket: Object.freeze({ ...DEFAULT_WEBSOCKET_LIMITS }),
});

export const DEFAULT_RESET_CONFIG = Object.freeze({
  strategy: 'teleport',
  teleport: Object.freeze({
    spawn: Object.freeze({ x: 0, y: 64, z: 0 }),
    yaw: 0,
    pitch: 0,
    selector: '@s',
    clear_inventory: true,
    restore_health: true,
    restore_food: true,
  }),
});

/**
 * Property names that must never be copied by {@link deepMerge}.
 *
 * `JSON.parse('{"__proto__": {...}}')` and TOML tables named `constructor` /
 * `prototype` produce *own* properties with these names. Assigning them onto a
 * merge target either walks the `Object.prototype.__proto__` setter (polluting
 * the prototype chain of every object in the process) or shadows the
 * constructor, so they are dropped from both sides of the merge.
 */
const POLLUTING_KEYS: readonly string[] = Object.freeze(['__proto__', 'constructor', 'prototype']);

/** True when `key` is one of the prototype-pollution vectors in {@link POLLUTING_KEYS}. */
export function isPollutingKey(key: string): boolean {
  return POLLUTING_KEYS.includes(key);
}

/**
 * True only for objects whose prototype is `Object.prototype` or `null`.
 *
 * The looser "any non-array object" test used previously treated
 * `Object.prototype` itself — reachable through a `constructor.prototype`
 * override — as a mergeable plain object.
 */
function isPlainObject(value: any): boolean {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    return false;
  }
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

/**
 * Recursively merge `override` onto `base`, returning a new object.
 *
 * Keys in {@link POLLUTING_KEYS} are skipped on both sides, so neither a
 * malicious config file nor a polluted `base` can reach `Object.prototype`.
 */
export function deepMerge(base: any, override: any): any {
  const output: Record<string, any> = {};
  for (const key of Object.keys(Object(base))) {
    if (isPollutingKey(key)) continue;
    output[key] = (base as any)[key];
  }
  if (!isPlainObject(override)) return output;
  for (const key of Object.keys(override)) {
    if (isPollutingKey(key)) continue;
    const value = override[key];
    if (isPlainObject(value) && isPlainObject(output[key])) {
      output[key] = deepMerge(output[key], value);
    } else if (value !== undefined) {
      output[key] = value;
    }
  }
  return output;
}

function requireIntInRange(value: any, name: string, min: number): number {
  if (!Number.isInteger(value) || value < min) {
    throw new Error(`websocket.${name} must be an integer >= ${min}, got ${JSON.stringify(value)}`);
  }
  return value;
}

/**
 * Validate and fill in the `[websocket]` hardening table.
 *
 * Unknown keys are dropped; every known key falls back to
 * {@link DEFAULT_WEBSOCKET_LIMITS} when absent, which is what keeps existing
 * `env.toml` files (none of which carry a `[websocket]` table) parsing
 * unchanged.
 */
export function normalizeWebsocketConfig(raw: any = {}): WebsocketLimits {
  const merged = deepMerge(DEFAULT_WEBSOCKET_LIMITS, raw);
  const authTokenRaw = merged.auth_token;
  let authToken: string | null = null;
  if (authTokenRaw !== null && authTokenRaw !== undefined && authTokenRaw !== '') {
    if (typeof authTokenRaw !== 'string') {
      throw new Error(`websocket.auth_token must be a string or null, got ${typeof authTokenRaw}`);
    }
    authToken = authTokenRaw;
  }
  const authQueryParam = merged.auth_query_param;
  if (typeof authQueryParam !== 'string' || authQueryParam.length === 0) {
    throw new Error(
      `websocket.auth_query_param must be a non-empty string, got ${JSON.stringify(authQueryParam)}`,
    );
  }
  return {
    max_payload_bytes: requireIntInRange(merged.max_payload_bytes, 'max_payload_bytes', 1),
    max_queue_depth: requireIntInRange(merged.max_queue_depth, 'max_queue_depth', 1),
    idle_timeout_ms: requireIntInRange(merged.idle_timeout_ms, 'idle_timeout_ms', 0),
    ping_interval_ms: requireIntInRange(merged.ping_interval_ms, 'ping_interval_ms', 0),
    ping_timeout_ms: requireIntInRange(merged.ping_timeout_ms, 'ping_timeout_ms', 0),
    auth_token: authToken,
    auth_query_param: authQueryParam,
  };
}

export function normalizeEnvConfig(raw: any = {}): EnvConfig {
  const cfg = deepMerge(DEFAULT_ENV_CONFIG, raw);
  const wsUrl = new URL(cfg.ws_url);
  if (wsUrl.protocol !== 'ws:' && wsUrl.protocol !== 'wss:') {
    throw new Error(`ws_url must use ws:// or wss://, got ${cfg.ws_url}`);
  }
  const portText = wsUrl.port || (wsUrl.protocol === 'wss:' ? '443' : '80');
  // `url`/`host`/`port` stay derived from `ws_url` (single source of truth for
  // the bind address); the `[websocket]` table only supplies hardening limits.
  const websocket: WebsocketConfig = {
    ...normalizeWebsocketConfig(cfg.websocket),
    url: cfg.ws_url,
    host: wsUrl.hostname,
    port: Number(portText),
    bind_host: resolveBindHost(wsUrl.hostname, cfg.websocket?.bind_host),
  };
  const reconnect = {
    backoff_ms: Array.isArray(cfg.reconnect?.backoff_ms)
      ? cfg.reconnect.backoff_ms
      : DEFAULT_RECONNECT_CONFIG.backoff_ms,
    max_attempts: cfg.reconnect?.max_attempts ?? DEFAULT_RECONNECT_CONFIG.max_attempts,
    stale_timeout_ms: cfg.reconnect?.stale_timeout_ms ?? DEFAULT_RECONNECT_CONFIG.stale_timeout_ms,
  };
  if (!Number.isInteger(websocket.port) || websocket.port <= 0) {
    throw new Error(`ws_url port must be positive integer, got ${portText}`);
  }
  return { ...cfg, websocket, reconnect };
}

export function normalizeResetConfig(raw: any = {}): ResetConfig {
  return deepMerge(DEFAULT_RESET_CONFIG, raw);
}

export function resolveConfigPath(pathValue: any, repoRoot: string = DEFAULT_REPO_ROOT): string {
  if (typeof pathValue !== 'string' || pathValue.length === 0) {
    throw new Error(`config path must be a non-empty string, got ${pathValue}`);
  }
  return isAbsolute(pathValue) ? pathValue : resolve(repoRoot, pathValue);
}

async function defaultTomlParser(): Promise<(text: string) => any> {
  const module = await import('smol-toml');
  const parser = module.parse ?? (module as any).default?.parse ?? (module as any).default;
  if (typeof parser !== 'function') {
    throw new Error('smol-toml module does not expose a parse function');
  }
  return parser;
}

async function parseTomlFile(path: string, tomlParse?: (text: string) => any): Promise<any> {
  const raw = await readFile(path, 'utf8');
  const parser = tomlParse ?? await defaultTomlParser();
  return parser(raw);
}

export async function loadEnvConfig(
  path: string = resolve(DEFAULT_CONFIG_DIR, 'env.toml'),
  options: { tomlParse?: (text: string) => any } = {}
): Promise<EnvConfig> {
  const parsed = await parseTomlFile(path, options.tomlParse);
  return normalizeEnvConfig(parsed);
}

export async function loadResetConfig(
  path: string = resolve(DEFAULT_CONFIG_DIR, 'reset.toml'),
  options: { tomlParse?: (text: string) => any } = {}
): Promise<ResetConfig> {
  const parsed = await parseTomlFile(path, options.tomlParse);
  return normalizeResetConfig(parsed);
}

export async function loadConfigBundle(
  options: {
    repoRoot?: string;
    envPath?: string;
    tomlParse?: (text: string) => any;
    actionMapPath?: string;
    rewardsPath?: string;
    resetPath?: string;
  } = {}
): Promise<ConfigBundle> {
  const repoRoot = options.repoRoot ?? DEFAULT_REPO_ROOT;
  const envPath = options.envPath ?? resolve(repoRoot, 'configs', 'minecraft', 'env.toml');
  const tomlParse = options.tomlParse ?? await defaultTomlParser();
  const env = await loadEnvConfig(envPath, { tomlParse });
  const actionMapPath = options.actionMapPath ?? resolveConfigPath(env.action_map_path, repoRoot);
  const rewardsPath = options.rewardsPath ?? resolveConfigPath(env.rewards_path, repoRoot);
  const resetPath = options.resetPath ?? resolveConfigPath(env.reset_path, repoRoot);

  const actionMap = await loadActionMap(actionMapPath, tomlParse);
  const rewardConfig = await loadRewardConfig(rewardsPath, tomlParse);
  const reset = await loadResetConfig(resetPath, { tomlParse });
  const rewardFn = buildReward({ reward: rewardConfig.entries });
  const schemaId = combinedSchemaId(actionMap.schemaId, rewardConfig.canonicalSha256());

  // Load block embeddings mapping table
  const defaultEmbeddingsPath = resolve(repoRoot, 'configs', 'minecraft', 'block_embeddings.toml');
  const blockEmbeddingsPath = resolveConfigPath(
    env.observation?.block_embeddings_path ?? defaultEmbeddingsPath,
    repoRoot,
  );
  let blockEmbeddings = {};
  try {
    const parsedEmbeddings = await parseTomlFile(blockEmbeddingsPath, tomlParse);
    blockEmbeddings = parsedEmbeddings.blocks ?? {};
  } catch (_err) {
    // optional / best-effort fallback
  }

  // Inject block embeddings into the observation config
  const observation = {
    ...env.observation,
    block_embeddings: blockEmbeddings,
  };
  const envWithEmbeddings = { ...env, observation };

  return {
    env: envWithEmbeddings,
    reset,
    actionMap,
    rewardConfig,
    rewardFn,
    schemaId,
    paths: {
      envPath,
      actionMapPath,
      rewardsPath,
      resetPath,
    },
  };
}

export function buildConfigBundleFromObjects(options: {
  env: any;
  reset: any;
  actionMapData: any;
  rewardData: any;
}): ConfigBundle {
  const actionMap = options.actionMapData;
  const rewardConfig = buildRewardConfig(options.rewardData);
  return {
    env: normalizeEnvConfig(options.env),
    reset: normalizeResetConfig(options.reset),
    actionMap,
    rewardConfig,
    rewardFn: buildReward({ reward: rewardConfig.entries }),
    schemaId: combinedSchemaId(actionMap.schemaId, rewardConfig.canonicalSha256()),
  };
}