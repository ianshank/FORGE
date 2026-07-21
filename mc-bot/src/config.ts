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
  websocket?: {
    url: string;
    host: string;
    port: number;
  };
  [key: string]: any;
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

function isPlainObject(value: any): boolean {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

export function deepMerge(base: any, override: any): any {
  const output = { ...base };
  if (!isPlainObject(override)) return output;
  for (const [key, value] of Object.entries(override)) {
    if (isPlainObject(value) && isPlainObject(base[key])) {
      output[key] = deepMerge(base[key], value);
    } else if (value !== undefined) {
      output[key] = value;
    }
  }
  return output;
}

export function normalizeEnvConfig(raw: any = {}): EnvConfig {
  const cfg = deepMerge(DEFAULT_ENV_CONFIG, raw);
  const wsUrl = new URL(cfg.ws_url);
  if (wsUrl.protocol !== 'ws:' && wsUrl.protocol !== 'wss:') {
    throw new Error(`ws_url must use ws:// or wss://, got ${cfg.ws_url}`);
  }
  const portText = wsUrl.port || (wsUrl.protocol === 'wss:' ? '443' : '80');
  const websocket = {
    url: cfg.ws_url,
    host: wsUrl.hostname,
    port: Number(portText),
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