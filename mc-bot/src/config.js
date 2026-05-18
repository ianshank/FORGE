import { readFile } from 'node:fs/promises';
import { dirname, isAbsolute, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { loadActionMap } from './action_map.js';
import { buildReward } from './reward/index.js';
import { buildRewardConfig, combinedSchemaId, loadRewardConfig } from './reward_config.js';

const MODULE_DIR = dirname(fileURLToPath(import.meta.url));
export const DEFAULT_REPO_ROOT = resolve(MODULE_DIR, '..', '..');
export const DEFAULT_CONFIG_DIR = resolve(DEFAULT_REPO_ROOT, 'configs', 'minecraft');

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
  }),
  viewer: Object.freeze({
    enabled: false,
    host: '127.0.0.1',
    port: 3007,
    first_person: true,
    view_distance: 6,
  }),
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

function isPlainObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

export function deepMerge(base, override) {
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

export function normalizeEnvConfig(raw = {}) {
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
  if (!Number.isInteger(websocket.port) || websocket.port <= 0) {
    throw new Error(`ws_url port must be positive integer, got ${portText}`);
  }
  return { ...cfg, websocket };
}

export function normalizeResetConfig(raw = {}) {
  return deepMerge(DEFAULT_RESET_CONFIG, raw);
}

export function resolveConfigPath(pathValue, repoRoot = DEFAULT_REPO_ROOT) {
  if (typeof pathValue !== 'string' || pathValue.length === 0) {
    throw new Error(`config path must be a non-empty string, got ${pathValue}`);
  }
  return isAbsolute(pathValue) ? pathValue : resolve(repoRoot, pathValue);
}

async function defaultTomlParser() {
  const module = await import('smol-toml');
  const parser = module.parse ?? module.default?.parse ?? module.default;
  if (typeof parser !== 'function') {
    throw new Error('smol-toml module does not expose a parse function');
  }
  return parser;
}

async function parseTomlFile(path, tomlParse) {
  const raw = await readFile(path, 'utf8');
  const parser = tomlParse ?? await defaultTomlParser();
  return parser(raw);
}

export async function loadEnvConfig(path = resolve(DEFAULT_CONFIG_DIR, 'env.toml'), options = {}) {
  const parsed = await parseTomlFile(path, options.tomlParse);
  return normalizeEnvConfig(parsed);
}

export async function loadResetConfig(path = resolve(DEFAULT_CONFIG_DIR, 'reset.toml'), options = {}) {
  const parsed = await parseTomlFile(path, options.tomlParse);
  return normalizeResetConfig(parsed);
}

export async function loadConfigBundle(options = {}) {
  const repoRoot = options.repoRoot ?? DEFAULT_REPO_ROOT;
  const envPath = options.envPath ?? resolve(DEFAULT_CONFIG_DIR, 'env.toml');
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

  return {
    env,
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

export function buildConfigBundleFromObjects({ env, reset, actionMapData, rewardData }) {
  const actionMap = actionMapData;
  const rewardConfig = buildRewardConfig(rewardData);
  return {
    env: normalizeEnvConfig(env),
    reset: normalizeResetConfig(reset),
    actionMap,
    rewardConfig,
    rewardFn: buildReward({ reward: rewardConfig.entries }),
    schemaId: combinedSchemaId(actionMap.schemaId, rewardConfig.canonicalSha256()),
  };
}