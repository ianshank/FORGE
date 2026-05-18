// Wire protocol — mirrors crates/forge-env-mc/src/protocol.rs.
// JSON-only in v1. SCHEMA_VERSION must match the Rust constant.

export const SCHEMA_VERSION = 1;

/**
 * Build a Hello server message.
 *
 * @param {{ actionCount: number, obsDim: number, schemaId: string }} args
 * @returns {object}
 */
export function helloMsg({ actionCount, obsDim, schemaId }) {
  if (!Number.isInteger(actionCount) || actionCount <= 0) {
    throw new Error(`actionCount must be positive integer, got ${actionCount}`);
  }
  if (!Number.isInteger(obsDim) || obsDim <= 0) {
    throw new Error(`obsDim must be positive integer, got ${obsDim}`);
  }
  if (typeof schemaId !== 'string' || schemaId.length === 0) {
    throw new Error('schemaId must be a non-empty string');
  }
  return {
    type: 'hello',
    schema_version: SCHEMA_VERSION,
    action_count: actionCount,
    obs_dim: obsDim,
    schema_id: schemaId,
  };
}

/**
 * Build an Observation server message.
 *
 * @param {{ tick: number, obs: number[], reward: number,
 *           terminated: boolean, truncated: boolean, info?: object }} args
 * @returns {object}
 */
export function observationMsg({ tick, obs, reward, terminated, truncated, info }) {
  if (!Number.isFinite(tick) || tick < 0) {
    throw new Error(`tick must be non-negative finite number, got ${tick}`);
  }
  if (!Array.isArray(obs)) {
    throw new Error('obs must be an array of numbers');
  }
  if (!Number.isFinite(reward)) {
    throw new Error(`reward must be finite, got ${reward}`);
  }
  return {
    type: 'observation',
    tick,
    obs,
    reward,
    terminated: Boolean(terminated),
    truncated: Boolean(truncated),
    info: info ?? {},
  };
}

/**
 * Build an Error server message.
 *
 * @param {string} code
 * @param {string} message
 * @returns {object}
 */
export function errorMsg(code, message) {
  return { type: 'error', code: String(code), message: String(message) };
}

/**
 * Parse a client message. Throws if `type` is unknown or required
 * fields are missing — protects the bot from malformed requests.
 *
 * @param {string|object} input — JSON string or already-parsed object.
 * @returns {{ type: 'reset', seed: ?number } | { type: 'step', action_id: number } | { type: 'close' }}
 */
export function parseClientMsg(input) {
  const obj = typeof input === 'string' ? JSON.parse(input) : input;
  if (!obj || typeof obj !== 'object') {
    throw new Error(`client msg must be an object, got ${typeof obj}`);
  }
  switch (obj.type) {
    case 'reset': {
      // Seed semantics: optional u64-shaped non-negative integer. We
      // accept null/undefined as "caller didn't pin a seed". Reject
      // anything that can't survive a roundtrip through Rust's u64 —
      // floats, negatives, and values above Number.MAX_SAFE_INTEGER
      // would silently corrupt determinism on the other side.
      if (obj.seed === null || obj.seed === undefined) {
        return { type: 'reset', seed: null };
      }
      const raw = obj.seed;
      const seed = typeof raw === 'number' ? raw : Number(raw);
      if (
        !Number.isFinite(seed) ||
        !Number.isInteger(seed) ||
        seed < 0 ||
        seed > Number.MAX_SAFE_INTEGER
      ) {
        throw new Error(
          `reset.seed must be a non-negative integer \u2264 Number.MAX_SAFE_INTEGER or null, got ${JSON.stringify(
            obj.seed,
          )}`,
        );
      }
      return { type: 'reset', seed };
    }
    case 'step': {
      if (!Number.isInteger(obj.action_id) || obj.action_id < 0) {
        throw new Error(`step.action_id must be a non-negative integer, got ${obj.action_id}`);
      }
      return { type: 'step', action_id: obj.action_id };
    }
    case 'close':
      return { type: 'close' };
    default:
      throw new Error(`unknown client msg type: ${JSON.stringify(obj.type)}`);
  }
}
