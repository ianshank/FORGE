import { DEFAULT_HASH_MOD, finiteNumber, stableStringHash } from './hash.js';
import { encodeBlockGrid, gridFlatDim, type GridShape } from './observation_grid.js';

const DEFAULT_POSITION_SCALE = 1024;
const DEFAULT_MAX_STACK_SIZE = 64;
const DEFAULT_HOTBAR_SLOTS = 9;
const DEFAULT_SLOT_FEATURES = 2;
const HOTBAR_SLOT_OFFSET = 36;

export interface Snapshot {
  tick: number;
  position: { x: number; y: number; z: number };
  velocity: { x: number; y: number; z: number };
  yaw: number;
  pitch: number;
  health: number;
  food: number;
  oxygen: number;
  experienceLevel: number;
  experienceProgress: number;
  inventory: Record<string, number>;
  hotbar: any[];
  gridMissedTiles?: number;
  gridTopBlockTypes?: Array<[string, number]>;
  gridShape?: GridShape;
  obs?: number[];
}

// Re-export for any downstream module that still imports these from
// observation.js (the canonical home is hash.js, but a deep refactor
// of every caller would expand the diff for zero gain).
export { stableStringHash };

function boolDefault(value: any, fallback: boolean): boolean {
  return typeof value === 'boolean' ? value : fallback;
}

function flatVectorDimFromConfig(config: any): number | null {
  // Optional zero-padding target for the flat (non-grid) vector. Lets
  // operators decouple the bot-emitted shape from the trainer's
  // expected `vector_dim` without inventing semantically-empty
  // features. `null`/omitted ⇒ no padding (legacy 31-float shape).
  const raw = config?.flat_vector_dim;
  return Number.isInteger(raw) && raw > 0 ? raw : null;
}

function inventorySlots(config: any): number {
  const slots = config?.inventory_slots;
  return Number.isInteger(slots) && slots > 0 ? slots : DEFAULT_HOTBAR_SLOTS;
}

function slotFeatures(config: any): number {
  const features = config?.inventory_features_per_slot;
  return Number.isInteger(features) && features > 0 ? features : DEFAULT_SLOT_FEATURES;
}

function flatVectorDimRaw(config: any): number {
  let dim = 0;
  if (boolDefault(config.include_position, true)) dim += 3;
  if (boolDefault(config.include_velocity, true)) dim += 3;
  if (boolDefault(config.include_orientation, true)) dim += 2;
  if (boolDefault(config.include_vitals, true)) dim += 5;
  if (boolDefault(config.include_inventory, true)) {
    dim += inventorySlots(config) * slotFeatures(config);
  }
  return dim;
}

export function computeFlatVectorDim(config: any = {}): number {
  const rawFlat = flatVectorDimRaw(config);
  const padTarget = flatVectorDimFromConfig(config);
  return padTarget !== null ? Math.max(rawFlat, padTarget) : rawFlat;
}

export function computeObsDim(config: any = {}): number {
  const flat = computeFlatVectorDim(config);
  const gridDim = boolDefault(config?.include_block_grid, false)
    ? gridFlatDim(config)
    : 0;
  return gridDim + flat;
}

function itemNumericId(item: any, hashMod: number): number {
  if (!item) return 0;
  if (Number.isFinite(item.type)) return Number(item.type);
  if (typeof item.name === 'string' && item.name.length > 0) {
    return stableStringHash(item.name, hashMod);
  }
  return 0;
}

function getHotbarItem(bot: any, slotIndex: number): any {
  const slots = bot.inventory?.slots;
  if (Array.isArray(slots)) {
    return slots[HOTBAR_SLOT_OFFSET + slotIndex] ?? null;
  }
  return null;
}

function buildInventoryMap(bot: any): Record<string, number> {
  const counts: Record<string, number> = {};
  const items = typeof bot.inventory?.items === 'function'
    ? bot.inventory.items()
    : (bot.inventory?.slots ?? []).filter(Boolean);
  for (const item of items) {
    const name = item.name ?? String(item.type ?? 'unknown');
    counts[name] = (counts[name] ?? 0) + finiteNumber(item.count, 0);
  }
  return counts;
}

function pushPosition(vector: number[], position: any, config: any): void {
  const scale = finiteNumber(config.position_scale, DEFAULT_POSITION_SCALE) || DEFAULT_POSITION_SCALE;
  vector.push(finiteNumber(position.x) / scale);
  vector.push(finiteNumber(position.y) / scale);
  vector.push(finiteNumber(position.z) / scale);
}

function pushVelocity(vector: number[], velocity: any): void {
  vector.push(finiteNumber(velocity.x));
  vector.push(finiteNumber(velocity.y));
  vector.push(finiteNumber(velocity.z));
}

function pushOrientation(vector: number[], snapshot: any): void {
  vector.push(finiteNumber(snapshot.yaw) / Math.PI);
  vector.push(finiteNumber(snapshot.pitch) / Math.PI);
}

function pushVitals(vector: number[], snapshot: any, config: any): void {
  const maxHealth = finiteNumber(config.max_health, 20) || 20;
  const maxFood = finiteNumber(config.max_food, 20) || 20;
  const maxOxygen = finiteNumber(config.max_oxygen, 20) || 20;
  vector.push(finiteNumber(snapshot.health) / maxHealth);
  vector.push(finiteNumber(snapshot.food) / maxFood);
  vector.push(finiteNumber(snapshot.oxygen) / maxOxygen);
  vector.push(finiteNumber(snapshot.experienceLevel));
  vector.push(finiteNumber(snapshot.experienceProgress));
}

function pushInventory(vector: number[], snapshot: Snapshot, config: any): void {
  const hashMod = Number.isInteger(config.hash_mod) && config.hash_mod > 0
    ? config.hash_mod
    : DEFAULT_HASH_MOD;
  const maxStackSize = finiteNumber(config.max_stack_size, DEFAULT_MAX_STACK_SIZE) || DEFAULT_MAX_STACK_SIZE;
  const slots = inventorySlots(config);
  const features = slotFeatures(config);
  for (let slotIndex = 0; slotIndex < slots; slotIndex += 1) {
    const item = snapshot.hotbar[slotIndex] ?? null;
    const idValue = itemNumericId(item, hashMod) / hashMod;
    const countValue = finiteNumber(item?.count, 0) / maxStackSize;
    vector.push(idValue);
    if (features > 1) vector.push(countValue);
    for (let featureIndex = 2; featureIndex < features; featureIndex += 1) {
      vector.push(0);
    }
  }
}

export function observationVectorFromSnapshot(snapshot: Snapshot, config: any = {}): number[] {
  const vector: number[] = [];
  if (boolDefault(config.include_position, true)) pushPosition(vector, snapshot.position, config);
  if (boolDefault(config.include_velocity, true)) pushVelocity(vector, snapshot.velocity);
  if (boolDefault(config.include_orientation, true)) pushOrientation(vector, snapshot);
  if (boolDefault(config.include_vitals, true)) pushVitals(vector, snapshot, config);
  if (boolDefault(config.include_inventory, true)) pushInventory(vector, snapshot, config);
  const targetFlat = computeFlatVectorDim(config);
  if (vector.length > targetFlat) {
    throw new Error(
      `observation length ${vector.length} exceeds computeFlatVectorDim=${targetFlat}`,
    );
  }
  // Zero-pad up to the configured `flat_vector_dim`. This decouples the
  // bot-emitted feature surface from the trainer's expected
  // `vector_dim` without inventing semantically-empty features.
  while (vector.length < targetFlat) vector.push(0);
  return vector;
}

export function snapshotObservation(bot: any, config: any = {}): Snapshot {
  const entity = bot.entity ?? {};
  const position = entity.position ?? bot.position ?? {};
  const velocity = entity.velocity ?? bot.velocity ?? {};
  const hotbar: any[] = [];
  const slots = inventorySlots(config);
  for (let slotIndex = 0; slotIndex < slots; slotIndex += 1) {
    hotbar.push(getHotbarItem(bot, slotIndex));
  }

  const snapshot: Snapshot = {
    // NOTE: only `bot.time.age` is a monotonic tick counter. `bot.time.time`
    // is time-of-day in [0, 24000) and wraps every Minecraft day, so using it
    // as a fallback makes trainer-side sequence ordering jump backwards on
    // long episodes. `bot.tick` is the explicit escape hatch for stubs/tests.
    tick: finiteNumber(bot.time?.age ?? bot.tick, 0),
    position: {
      x: finiteNumber(position.x),
      y: finiteNumber(position.y),
      z: finiteNumber(position.z),
    },
    velocity: {
      x: finiteNumber(velocity.x),
      y: finiteNumber(velocity.y),
      z: finiteNumber(velocity.z),
    },
    yaw: finiteNumber(entity.yaw),
    pitch: finiteNumber(entity.pitch),
    health: finiteNumber(bot.health),
    food: finiteNumber(bot.food),
    oxygen: finiteNumber(bot.oxygenLevel ?? bot.oxygen),
    experienceLevel: finiteNumber(bot.experience?.level),
    experienceProgress: finiteNumber(bot.experience?.progress),
    inventory: buildInventoryMap(bot),
    hotbar,
  };
  const flatVector = observationVectorFromSnapshot(snapshot, config);
  if (boolDefault(config?.include_block_grid, false)) {
    const grid = encodeBlockGrid(bot, config);
    snapshot.gridMissedTiles = grid.missedTiles;
    snapshot.gridTopBlockTypes = grid.topBlockTypes;
    snapshot.gridShape = grid.shape;
    snapshot.obs = [...grid.floats, ...flatVector];
  } else {
    snapshot.obs = flatVector;
  }
  return snapshot;
}