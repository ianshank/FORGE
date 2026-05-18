const DEFAULT_POSITION_SCALE = 1024;
const DEFAULT_HASH_MOD = 4096;
const DEFAULT_MAX_STACK_SIZE = 64;
const DEFAULT_HOTBAR_SLOTS = 9;
const DEFAULT_SLOT_FEATURES = 2;
const HOTBAR_SLOT_OFFSET = 36;

function finiteNumber(value, fallback = 0) {
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue : fallback;
}

function boolDefault(value, fallback) {
  return typeof value === 'boolean' ? value : fallback;
}

function inventorySlots(config) {
  const slots = config?.inventory_slots;
  return Number.isInteger(slots) && slots > 0 ? slots : DEFAULT_HOTBAR_SLOTS;
}

function slotFeatures(config) {
  const features = config?.inventory_features_per_slot;
  return Number.isInteger(features) && features > 0 ? features : DEFAULT_SLOT_FEATURES;
}

export function computeObsDim(config = {}) {
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

export function stableStringHash(text, modulus = DEFAULT_HASH_MOD) {
  const hashMod = Number.isInteger(modulus) && modulus > 0 ? modulus : DEFAULT_HASH_MOD;
  let hash = 2166136261;
  for (const char of String(text)) {
    hash ^= char.charCodeAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0) % hashMod;
}

function itemNumericId(item, hashMod) {
  if (!item) return 0;
  if (Number.isFinite(item.type)) return Number(item.type);
  if (typeof item.name === 'string' && item.name.length > 0) {
    return stableStringHash(item.name, hashMod);
  }
  return 0;
}

function getHotbarItem(bot, slotIndex) {
  const slots = bot.inventory?.slots;
  if (Array.isArray(slots)) {
    return slots[HOTBAR_SLOT_OFFSET + slotIndex] ?? null;
  }
  return null;
}

function buildInventoryMap(bot) {
  const counts = {};
  const items = typeof bot.inventory?.items === 'function'
    ? bot.inventory.items()
    : (bot.inventory?.slots ?? []).filter(Boolean);
  for (const item of items) {
    const name = item.name ?? String(item.type ?? 'unknown');
    counts[name] = (counts[name] ?? 0) + finiteNumber(item.count, 0);
  }
  return counts;
}

function pushPosition(vector, position, config) {
  const scale = finiteNumber(config.position_scale, DEFAULT_POSITION_SCALE) || DEFAULT_POSITION_SCALE;
  vector.push(finiteNumber(position.x) / scale);
  vector.push(finiteNumber(position.y) / scale);
  vector.push(finiteNumber(position.z) / scale);
}

function pushVelocity(vector, velocity) {
  vector.push(finiteNumber(velocity.x));
  vector.push(finiteNumber(velocity.y));
  vector.push(finiteNumber(velocity.z));
}

function pushOrientation(vector, snapshot) {
  vector.push(finiteNumber(snapshot.yaw) / Math.PI);
  vector.push(finiteNumber(snapshot.pitch) / Math.PI);
}

function pushVitals(vector, snapshot, config) {
  const maxHealth = finiteNumber(config.max_health, 20) || 20;
  const maxFood = finiteNumber(config.max_food, 20) || 20;
  const maxOxygen = finiteNumber(config.max_oxygen, 20) || 20;
  vector.push(finiteNumber(snapshot.health) / maxHealth);
  vector.push(finiteNumber(snapshot.food) / maxFood);
  vector.push(finiteNumber(snapshot.oxygen) / maxOxygen);
  vector.push(finiteNumber(snapshot.experienceLevel));
  vector.push(finiteNumber(snapshot.experienceProgress));
}

function pushInventory(vector, snapshot, config) {
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

export function observationVectorFromSnapshot(snapshot, config = {}) {
  const vector = [];
  if (boolDefault(config.include_position, true)) pushPosition(vector, snapshot.position, config);
  if (boolDefault(config.include_velocity, true)) pushVelocity(vector, snapshot.velocity);
  if (boolDefault(config.include_orientation, true)) pushOrientation(vector, snapshot);
  if (boolDefault(config.include_vitals, true)) pushVitals(vector, snapshot, config);
  if (boolDefault(config.include_inventory, true)) pushInventory(vector, snapshot, config);
  const expectedDim = computeObsDim(config);
  if (vector.length !== expectedDim) {
    throw new Error(`observation length ${vector.length} did not match computed dim ${expectedDim}`);
  }
  return vector;
}

export function snapshotObservation(bot, config = {}) {
  const entity = bot.entity ?? {};
  const position = entity.position ?? bot.position ?? {};
  const velocity = entity.velocity ?? bot.velocity ?? {};
  const hotbar = [];
  const slots = inventorySlots(config);
  for (let slotIndex = 0; slotIndex < slots; slotIndex += 1) {
    hotbar.push(getHotbarItem(bot, slotIndex));
  }

  const snapshot = {
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
  snapshot.obs = observationVectorFromSnapshot(snapshot, config);
  return snapshot;
}