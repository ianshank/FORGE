export async function startViewer(bot, config = {}, options = {}) {
  if (!config.enabled) {
    return null;
  }
  const viewerModule = options.viewerModule ?? await import('prismarine-viewer');
  const start = viewerModule.mineflayer ?? viewerModule.default?.mineflayer ?? viewerModule.default;
  if (typeof start !== 'function') {
    throw new Error('prismarine-viewer does not expose a mineflayer viewer function');
  }
  return start(bot, {
    port: config.port,
    host: config.host,
    firstPerson: config.first_person,
    viewDistance: config.view_distance,
  });
}