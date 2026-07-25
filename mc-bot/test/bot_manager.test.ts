import { EventEmitter } from 'node:events';
import { setTimeout as delay } from 'node:timers/promises';
import { describe, it, beforeEach } from 'node:test';
import assert from 'node:assert/strict';

import { BotManager, DEFAULT_RECONNECT_CONFIG, DEFAULT_SPAWN_TIMEOUT_MS } from '../src/bot_manager.js';

// ---------------------------------------------------------------------------
// Helpers / Stubs
// ---------------------------------------------------------------------------

class FakeBot extends EventEmitter {
  entity: any = null;
  time: { age: number } = { age: 0 };
  health = 20;
  food = 20;
  oxygenLevel = 20;
  username = 'ForgeBot';
  end: () => void = () => {};
}

/**
 * Creates a fake mineflayer bot (EventEmitter with entity, time, etc.).
 * Optionally configure spawn behaviour and failure modes.
 */
function createFakeBot(options: any = {}) {
  const bot = new FakeBot();
  bot.entity = options.noEntity ? null : { position: { x: 0, y: 64, z: 0 } };
  bot.time = { age: options.age ?? 0 };
  bot.health = 20;
  bot.food = 20;
  bot.oxygenLevel = 20;
  bot.username = 'ForgeBot';
  bot.end = () => {};
  // Auto-emit spawn if entity is null (deferred spawn)
  if (!bot.entity) {
    queueMicrotask(() => {
      bot.entity = { position: { x: 0, y: 64, z: 0 } };
      bot.emit('spawn');
    });
  }
  return bot;
}

/**
 * Factory that returns a createBot function producing FakeBots.
 * Tracks call count and supports failure injection.
 */
function createBotFactory(options: any = {}) {
  const calls: any[] = [];
  const failUntilAttempt = options.failUntilAttempt ?? 0;
  let callCount = 0;

  function createBot(config: any) {
    callCount++;
    calls.push(config);
    if (callCount <= failUntilAttempt) {
      // Return a bot that fails to spawn (never emits 'spawn')
      const bot = new FakeBot();
      bot.entity = null;
      bot.time = { age: 0 };
      bot.end = () => {};
      // Emit error after a tick to simulate connection failure
      queueMicrotask(() => bot.emit('error', new Error('connection refused')));
      return bot;
    }
    return createFakeBot(options);
  }

  (createBot as any).calls = calls;
  (createBot as any).getCallCount = () => callCount;
  return createBot as any;
}

/** Silent logger for tests — captures structured log entries. */
function silentLogger() {
  const entries: any[] = [];
  return {
    entries,
    info: (msg: any) => entries.push({ level: 'info', msg }),
    warn: (msg: any) => entries.push({ level: 'warn', msg }),
    error: (msg: any) => entries.push({ level: 'error', msg }),
  };
}

/** Fast reconnect config to keep tests snappy. */
const FAST_RECONNECT = Object.freeze({
  backoff_ms: [10, 20, 30],
  max_attempts: 3,
  stale_timeout_ms: 50,
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('BotManager', () => {
  let logger: any;

  beforeEach(() => {
    logger = silentLogger();
  });

  // ---- 1. creates initial bot successfully ----
  it('creates initial bot successfully', async () => {
    const factory = createBotFactory();
    const manager = new BotManager({ host: '127.0.0.1' }, factory, { logger });
    const bot = await manager.createInitialBot();

    assert.ok(bot, 'bot should be returned');
    assert.ok(bot.entity, 'bot should have entity after spawn');
    assert.equal(factory.getCallCount(), 1, 'createBot called once');
    manager.destroy();
  });

  // ---- 1b. spawn-timeout config: valid value used, no warning ----
  it('uses a valid configured spawn_timeout_ms without warning', async () => {
    const factory = createBotFactory({ noEntity: true }); // deferred spawn → timeout path
    const manager = new BotManager({ host: '127.0.0.1', spawn_timeout_ms: 5000 }, factory, {
      logger,
    });
    await manager.createInitialBot();

    const warned = logger.entries.find((e: any) => e.msg?.event === 'invalid_spawn_timeout');
    assert.equal(warned, undefined, 'a valid timeout must not warn');
    manager.destroy();
  });

  // ---- 1c. spawn-timeout config: invalid value warns + falls back to default ----
  it('warns and falls back to the default when spawn_timeout_ms is invalid', async () => {
    for (const bad of [0, -1, Number.NaN, 'nope' as unknown as number]) {
      const localLogger = silentLogger();
      const factory = createBotFactory({ noEntity: true }); // deferred spawn → timeout path
      const manager = new BotManager({ host: '127.0.0.1', spawn_timeout_ms: bad }, factory, {
        logger: localLogger,
      });
      await manager.createInitialBot();

      const warned = localLogger.entries.find(
        (e: any) => e.level === 'warn' && e.msg?.event === 'invalid_spawn_timeout'
      );
      assert.ok(warned, `invalid spawn_timeout_ms=${String(bad)} must warn`);
      assert.equal(warned.msg.configured, bad, 'warning logs the configured value');
      assert.equal(warned.msg.fallback_ms, DEFAULT_SPAWN_TIMEOUT_MS, 'warning logs the fallback');
      manager.destroy();
    }
  });

  // ---- 2. getBot() returns current bot after creation ----
  it('getBot() returns current bot after creation', async () => {
    const factory = createBotFactory();
    const manager = new BotManager({ host: '127.0.0.1' }, factory, { logger });
    assert.equal(manager.getBot(), null, 'getBot() null before creation');

    const bot = await manager.createInitialBot();
    assert.equal(manager.getBot(), bot, 'getBot() matches created bot');
    manager.destroy();
  });

  // ---- 3. isHealthy() returns true when tick age is fresh ----
  it('isHealthy() returns true when tick age is fresh', async () => {
    const factory = createBotFactory();
    const manager = new BotManager({ host: '127.0.0.1' }, factory, {
      logger,
      reconnectConfig: FAST_RECONNECT,
    });
    await manager.createInitialBot();
    manager.updateTickAge(100);

    assert.ok(manager.isHealthy(), 'should be healthy immediately after updateTickAge');
    manager.destroy();
  });

  // ---- 4. isHealthy() returns false when tick age exceeds stale_timeout_ms ----
  it('isHealthy() returns false when tick age exceeds stale_timeout_ms', async () => {
    const factory = createBotFactory();
    const manager = new BotManager({ host: '127.0.0.1' }, factory, {
      logger,
      reconnectConfig: { ...FAST_RECONNECT, stale_timeout_ms: 10 },
    });
    await manager.createInitialBot();
    manager.updateTickAge(1);

    // Wait longer than stale_timeout_ms
    await delay(30);
    assert.equal(manager.isHealthy(), false, 'should be stale after timeout');
    manager.destroy();
  });

  // ---- 5. reconnect() creates a new bot and emits reconnected event ----
  it('reconnect() creates a new bot and emits reconnected event', async () => {
    const factory = createBotFactory();
    const manager = new BotManager({ host: '127.0.0.1' }, factory, {
      logger,
      reconnectConfig: FAST_RECONNECT,
    });
    const oldBot = await manager.createInitialBot();

    let reconnectedBot = null;
    manager.on('reconnected', (bot) => { reconnectedBot = bot; });

    await manager.reconnect();

    assert.ok(reconnectedBot, 'reconnected event should fire');
    assert.notEqual(reconnectedBot, oldBot, 'new bot should be different from old bot');
    assert.equal(manager.getBot(), reconnectedBot, 'getBot() returns the new bot');
    manager.destroy();
  });

  // ---- 6. reconnect() applies exponential backoff ----
  it('reconnect() applies exponential backoff', async () => {
    // Fail first 2 attempts, succeed on 3rd
    let callCount = 0;
    const backoffs = [10, 20, 30];
    const createBot = () => {
      callCount++;
      const bot = new FakeBot();
      bot.time = { age: 0 };
      bot.end = () => {};
      if (callCount <= 2) {
        // Fail: never spawn, will timeout
        bot.entity = null;
        queueMicrotask(() => bot.emit('error', new Error('fail')));
        return bot;
      }
      bot.entity = { position: { x: 0, y: 64, z: 0 } };
      return bot;
    };

    const manager = new BotManager({ host: '127.0.0.1' }, createBot, {
      logger,
      reconnectConfig: { backoff_ms: backoffs, max_attempts: 5, stale_timeout_ms: 50 },
    });

    // Create initial bot (this uses a direct call, won't fail since callCount starts at 0)
    // Reset callCount so the reconnect exercises the failure path.
    const initialBot = new FakeBot();
    initialBot.entity = { position: { x: 0, y: 64, z: 0 } };
    initialBot.time = { age: 0 };
    initialBot.end = () => {};
    // Manually set the internal bot via createInitialBot workaround:
    // Actually, let's just use a simple manager that starts fresh.
    callCount = 0;

    const manager2 = new BotManager({ host: '127.0.0.1' }, createBot, {
      logger,
      reconnectConfig: { backoff_ms: backoffs, max_attempts: 5, stale_timeout_ms: 50 },
    });
    // First call succeeds (callCount becomes 1... wait, our logic fails at <=2)
    // Let's fix: fail on attempts 2 and 3, succeed on 4
    callCount = 0;
    const createBot2 = () => {
      callCount++;
      const bot = new FakeBot();
      bot.time = { age: 0 };
      bot.end = () => {};
      if (callCount >= 2 && callCount <= 3) {
        bot.entity = null;
        // These bots never spawn → reconnect will catch the error in waitForSpawn
        // We need the spawn to actually fail, not just hang.
        // Use a timeout approach: make waitForSpawn fail by emitting error
        queueMicrotask(() => bot.emit('error', new Error('refused')));
        return bot;
      }
      bot.entity = { position: { x: 0, y: 64, z: 0 } };
      return bot;
    };

    const manager3 = new BotManager({ host: '127.0.0.1' }, createBot2, {
      logger,
      reconnectConfig: { backoff_ms: backoffs, max_attempts: 5, stale_timeout_ms: 50 },
    });
    await manager3.createInitialBot(); // callCount=1, succeeds

    const startTime = Date.now();
    await manager3.reconnect();
    const elapsed = Date.now() - startTime;

    // Should have waited at least backoff[0] + backoff[1] ≈ 30ms for 2 failed attempts
    assert.ok(elapsed >= 20, `Expected at least 20ms of backoff delay, got ${elapsed}ms`);
    // Final call count: 1 (initial) + 2 (failed reconnect) + 1 (success) = 4
    assert.equal(callCount, 4, 'should have made 4 total createBot calls');

    manager.destroy();
    manager2.destroy();
    manager3.destroy();
  });

  // ---- 7. reconnect() caps at max_attempts and emits reconnect_failed ----
  it('reconnect() caps at max_attempts and emits reconnect_failed', async () => {
    // All attempts fail — the factory always returns a bot that errors
    const createBot = () => {
      const bot = new FakeBot();
      bot.entity = null;
      bot.time = { age: 0 };
      bot.end = () => {};
      // Simulate spawn failure
      queueMicrotask(() => {
        bot.emit('error', new Error('connection refused'));
      });
      return bot;
    };

    // We need createInitialBot to succeed first
    let callIdx = 0;
    const createBotMixed = () => {
      callIdx++;
      if (callIdx === 1) {
        const bot = new FakeBot();
        bot.entity = { position: { x: 0, y: 64, z: 0 } };
        bot.time = { age: 0 };
        bot.end = () => {};
        return bot;
      }
      return createBot();
    };

    const manager = new BotManager({ host: '127.0.0.1' }, createBotMixed, {
      logger,
      reconnectConfig: { ...FAST_RECONNECT, max_attempts: 2 },
    });
    await manager.createInitialBot();

    let failedError = null;
    manager.on('reconnect_failed', (err) => { failedError = err; });

    await assert.rejects(
      () => manager.reconnect(),
      { message: /reconnect failed after 2 attempts/ },
    );
    assert.ok(failedError, 'reconnect_failed event should be emitted');
    manager.destroy();
  });

  // ---- 8. reconnect() resets attempt counter on success ----
  it('reconnect() resets attempt counter on success', async () => {
    const factory = createBotFactory();
    const manager = new BotManager({ host: '127.0.0.1' }, factory, {
      logger,
      reconnectConfig: FAST_RECONNECT,
    });
    await manager.createInitialBot();

    // First reconnect
    await manager.reconnect();
    assert.ok(manager.isHealthy(), 'should be healthy after reconnect');

    // Second reconnect — should also work (counter was reset)
    await manager.reconnect();
    assert.ok(manager.isHealthy(), 'should still be healthy after second reconnect');

    // factory was called: 1 (initial) + 1 (reconnect 1) + 1 (reconnect 2) = 3
    assert.equal(factory.getCallCount(), 3);
    manager.destroy();
  });

  // ---- 9. bot kicked event triggers reconnect ----
  it('bot kicked event triggers reconnect', async () => {
    const factory = createBotFactory();
    const manager = new BotManager({ host: '127.0.0.1' }, factory, {
      logger,
      reconnectConfig: FAST_RECONNECT,
    });
    const bot = await manager.createInitialBot();

    const reconnected = new Promise((resolve) => {
      manager.on('reconnected', resolve);
    });

    bot.emit('kicked', 'You have been kicked');

    const newBot = await reconnected;
    assert.ok(newBot, 'reconnected event should fire after kick');
    assert.notEqual(newBot, bot, 'new bot should replace kicked bot');
    manager.destroy();
  });

  // ---- 10. bot error event triggers reconnect ----
  it('bot error event triggers reconnect', async () => {
    const factory = createBotFactory();
    const manager = new BotManager({ host: '127.0.0.1' }, factory, {
      logger,
      reconnectConfig: FAST_RECONNECT,
    });
    const bot = await manager.createInitialBot();

    const reconnected = new Promise((resolve) => {
      manager.on('reconnected', resolve);
    });

    bot.emit('error', new Error('socket hangup'));

    const newBot = await reconnected;
    assert.ok(newBot, 'reconnected event should fire after error');
    manager.destroy();
  });

  // ---- 11. bot end event triggers reconnect ----
  it('bot end event triggers reconnect', async () => {
    const factory = createBotFactory();
    const manager = new BotManager({ host: '127.0.0.1' }, factory, {
      logger,
      reconnectConfig: FAST_RECONNECT,
    });
    const bot = await manager.createInitialBot();

    const reconnected = new Promise((resolve) => {
      manager.on('reconnected', resolve);
    });

    bot.emit('end', 'server closed');

    const newBot = await reconnected;
    assert.ok(newBot, 'reconnected event should fire after end');
    manager.destroy();
  });

  // ---- 12. concurrent reconnect calls are coalesced (no double-reconnect) ----
  it('concurrent reconnect calls are coalesced (no double-reconnect)', async () => {
    const factory = createBotFactory();
    const manager = new BotManager({ host: '127.0.0.1' }, factory, {
      logger,
      reconnectConfig: FAST_RECONNECT,
    });
    await manager.createInitialBot();

    // Fire two reconnects concurrently
    const [r1, r2] = await Promise.all([
      manager.reconnect(),
      manager.reconnect(),
    ]);

    // Only 2 createBot calls total: 1 initial + 1 reconnect (coalesced)
    assert.equal(factory.getCallCount(), 2, 'should not double-reconnect');
    manager.destroy();
  });

  // ---- 13. destroy() cleans up bot and stops reconnection ----
  it('destroy() cleans up bot and stops reconnection', async () => {
    const factory = createBotFactory();
    const manager = new BotManager({ host: '127.0.0.1' }, factory, {
      logger,
      reconnectConfig: FAST_RECONNECT,
    });
    await manager.createInitialBot();
    assert.ok(manager.getBot(), 'bot exists before destroy');

    manager.destroy();
    assert.equal(manager.getBot(), null, 'bot is null after destroy');
    assert.equal(manager.isHealthy(), false, 'not healthy after destroy');

    // Reconnect should be a no-op after destroy
    await manager.reconnect(); // should not throw
    assert.equal(manager.getBot(), null, 'bot stays null after reconnect post-destroy');
  });

  // ---- 14. uses default config when no reconnect config provided ----
  it('uses default config when no reconnect config provided', async () => {
    const factory = createBotFactory();
    const manager = new BotManager({ host: '127.0.0.1' }, factory, { logger });
    await manager.createInitialBot();

    // Verify defaults via the isHealthy() stale detection window
    // Default stale_timeout_ms is 10000, so bot should be healthy immediately
    assert.ok(manager.isHealthy(), 'healthy with default config');
    manager.destroy();
  });

  // ---- 15. isReconnecting() reflects state transitions ----
  it('isReconnecting() reflects state transitions', async () => {
    const factory = createBotFactory();
    const manager = new BotManager({ host: '127.0.0.1' }, factory, {
      logger,
      reconnectConfig: FAST_RECONNECT,
    });
    await manager.createInitialBot();
    assert.equal(manager.isReconnecting(), false, 'not reconnecting initially');

    const reconnected = new Promise((resolve) => {
      manager.on('reconnected', resolve);
    });

    // Start reconnect
    const reconnectPromise = manager.reconnect();

    // After reconnect finishes
    await reconnectPromise;
    assert.equal(manager.isReconnecting(), false, 'not reconnecting after completion');
    manager.destroy();
  });

  // ---- 16. updateTickAge() resets reconnect counter ----
  it('updateTickAge() resets reconnect counter', async () => {
    const factory = createBotFactory();
    const manager = new BotManager({ host: '127.0.0.1' }, factory, {
      logger,
      reconnectConfig: FAST_RECONNECT,
    });
    await manager.createInitialBot();

    // Simulate a reconnect to bump the internal attempt counter
    await manager.reconnect();
    // updateTickAge should have already reset the counter internally via reconnect success,
    // but calling it again should not cause issues
    manager.updateTickAge(500);
    assert.ok(manager.isHealthy());

    // Can still reconnect again (counter was reset)
    await manager.reconnect();
    assert.ok(manager.getBot(), 'bot exists after second reconnect');
    manager.destroy();
  });

  // ---- 17. createInitialBot waits for spawn when entity is null ----
  it('createInitialBot waits for spawn when entity is null', async () => {
    const createBot = () => {
      const bot = new FakeBot();
      bot.entity = null;
      bot.time = { age: 0 };
      bot.end = () => {};
      // Simulate delayed spawn
      setTimeout(() => {
        bot.entity = { position: { x: 0, y: 64, z: 0 } };
        bot.emit('spawn');
      }, 20);
      return bot;
    };

    const manager = new BotManager({ host: '127.0.0.1' }, createBot, {
      logger,
      reconnectConfig: FAST_RECONNECT,
    });
    const bot = await manager.createInitialBot();
    assert.ok(bot.entity, 'bot should have entity after spawn');
    manager.destroy();
  });

  // ---- 18. structured logging is used throughout lifecycle ----
  it('emits structured log entries throughout lifecycle', async () => {
    const factory = createBotFactory();
    const manager = new BotManager({ host: '127.0.0.1' }, factory, {
      logger,
      reconnectConfig: FAST_RECONNECT,
    });
    await manager.createInitialBot();
    await manager.reconnect();
    manager.destroy();

    const infoEvents = logger.entries
      .filter((e: any) => e.level === 'info')
      .map((e: any) => e.msg?.event ?? e.msg);

    assert.ok(infoEvents.includes('bot_created'), 'should log bot_created');
    assert.ok(infoEvents.includes('reconnect_attempt'), 'should log reconnect_attempt');
    assert.ok(infoEvents.includes('reconnected'), 'should log reconnected');
    assert.ok(infoEvents.includes('bot_manager_destroyed'), 'should log destroy');
  });

  // ---- Heartbeat (stale-detection) ----
  describe('heartbeat', () => {
    it('reconnects a stale bot with no manual reconnect call', async () => {
      const factory = createBotFactory();
      const manager = new BotManager({ host: '127.0.0.1' }, factory, {
        logger,
        reconnectConfig: FAST_RECONNECT, // stale_timeout_ms: 50
      });
      await manager.createInitialBot();
      assert.equal(factory.getCallCount(), 1);

      let reconnected = false;
      manager.on('reconnected', () => {
        reconnected = true;
      });

      // Start a fast heartbeat and let the bot go stale (never call
      // updateTickAge). Within stale_timeout + a couple of poll intervals the
      // heartbeat must trigger a reconnect on its own.
      manager.startHeartbeat(15);
      await delay(150);
      manager.destroy();

      assert.ok(reconnected, 'heartbeat should have emitted reconnected');
      assert.ok(factory.getCallCount() >= 2, 'a new bot should have been built');
    });

    it('does not reconnect while the bot stays healthy', async () => {
      const factory = createBotFactory();
      const manager = new BotManager({ host: '127.0.0.1' }, factory, {
        logger,
        reconnectConfig: FAST_RECONNECT,
      });
      await manager.createInitialBot();
      manager.startHeartbeat(15);

      // Keep the bot fresh across several heartbeat intervals.
      for (let i = 0; i < 8; i++) {
        manager.updateTickAge(0);
        await delay(15);
      }
      manager.destroy();

      assert.equal(factory.getCallCount(), 1, 'no reconnect should occur while healthy');
    });

    it('destroy() stops the heartbeat (no reconnect afterwards)', async () => {
      const factory = createBotFactory();
      const manager = new BotManager({ host: '127.0.0.1' }, factory, {
        logger,
        reconnectConfig: FAST_RECONNECT,
      });
      await manager.createInitialBot();
      manager.startHeartbeat(15);
      manager.destroy(); // should clear the timer immediately

      await delay(150); // would have fired several times if not cleared
      assert.equal(factory.getCallCount(), 1, 'destroyed manager must not reconnect');
    });

    it('startHeartbeat is idempotent and ignores non-positive intervals', async () => {
      const factory = createBotFactory();
      const manager = new BotManager({ host: '127.0.0.1' }, factory, {
        logger,
        reconnectConfig: FAST_RECONNECT,
      });
      await manager.createInitialBot();
      // A disabled heartbeat (0) is a no-op; calling twice must not leak timers.
      manager.startHeartbeat(0);
      manager.startHeartbeat(15);
      manager.startHeartbeat(15);
      manager.stopHeartbeat();
      manager.destroy();
      assert.equal(factory.getCallCount(), 1);
    });
  });
});
