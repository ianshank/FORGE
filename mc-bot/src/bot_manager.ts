import { EventEmitter } from 'node:events';
import { setTimeout as delay } from 'node:timers/promises';

/**
 * Default reconnection configuration.
 * All values are overridable via the `[reconnect]` table in env.toml.
 */
export const DEFAULT_RECONNECT_CONFIG = Object.freeze({
  backoff_ms: Object.freeze([500, 1000, 2000, 4000, 8000]),
  max_attempts: 5,
  stale_timeout_ms: 10_000,
});

export interface ReconnectConfig {
  backoff_ms?: number[];
  max_attempts?: number;
  stale_timeout_ms?: number;
}

/**
 * Manages the mineflayer bot lifecycle — creation, health-checking,
 * teardown, and auto-reconnection with exponential backoff.
 *
 * Events emitted:
 *  - `reconnected`      (bot)   — new bot is connected and spawned
 *  - `reconnect_failed` (error) — all retry attempts exhausted
 */
export class BotManager extends EventEmitter {
  #botConfig: any;
  #createBot: (config: any) => any;
  #bot: any = null;
  #lastTickTime = 0;
  #reconnectAttempts = 0;
  #reconnecting = false;
  #reconnectPromise: Promise<void> | null = null;
  #destroyed = false;
  #reconnectConfig: { backoff_ms: number[]; max_attempts: number; stale_timeout_ms: number };
  #logger: any;
  #heartbeatTimer: ReturnType<typeof setInterval> | null = null;

  constructor(
    botConfig: any,
    createBot: (config: any) => any,
    options: { reconnectConfig?: ReconnectConfig; logger?: any } = {}
  ) {
    super();
    this.#botConfig = botConfig;
    this.#createBot = createBot;
    this.#logger = options.logger ?? console;
    const rc = options.reconnectConfig ?? {};
    this.#reconnectConfig = {
      backoff_ms: Array.isArray(rc.backoff_ms) ? [...rc.backoff_ms] : [...DEFAULT_RECONNECT_CONFIG.backoff_ms],
      max_attempts: rc.max_attempts ?? DEFAULT_RECONNECT_CONFIG.max_attempts,
      stale_timeout_ms: rc.stale_timeout_ms ?? DEFAULT_RECONNECT_CONFIG.stale_timeout_ms,
    };
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /** Create the initial bot, wire events, and wait for spawn. */
  async createInitialBot(): Promise<any> {
    this.#bot = this.#buildBot();
    this.#wireEvents(this.#bot);
    await this.#waitForSpawn(this.#bot);
    this.#lastTickTime = Date.now();
    this.#logger.info?.({ event: 'bot_created', username: this.#botConfig.username });
    return this.#bot;
  }

  /** @returns the current mineflayer bot instance (may be null during reconnect) */
  getBot(): any {
    return this.#bot;
  }

  /** @returns {boolean} true when the bot's tick age was updated recently */
  isHealthy(): boolean {
    if (!this.#bot || this.#reconnecting) return false;
    return (Date.now() - this.#lastTickTime) < this.#reconnectConfig.stale_timeout_ms;
  }

  /** @returns {boolean} true while a reconnect cycle is in progress */
  isReconnecting(): boolean {
    return this.#reconnecting;
  }

  /**
   * Called by the observation layer each time a successful snapshot is taken.
   * Resets reconnect counter and refreshes the stale-detection clock.
   *
   * @param {number} _age  bot.time.age (unused but kept for future metrics)
   */
  updateTickAge(_age: number): void {
    this.#lastTickTime = Date.now();
    this.#reconnectAttempts = 0;
  }

  /**
   * Trigger a reconnect cycle. Concurrent calls are coalesced — only one
   * reconnect loop runs at a time.
   *
   * @returns {Promise<void>} resolves when reconnected (or rejects on failure)
   */
  async reconnect(): Promise<void> {
    if (this.#destroyed) return;
    if (this.#reconnectPromise) return this.#reconnectPromise;

    this.#reconnectPromise = this.#doReconnect();
    try {
      await this.#reconnectPromise;
    } finally {
      this.#reconnectPromise = null;
    }
  }

  /**
   * Start a background heartbeat that polls {@link isHealthy} every
   * `intervalMs` and triggers a {@link reconnect} when the bot has gone stale.
   *
   * This closes the half-open-socket gap: when the Minecraft server stops
   * sending ticks but the TCP socket stays open, mineflayer emits no
   * `error`/`end`, so nothing would otherwise call `reconnect()`. The monitor
   * reuses the existing (coalesced) `reconnect()` and the existing health
   * check — no new reconnection logic. Idempotent: a prior timer is cleared
   * first. A non-positive `intervalMs` disables the heartbeat.
   *
   * @param intervalMs poll interval in milliseconds (e.g. `env.heartbeat_ms`).
   */
  startHeartbeat(intervalMs: number): void {
    this.stopHeartbeat();
    if (this.#destroyed || !Number.isFinite(intervalMs) || intervalMs <= 0) {
      return;
    }
    this.#heartbeatTimer = setInterval(() => {
      if (this.#destroyed || this.#reconnecting || this.isHealthy()) {
        return;
      }
      this.#logger.warn?.({ event: 'heartbeat_stale', msg: 'bot stale, triggering reconnect' });
      this.reconnect().catch((err: unknown) => {
        const message = err instanceof Error ? err.message : String(err);
        this.#logger.warn?.({ event: 'heartbeat_reconnect_error', error: message });
      });
    }, intervalMs);
    // Don't let the heartbeat keep the Node process alive on its own.
    this.#heartbeatTimer.unref?.();
  }

  /** Stop the background heartbeat if running. */
  stopHeartbeat(): void {
    if (this.#heartbeatTimer !== null) {
      clearInterval(this.#heartbeatTimer);
      this.#heartbeatTimer = null;
    }
  }

  /** Tear down the current bot and stop accepting reconnection attempts. */
  destroy(): void {
    this.#destroyed = true;
    this.#reconnecting = false;
    this.#reconnectPromise = null;
    this.stopHeartbeat();
    this.#teardownBot();
    this.#logger.info?.({ event: 'bot_manager_destroyed' });
  }

  // ---------------------------------------------------------------------------
  // Internals
  // ---------------------------------------------------------------------------

  /** @returns a fresh mineflayer bot */
  #buildBot(): any {
    return this.#createBot(this.#botConfig);
  }

  /** Remove all listeners and clean up the old bot. */
  #teardownBot(): void {
    if (this.#bot) {
      try {
        this.#bot.removeAllListeners?.();
        this.#bot.end?.();
      } catch {
        /* best-effort cleanup */
      }
      this.#bot = null;
    }
  }

  /** Wire kicked / error / end events so they trigger auto-reconnect. */
  #wireEvents(bot: any): void {
    const onDisconnect = (reason: any) => {
      if (this.#destroyed) return;
      this.#logger.warn?.({ event: 'bot_disconnected', reason: String(reason) });
      this.reconnect().catch((err: any) => {
        this.#logger.error?.({ event: 'reconnect_unhandled_error', error: err.message });
      });
    };

    bot.on('kicked', (reason: any) => onDisconnect(`kicked: ${reason}`));
    bot.on('error', (err: any) => onDisconnect(`error: ${err?.message ?? err}`));
    bot.on('end', (reason: any) => onDisconnect(`end: ${reason ?? 'unknown'}`));
  }

  /** Wait for the bot's `spawn` event (or resolve immediately if already spawned). */
  #waitForSpawn(bot: any): Promise<void> {
    if (bot.entity) return Promise.resolve();
    return new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => {
        cleanup();
        reject(new Error('Spawn timeout: bot failed to spawn within 30000ms'));
      }, 30000);

      const onSpawn = () => {
        cleanup();
        resolve();
      };
      const onError = (err: any) => {
        cleanup();
        reject(err instanceof Error ? err : new Error(String(err)));
      };
      const onKicked = (reason: any) => {
        cleanup();
        reject(new Error(`Kicked during spawn: ${reason}`));
      };
      const onEnd = (reason: any) => {
        cleanup();
        reject(new Error(`Connection ended during spawn: ${reason ?? 'unknown'}`));
      };

      const cleanup = () => {
        clearTimeout(timeout);
        bot.removeListener('spawn', onSpawn);
        bot.removeListener('error', onError);
        bot.removeListener('kicked', onKicked);
        bot.removeListener('end', onEnd);
      };

      bot.once('spawn', onSpawn);
      bot.once('error', onError);
      bot.once('kicked', onKicked);
      bot.once('end', onEnd);
    });
  }

  /** Core reconnect loop with exponential backoff. */
  async #doReconnect(): Promise<void> {
    if (this.#reconnecting) return;
    this.#reconnecting = true;

    try {
      while (this.#reconnectAttempts < this.#reconnectConfig.max_attempts) {
        if (this.#destroyed) return;

        const attempt = this.#reconnectAttempts;
        const backoffIndex = Math.min(attempt, this.#reconnectConfig.backoff_ms.length - 1);
        const backoffMs = this.#reconnectConfig.backoff_ms[backoffIndex];

        this.#logger.info?.({
          event: 'reconnect_attempt',
          attempt: attempt + 1,
          max: this.#reconnectConfig.max_attempts,
          backoff_ms: backoffMs,
        });

        await delay(backoffMs);
        if (this.#destroyed) return;

        this.#teardownBot();

        try {
          this.#bot = this.#buildBot();
          this.#wireEvents(this.#bot);
          await this.#waitForSpawn(this.#bot);
          this.#lastTickTime = Date.now();
          this.#reconnectAttempts = 0;
          this.#reconnecting = false;
          this.#logger.info?.({ event: 'reconnected', attempt: attempt + 1 });
          this.emit('reconnected', this.#bot);
          return;
        } catch (err: any) {
          this.#reconnectAttempts += 1;
          this.#logger.warn?.({
            event: 'reconnect_attempt_failed',
            attempt: attempt + 1,
            error: err?.message ?? String(err),
          });
        }
      }

      // Exhausted all attempts
      this.#reconnecting = false;
      const failError = new Error(
        `reconnect failed after ${this.#reconnectConfig.max_attempts} attempts`,
      );
      this.#logger.error?.({ event: 'reconnect_failed', max_attempts: this.#reconnectConfig.max_attempts });
      this.emit('reconnect_failed', failError);
      throw failError;
    } catch (err) {
      this.#reconnecting = false;
      throw err;
    }
  }
}
