import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import { createLogger, jsonFormatFromEnv, LOG_FORMAT_ENV } from '../src/logger.js';

function captureConsole() {
  const lines: { method: string; arg: unknown }[] = [];
  return {
    sink: {
      log: (arg: unknown) => lines.push({ method: 'log', arg }),
      info: (arg: unknown) => lines.push({ method: 'info', arg }),
      warn: (arg: unknown) => lines.push({ method: 'warn', arg }),
      error: (arg: unknown) => lines.push({ method: 'error', arg }),
      debug: (arg: unknown) => lines.push({ method: 'debug', arg }),
    },
    lines,
  };
}

describe('logger — jsonFormatFromEnv', () => {
  it('returns false when unset', () => {
    assert.equal(jsonFormatFromEnv({} as NodeJS.ProcessEnv), false);
  });
  it('is case-insensitive and trims', () => {
    assert.equal(jsonFormatFromEnv({ [LOG_FORMAT_ENV]: ' JSON ' } as NodeJS.ProcessEnv), true);
    assert.equal(jsonFormatFromEnv({ [LOG_FORMAT_ENV]: 'text' } as NodeJS.ProcessEnv), false);
    assert.equal(jsonFormatFromEnv({ [LOG_FORMAT_ENV]: 'yaml' } as NodeJS.ProcessEnv), false);
  });
});

describe('logger — text mode (default)', () => {
  it('delegates string and object payloads to console verbatim', () => {
    const { sink, lines } = captureConsole();
    const logger = createLogger({ json: false, console: sink });
    logger.info('hello');
    logger.warn({ event: 'oops', code: 3 });
    assert.deepEqual(lines[0], { method: 'info', arg: 'hello' });
    assert.deepEqual(lines[1], { method: 'warn', arg: { event: 'oops', code: 3 } });
  });
});

describe('logger — json mode', () => {
  it('emits one JSON object per line with level + timestamp', () => {
    const { sink, lines } = captureConsole();
    const logger = createLogger({ json: true, console: sink });
    logger.info('listening');
    assert.equal(lines.length, 1);
    assert.equal(lines[0].method, 'log'); // info → stdout
    const parsed = JSON.parse(lines[0].arg as string);
    assert.equal(parsed.level, 'info');
    assert.equal(parsed.message, 'listening');
    assert.equal(typeof parsed.timestamp, 'string');
  });

  it('flattens structured payloads and routes warn/error to stderr', () => {
    const { sink, lines } = captureConsole();
    const logger = createLogger({ json: true, console: sink });
    logger.warn({ event: 'reset_error', error: 'boom' });
    assert.equal(lines[0].method, 'error'); // warn → stderr
    const parsed = JSON.parse(lines[0].arg as string);
    assert.equal(parsed.level, 'warn');
    assert.equal(parsed.event, 'reset_error');
    assert.equal(parsed.error, 'boom');
  });

  it('serializes Error objects with message/stack/name', () => {
    const { sink, lines } = captureConsole();
    const logger = createLogger({ json: true, console: sink });
    logger.error(new Error('kaboom'));
    const parsed = JSON.parse(lines[0].arg as string);
    assert.equal(parsed.level, 'error');
    assert.equal(parsed.message, 'kaboom');
    assert.equal(parsed.name, 'Error');
    assert.equal(typeof parsed.stack, 'string');
  });

  it('does not let payload timestamp/level clobber system metadata', () => {
    const { sink, lines } = captureConsole();
    const logger = createLogger({ json: true, console: sink });
    logger.info({ timestamp: 'spoofed', level: 'fatal', event: 'x' });
    const parsed = JSON.parse(lines[0].arg as string);
    assert.notEqual(parsed.timestamp, 'spoofed');
    assert.equal(parsed.level, 'info');
    assert.equal(parsed.event, 'x');
  });

  it('selects json when FORGE_LOG_FORMAT=json via env', () => {
    const { sink, lines } = captureConsole();
    const logger = createLogger({ env: { [LOG_FORMAT_ENV]: 'json' } as NodeJS.ProcessEnv, console: sink });
    logger.info('x');
    assert.doesNotThrow(() => JSON.parse(lines[0].arg as string));
  });
});
