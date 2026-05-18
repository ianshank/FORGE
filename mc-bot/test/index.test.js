import { EventEmitter } from 'node:events';
import { setTimeout as delay } from 'node:timers/promises';
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import { buildActionMap } from '../src/action_map.js';
import { buildConfigBundleFromObjects } from '../src/config.js';
import { createConnectionHandler, validateObservationConfig } from '../src/index.js';

class FakeSocket extends EventEmitter {
  constructor() {
    super();
    this.OPEN = 1;
    this.CLOSED = 3;
    this.readyState = this.OPEN;
    this.sent = [];
  }

  send(text) {
    this.sent.push(JSON.parse(text));
  }

  close() {
    this.readyState = this.CLOSED;
    this.emit('close');
  }
}

function stubBot() {
  return {
    username: 'ForgeBot',
    chats: [],
    waits: [],
    controls: [],
    time: { age: 0 },
    entity: {
      position: { x: 10, y: 64, z: 0 },
      velocity: { x: 0, y: 0, z: 0 },
      yaw: 0,
      pitch: 0,
    },
    health: 20,
    food: 20,
    oxygenLevel: 20,
    inventory: { slots: Array.from({ length: 45 }, () => null), items: () => [] },
    chat(command) {
      this.chats.push(command);
    },
    setControlState(control, state) {
      this.controls.push([control, state]);
    },
    async waitForTicks(ticks) {
      this.waits.push(ticks);
      this.time.age += ticks;
    },
  };
}

function bundle() {
  const actionMap = buildActionMap({
    action: [
      { id: 0, kind: 'noop', ticks: 1 },
      { id: 1, kind: 'move', direction: 'forward', ticks: 2 },
    ],
  });
  return buildConfigBundleFromObjects({
    env: {
      ws_url: 'ws://127.0.0.1:8765',
      episode: { max_ticks: 3, action_repeat: 1 },
      observation: {
        include_velocity: false,
        include_orientation: false,
        include_vitals: false,
        include_inventory: false,
      },
    },
    reset: { teleport: { selector: 'ForgeBot' } },
    actionMapData: actionMap,
    rewardData: { reward: [{ kind: 'survival', value: 0.5 }] },
  });
}

describe('index protocol handler', () => {
  it('rejects mismatched expected observation dimensions', () => {
    const configBundle = bundle();
    configBundle.env.observation.expected_dim = 99;
    assert.throws(() => validateObservationConfig(configBundle.env, 3), /expected_dim/);
  });

  it('sends hello, reset observation, and step observation', async () => {
    const bot = stubBot();
    const socket = new FakeSocket();
    const handleConnection = createConnectionHandler({ bot, bundle: bundle(), logger: { warn() {} } });
    handleConnection(socket);

    assert.equal(socket.sent[0].type, 'hello');
    assert.equal(socket.sent[0].action_count, 2);

    socket.emit('message', JSON.stringify({ type: 'reset', seed: 7 }));
    await delay(0);
    assert.equal(socket.sent[1].type, 'observation');
    assert.equal(socket.sent[1].reward, 0);
    assert.ok(bot.chats.some((command) => command.startsWith('/tp')));

    socket.emit('message', JSON.stringify({ type: 'step', action_id: 1 }));
    await delay(0);
    assert.equal(socket.sent[2].type, 'observation');
    assert.equal(socket.sent[2].reward, 0.5);
    assert.deepEqual(bot.controls, [['forward', true], ['forward', false]]);
  });

  it('reports invalid actions and rejects a second active client', async () => {
    const bot = stubBot();
    const first = new FakeSocket();
    const second = new FakeSocket();
    const handleConnection = createConnectionHandler({ bot, bundle: bundle(), logger: { warn() {} } });
    handleConnection(first);
    handleConnection(second);
    assert.equal(second.sent[0].code, 'BUSY');

    first.emit('message', JSON.stringify({ type: 'step', action_id: 99 }));
    await delay(0);
    assert.equal(first.sent.at(-1).code, 'INVALID_ACTION');
  });
});