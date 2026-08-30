// FORGE in-browser WASM demo glue.
//
// Loads the wasm-pack (`--target web`) output and drives the ForgeWasmEnv
// reset/step/render surface entirely client-side. All simulation parameters
// come from DEMO_CONFIG below (passed to the env constructor) — nothing about
// the world is hard-coded in this file beyond the demo's own playback cadence,
// which is itself a named constant, and the u64 seed ceiling.

import init, { ForgeWasmEnv } from "./pkg/forge_wasm.js";

// Simulation config handed to the env. An empty object makes the Rust side use
// ForgeConfig::default(); override fields here to customise the demo world.
const DEMO_CONFIG = {};

// Largest value a Rust u64 seed can hold. Seeds outside this range are
// rejected rather than silently truncated.
const MAX_SEED = (1n << 64n) - 1n;

// Milliseconds between auto-play steps.
const PLAY_INTERVAL_MS = 120;
// Steps after which a running episode is auto-reset (in addition to the env's
// own termination/truncation signalling).
const MAX_STEPS_PER_EPISODE = 600;

const el = (id) => document.getElementById(id);

// `reset` takes a Rust Option<u64>, which wasm-bindgen lowers to an i64 wasm
// parameter — so seeds cross this boundary as BigInt. Passing a JS Number
// throws a TypeError; passing undefined means "no seed".
function parseSeed(text) {
  const trimmed = text.trim();
  if (!/^\d+$/.test(trimmed)) return null;
  const value = BigInt(trimmed);
  return value <= MAX_SEED ? value : null;
}

// Generated here rather than letting Rust derive one internally, so the demo
// can always display the seed it used. That is what makes the reproducibility
// claim checkable by a visitor instead of merely asserted.
function randomSeed() {
  const buf = new BigUint64Array(1);
  crypto.getRandomValues(buf);
  return buf[0];
}

const state = {
  env: null,
  actionCount: 1,
  episode: 0,
  steps: 0,
  timer: null,
};

function setStatus(text) {
  el("status").textContent = text;
}

function render() {
  el("grid").textContent = state.env.render_ascii();
}

function readActionCount(env) {
  try {
    const space = JSON.parse(env.action_space_json());
    if (typeof space.n === "number" && space.n > 0) {
      return space.n;
    }
  } catch (err) {
    console.warn("could not parse action space", err);
  }
  return 1;
}

// `seed` is a BigInt, or undefined to draw a fresh random one. The seed that
// actually ran is always written to #seed-used, whichever branch supplied it.
function resetEpisode(seed) {
  const effective = seed === undefined ? randomSeed() : seed;
  const response = JSON.parse(state.env.reset(effective));
  state.steps = 0;
  state.episode += 1;
  el("tick").textContent = "0";
  el("reward").textContent = "—";
  el("episode").textContent = String(state.episode);
  el("seed-used").textContent = String(effective);
  render();
  return response;
}

// Read the seed box for an explicit Reset. An empty box means "surprise me";
// anything unparseable falls back to random and says so, rather than silently
// ignoring what the user typed.
function resetFromInput() {
  const raw = el("seed").value;
  if (raw.trim() === "") {
    resetEpisode();
    setStatus("ready");
    return;
  }
  const seed = parseSeed(raw);
  if (seed === null) {
    resetEpisode();
    setStatus(`invalid seed (0..2^64-1) — used a random one`);
    return;
  }
  resetEpisode(seed);
  setStatus("ready");
}

function stepOnce() {
  const action = Math.floor(Math.random() * state.actionCount);
  const result = JSON.parse(state.env.step(action));
  state.steps += 1;

  el("tick").textContent = String(state.steps);
  const reward = Array.isArray(result.rewards) ? result.rewards[0] : 0;
  el("reward").textContent = (reward ?? 0).toFixed(2);
  render();

  if (result.terminated || result.truncated || state.steps >= MAX_STEPS_PER_EPISODE) {
    // Deliberately seedless: honouring the seed box here would make Play loop
    // the same episode forever. An explicit Reset is what applies a seed.
    resetEpisode();
  }
}

function play() {
  if (state.timer !== null) return;
  setStatus("playing");
  el("play").disabled = true;
  el("pause").disabled = false;
  state.timer = setInterval(stepOnce, PLAY_INTERVAL_MS);
}

function pause() {
  if (state.timer === null) return;
  clearInterval(state.timer);
  state.timer = null;
  setStatus("paused");
  el("play").disabled = false;
  el("pause").disabled = true;
}

async function main() {
  await init();
  state.env = new ForgeWasmEnv(JSON.stringify(DEMO_CONFIG));
  state.actionCount = readActionCount(state.env);
  el("actions").textContent = String(state.actionCount);

  resetEpisode();
  setStatus("ready");

  el("reset").addEventListener("click", () => {
    pause();
    resetFromInput();
  });
  el("step").addEventListener("click", () => {
    pause();
    stepOnce();
  });
  el("play").addEventListener("click", play);
  el("pause").addEventListener("click", pause);
}

main().catch((err) => {
  console.error(err);
  setStatus(`error: ${err}`);
  el("grid").textContent = String(err);
});
