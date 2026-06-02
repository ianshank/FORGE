// FORGE in-browser WASM demo glue.
//
// Loads the wasm-pack (`--target web`) output and drives the ForgeWasmEnv
// reset/step/render surface entirely client-side. All simulation parameters
// come from DEMO_CONFIG below (passed to the env constructor) — nothing about
// the world is hard-coded in this file beyond the demo's own playback cadence,
// which is itself a named constant.

import init, { ForgeWasmEnv } from "./pkg/forge_wasm.js";

// Simulation config handed to the env. An empty object makes the Rust side use
// ForgeConfig::default(); override fields here to customise the demo world.
const DEMO_CONFIG = {};

// Milliseconds between auto-play steps.
const PLAY_INTERVAL_MS = 120;
// Steps after which a running episode is auto-reset (in addition to the env's
// own termination/truncation signalling).
const MAX_STEPS_PER_EPISODE = 600;

const el = (id) => document.getElementById(id);

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

function resetEpisode(seed) {
  const response = JSON.parse(state.env.reset(seed));
  state.steps = 0;
  state.episode += 1;
  el("tick").textContent = "0";
  el("reward").textContent = "—";
  el("episode").textContent = String(state.episode);
  render();
  return response;
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
    resetEpisode();
    setStatus("ready");
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
