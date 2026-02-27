/**
 * demo.js — FORGE WASM Demo UI
 *
 * Loads the forge_wasm WebAssembly module, wires up all controls, and drives
 * the simulation canvas. All configuration comes from the HTML controls;
 * no values are hardcoded in this script.
 *
 * URL hash format: #seed=<N>[&size=<N>][&agents=<N>]
 */

/* ─── Module imports ──────────────────────────────────────────────────────── */
import init, { ForgeWasmEnv } from "./pkg/forge_wasm.js";

/* ─── Constants ──────────────────────────────────────────────────────────── */
const DEFAULT_SEED = 42;
const DEFAULT_WORLD_SIZE = 32;
const DEFAULT_NUM_AGENTS = 1;
const DAY_PHASE_NAMES = ["Dawn", "Day", "Dusk", "Night"];

/* ─── State ──────────────────────────────────────────────────────────────── */
let env = null;
let wasmReady = false;
let playIntervalId = null;
let cumulativeReward = 0;
let stepCount = 0;
let actionNames = [];

/* ─── DOM references ─────────────────────────────────────────────────────── */
const $ = (id) => document.getElementById(id);

const elCanvas = $("ascii-canvas");
const elTickCounter = $("tick-counter");
const elStatusBadge = $("status-badge");
const elAgentRow = $("agent-row");
const elErrorBanner = $("error-banner");
const elErrorMsg = $("error-message");
const elBtnReset = $("btn-reset");
const elBtnStep = $("btn-step");
const elBtnPlay = $("btn-play");
const elBtnPause = $("btn-pause");
const elFpsInput = $("fps-input");
const elFpsDisplay = $("fps-display");
const elActionSelect = $("action-select");
const elSeedInput = $("seed-input");
const elWorldSize = $("world-size-input");
const elNumAgents = $("num-agents-input");
const elStatReward = $("stat-reward");
const elStatEpLen = $("stat-ep-len");
const elStatTerminated = $("stat-terminated");
const elStatTruncated = $("stat-truncated");
const elStatAgents = $("stat-agents-alive");
const elStatDayPhase = $("stat-day-phase");
const elCopyLink = $("btn-copy-link");
const elCopyConfirm = $("copy-confirmation");
const elObsSpaceJson = $("obs-space-json");
const elActSpaceJson = $("act-space-json");

/* ─── Error handling ─────────────────────────────────────────────────────── */
function showError(msg) {
    elErrorMsg.textContent = msg;
    elErrorBanner.classList.remove("hidden");
}
function clearError() {
    elErrorBanner.classList.add("hidden");
}
$("error-dismiss").addEventListener("click", clearError);

function safeCall(fn, label = "WASM call") {
    try {
        return fn();
    } catch (err) {
        showError(`${label} failed: ${err.message ?? err}`);
        console.error(`[FORGE] ${label} error:`, err);
        return null;
    }
}

/* ─── URL hash routing ───────────────────────────────────────────────────── */
function parseHash() {
    const params = new URLSearchParams(window.location.hash.replace(/^#/, ""));
    return {
        seed: parseInt(params.get("seed") ?? DEFAULT_SEED, 10) || DEFAULT_SEED,
        size: parseInt(params.get("size") ?? DEFAULT_WORLD_SIZE, 10) || DEFAULT_WORLD_SIZE,
        agents: parseInt(params.get("agents") ?? DEFAULT_NUM_AGENTS, 10) || DEFAULT_NUM_AGENTS,
    };
}

function applyHashToControls() {
    const { seed, size, agents } = parseHash();
    elSeedInput.value = seed;
    elNumAgents.value = agents;
    // Find or approximate the closest world size option
    const opt = [...elWorldSize.options].find((o) => parseInt(o.value) === size);
    if (opt) elWorldSize.value = String(size);
}

function buildHashFromControls() {
    const params = new URLSearchParams({
        seed: elSeedInput.value,
        size: elWorldSize.value,
        agents: elNumAgents.value,
    });
    return `#${params.toString()}`;
}

window.addEventListener("hashchange", () => {
    applyHashToControls();
    if (wasmReady) resetEpisode();
});

/* ─── Config builder ─────────────────────────────────────────────────────── */
function buildConfigJson() {
    const size = parseInt(elWorldSize.value, 10);
    const agents = parseInt(elNumAgents.value, 10);
    return JSON.stringify({
        world: { width: size, height: size },
        agents: { num_agents: agents },
    });
}

/* ─── Env lifecycle ─────────────────────────────────────────────────────── */
function createEnv() {
    const configJson = buildConfigJson();
    env = safeCall(() => new ForgeWasmEnv(configJson), "ForgeWasmEnv constructor");
    if (!env) return false;

    // Populate action select
    const actionSpaceJson = safeCall(() => env.action_space_json(), "action_space_json");
    if (actionSpaceJson) {
        const actionSpace = JSON.parse(actionSpaceJson);
        actionNames = actionSpace.action_names ?? [];
        elActionSelect.innerHTML = actionNames
            .map((name, i) => `<option value="${i}">${i}: ${name}</option>`)
            .join("");
        elActSpaceJson.textContent = JSON.stringify(actionSpace, null, 2);
    }

    const obsSpaceJson = safeCall(() => env.observation_space_json(), "observation_space_json");
    if (obsSpaceJson) {
        elObsSpaceJson.textContent = JSON.stringify(JSON.parse(obsSpaceJson), null, 2);
    }

    return true;
}

function resetEpisode() {
    stopPlayback();
    clearError();

    if (!createEnv()) return;

    const seed = parseInt(elSeedInput.value, 10) || DEFAULT_SEED;
    const result = safeCall(() => JSON.parse(env.reset(BigInt(seed))), "env.reset");
    if (!result) return;

    cumulativeReward = 0;
    stepCount = 0;

    updateCanvas();
    updateStats({ rewards: result.rewards, terminated: result.terminated, truncated: result.truncated });

    setStatus("Ready", "ready");
    setControlsEnabled(true);

    // Update URL hash without triggering hashchange
    history.replaceState(null, "", buildHashFromControls());
}

/* ─── Stepping ───────────────────────────────────────────────────────────── */
function stepOnce() {
    if (!env) return;

    const action = parseInt(elActionSelect.value, 10) || 0;
    const result = safeCall(() => JSON.parse(env.step(action)), "env.step");
    if (!result) return;

    stepCount += 1;
    cumulativeReward += (result.rewards?.[0] ?? 0);

    updateCanvas();
    updateStats(result);

    if (result.terminated || result.truncated) {
        stopPlayback();
        setStatus(result.terminated ? "Terminated" : "Truncated", "done");
    }
}

/* ─── Playback ───────────────────────────────────────────────────────────── */
function startPlayback() {
    const fps = parseInt(elFpsInput.value, 10) || 10;
    stopPlayback();
    playIntervalId = setInterval(stepOnce, 1000 / fps);
    elBtnPlay.classList.add("hidden");
    elBtnPause.classList.remove("hidden");
}

function stopPlayback() {
    if (playIntervalId !== null) {
        clearInterval(playIntervalId);
        playIntervalId = null;
    }
    elBtnPlay.classList.remove("hidden");
    elBtnPause.classList.add("hidden");
}

/* ─── Canvas rendering ───────────────────────────────────────────────────── */
function updateCanvas() {
    if (!env) return;

    const ascii = safeCall(() => env.render_ascii(), "render_ascii") ?? "";
    elCanvas.textContent = ascii;

    const stateJson = safeCall(() => JSON.parse(env.get_state_json()), "get_state_json");
    if (!stateJson) return;

    elTickCounter.textContent = stateJson.tick;

    // Agent position badges
    const positions = stateJson.agent_positions ?? [];
    const alive = stateJson.agents_alive ?? [];
    elAgentRow.innerHTML = positions
        .map(([x, y], i) => {
            const icon = alive[i] ? "🤖" : "💀";
            return `<span class="agent-badge" title="Agent ${i}">${icon} A${i} (${x},${y})</span>`;
        })
        .join("");

    // Day phase stat
    elStatDayPhase.textContent = DAY_PHASE_NAMES[stateJson.day_phase] ?? stateJson.day_phase;
    elStatAgents.textContent = `${alive.filter(Boolean).length}/${alive.length}`;
}

function updateStats({ rewards, terminated, truncated }) {
    const r0 = rewards?.[0] ?? 0;
    elStatReward.textContent = cumulativeReward.toFixed(3);
    elStatEpLen.textContent = stepCount;
    elStatTerminated.textContent = terminated ? "✅ Yes" : "No";
    elStatTruncated.textContent = truncated ? "✅ Yes" : "No";
}

/* ─── UI helpers ─────────────────────────────────────────────────────────── */
function setStatus(text, state = "loading") {
    elStatusBadge.textContent = text;
    elStatusBadge.dataset.state = state;
}

function setControlsEnabled(enabled) {
    [elBtnReset, elBtnStep, elBtnPlay].forEach((btn) => (btn.disabled = !enabled));
}

/* ─── Event handlers ─────────────────────────────────────────────────────── */
elBtnReset.addEventListener("click", resetEpisode);
elBtnStep.addEventListener("click", stepOnce);
elBtnPlay.addEventListener("click", startPlayback);
elBtnPause.addEventListener("click", stopPlayback);

elFpsInput.addEventListener("input", () => {
    elFpsDisplay.textContent = elFpsInput.value;
    elFpsInput.setAttribute("aria-valuenow", elFpsInput.value);
    if (playIntervalId !== null) startPlayback(); // restart with new FPS
});

elCopyLink.addEventListener("click", async () => {
    const url = window.location.origin + window.location.pathname + buildHashFromControls();
    try {
        await navigator.clipboard.writeText(url);
        elCopyConfirm.textContent = "✅ Link copied!";
    } catch {
        elCopyConfirm.textContent = `Copy: ${url}`;
    }
    setTimeout(() => (elCopyConfirm.textContent = ""), 3000);
});

/* ─── Bootstrap ─────────────────────────────────────────────────────────── */
async function bootstrap() {
    setStatus("Loading WASM…", "loading");
    setControlsEnabled(false);

    try {
        await init();
        wasmReady = true;
        applyHashToControls();
        setStatus("Ready", "ready");
        resetEpisode();
    } catch (err) {
        setStatus("Failed to load WASM", "error");
        showError(`Failed to initialise WASM module: ${err.message ?? err}`);
        console.error("[FORGE] WASM init error:", err);
    }
}

bootstrap();
