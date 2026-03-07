/**
 * app.js — FORGE Demo UI logic (v4)
 *
 * Classes:
 *   ForgeTerminal  – live SSE output with colour syntax
 *   WorldRenderer  – ASCII grid -> canvas with agent trails
 *   StatsPanel     – numeric metric cards, inventory, day/night
 *   SectionNav     – sidebar state management
 *   Runner         – orchestrates SSE fetch calls & progress
 */

"use strict";

/* =========================================================
   Constants
   ========================================================= */
const SECTIONS_META = [
    { key: "worldgen", name: "World Generation", icon: "\u{1F30D}" },
    { key: "navigation", name: "Navigation", icon: "\u{1F9ED}" },
    { key: "gathering", name: "Resource Gathering", icon: "\u{1FAB5}" },
    { key: "crafting", name: "Crafting", icon: "\u2692\uFE0F" },
    { key: "multiagent", name: "Multi-Agent", icon: "\u{1F91D}" },
    { key: "daynight", name: "Day/Night Cycle", icon: "\u{1F319}" },
    { key: "determinism", name: "Determinism", icon: "\u{1F501}" },
    { key: "performance", name: "Performance", icon: "\u26A1" },
];

const TERRAIN_COLORS = {
    ".": "#4a5568",  // ground
    "~": "#2196F3",  // water
    "#": "#37474f",  // wall
    "T": "#4caf50",  // forest
    "M": "#ff9800",  // mountain
    "S": "#ffd54f",  // sand
    "I": "#80deea",  // ice
    "L": "#f44336",  // lava
    "A": "#ff1744",  // agent
    "R": "#e040fb",  // resource
    "O": "#ffcc02",  // object
};

const ITEM_NAMES = {
    0: "Wood", 1: "Stone", 2: "Ore", 3: "Fish", 4: "Fiber", 5: "Clay",
    10: "Axe", 11: "Pickaxe", 12: "Sword", 13: "Shield", 14: "Plank",
    15: "Bridge", 16: "Rope", 17: "Brick", 18: "Key", 19: "Torch",
    30: "CookedFish", 31: "Bread",
};

/* =========================================================
   ForgeTerminal
   ========================================================= */
class ForgeTerminal {
    constructor(el) {
        this.el = el;
        this._gridBuffer = [];
    }

    clear() {
        this.el.innerHTML = '';
        this._gridBuffer = [];
    }

    /** Append a raw line with syntax highlighting. */
    appendLine(text) {
        if (!text && text !== 0) return;

        // Skip SSE control tokens from display
        if (text === "__DONE__" || text === "__STREAM_END__") return;
        if (text.startsWith("__SECTION_START__") || text.startsWith("__SECTION_END__")) return;

        const span = document.createElement("span");
        span.className = "fade-in";
        span.innerHTML = this._colourize(text) + "\n";
        this.el.appendChild(span);

        // Collect grid lines for the world renderer
        if (this._isGridLine(text)) {
            this._gridBuffer.push(text.trim());
            if (this._gridBuffer.length >= 8) {
                window.worldRenderer.renderGrid(this._gridBuffer.join("\n"));
                this._gridBuffer = [];
            }
        } else {
            if (this._gridBuffer.length > 0) {
                window.worldRenderer.renderGrid(this._gridBuffer.join("\n"));
            }
            this._gridBuffer = [];
        }

        this._scrollBottom();
    }

    _scrollBottom() {
        this.el.scrollTop = this.el.scrollHeight;
    }

    _isGridLine(line) {
        const t = line.trim();
        if (t.length < 4) return false;
        const terrain = new Set(['.', '~', '#', 'T', 'M', 'S', 'I', 'L', 'A', 'R', 'O', ' ']);
        const ratio = [...t].filter(c => terrain.has(c)).length / t.length;
        return ratio > 0.7;
    }

    _colourize(text) {
        const escaped = this._esc(text);

        if (/={3,}/.test(text)) return `<span class="c-header">${escaped}</span>`;
        if (/\bPASS\b/.test(text)) return escaped.replace(/PASS/g, '<span class="c-pass">PASS</span>');
        if (/\bFAIL\b/.test(text)) return escaped.replace(/FAIL/g, '<span class="c-fail">FAIL</span>');
        if (/^  (Move|Step|\[Step)/.test(text)) return `<span class="c-info">${escaped}</span>`;
        if (/Crafted|icked up/.test(text)) return `<span class="c-pass">${escaped}</span>`;
        if (/nsufficient|ERROR/.test(text)) return `<span class="c-fail">${escaped}</span>`;
        if (/^---/.test(text)) return `<span class="c-info">${escaped}</span>`;
        if (this._isGridLine(text)) return this._colourizeGrid(text);
        if (text.trim() === "" || /^={2,}$/.test(text.trim())) {
            return `<span class="c-muted">${escaped}</span>`;
        }

        return escaped;
    }

    _colourizeGrid(line) {
        return [...line].map(ch => {
            const col = TERRAIN_COLORS[ch];
            if (col) return `<span style="color:${col}">${this._esc(ch)}</span>`;
            return this._esc(ch);
        }).join("");
    }

    _esc(s) {
        return s
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");
    }
}

/* =========================================================
   WorldRenderer — with agent trail support
   ========================================================= */
class WorldRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext("2d");
        this._agentTrail = []; // array of {x, y} positions
        this._maxTrailLen = 50;
        this._lastGrid = null;
        this._drawIdle();
    }

    _drawIdle() {
        const { ctx, canvas } = this;
        ctx.fillStyle = "#050c18";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#1a2540";
        ctx.font = "11px 'JetBrains Mono', monospace";
        ctx.textAlign = "center";
        ctx.fillText("World view appears here", canvas.width / 2, canvas.height / 2);
    }

    renderGrid(gridStr) {
        const lines = gridStr.split("\n").filter(l => l.trim().length > 0);
        if (!lines.length) return;

        this._lastGrid = lines;
        const rows = lines.length;
        const cols = Math.max(...lines.map(l => l.length));
        const { canvas, ctx } = this;

        const cw = canvas.width / cols;
        const ch = canvas.height / rows;

        ctx.fillStyle = "#050c18";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Find agent positions and track trail
        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < lines[r].length; c++) {
                const ch_char = lines[r][c];

                // Draw base terrain
                ctx.fillStyle = TERRAIN_COLORS[ch_char] || "#1a2540";
                ctx.fillRect(Math.floor(c * cw), Math.floor(r * ch),
                    Math.ceil(cw) + 1, Math.ceil(ch) + 1);

                if (ch_char === 'A') {
                    this._agentTrail.push({ x: c, y: r });
                    if (this._agentTrail.length > this._maxTrailLen) {
                        this._agentTrail.shift();
                    }
                }
            }
        }

        // Draw agent trail (fading from old to new)
        if (this._agentTrail.length > 1) {
            for (let i = 0; i < this._agentTrail.length - 1; i++) {
                const alpha = (i / this._agentTrail.length) * 0.6;
                const p = this._agentTrail[i];
                ctx.fillStyle = `rgba(255, 23, 68, ${alpha})`;
                const dotSize = Math.max(2, Math.min(cw, ch) * 0.4);
                ctx.beginPath();
                ctx.arc(
                    p.x * cw + cw / 2,
                    p.y * ch + ch / 2,
                    dotSize,
                    0, Math.PI * 2
                );
                ctx.fill();
            }
        }

        // Redraw agent on top with glow
        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < lines[r].length; c++) {
                if (lines[r][c] === 'A') {
                    const cx_pos = c * cw + cw / 2;
                    const cy_pos = r * ch + ch / 2;
                    const agentSize = Math.max(3, Math.min(cw, ch) * 0.5);

                    // Glow
                    ctx.shadowColor = "#ff1744";
                    ctx.shadowBlur = 8;
                    ctx.fillStyle = "#ff1744";
                    ctx.beginPath();
                    ctx.arc(cx_pos, cy_pos, agentSize, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.shadowBlur = 0;
                }
            }
        }
    }

    clearTrail() {
        this._agentTrail = [];
    }
}

/* =========================================================
   SectionNav
   ========================================================= */
const STATUSES = ["idle", "running", "pass", "fail"];

class SectionNav {
    constructor(listEl, miniListEl) {
        this.listEl = listEl;
        this.miniListEl = miniListEl;
        this.states = {};
        this._build();
    }

    _build() {
        this.listEl.innerHTML = "";
        this.miniListEl.innerHTML = "";

        for (const s of SECTIONS_META) {
            this.states[s.key] = "idle";

            const btn = document.createElement("button");
            btn.className = "section-btn";
            btn.id = `sbtn-${s.key}`;
            btn.innerHTML = `
        <span class="section-icon">${s.icon}</span>
        <span class="section-name">${s.name}</span>
        <span class="section-badge" id="badge-${s.key}">IDLE</span>
      `;
            btn.onclick = () => window.runner.runSection(s.key);
            this.listEl.appendChild(btn);

            const row = document.createElement("div");
            row.className = "mini-section";
            row.innerHTML = `
        <span class="mini-section-name">${s.name}</span>
        <span class="mini-status idle" id="mini-${s.key}">IDLE</span>
      `;
            this.miniListEl.appendChild(row);
        }
    }

    setStatus(key, status) {
        this.states[key] = status;
        const btn = document.getElementById(`sbtn-${key}`);
        const badge = document.getElementById(`badge-${key}`);
        const mini = document.getElementById(`mini-${key}`);
        if (!btn || !badge || !mini) return;

        btn.classList.remove(...STATUSES);
        mini.classList.remove(...STATUSES);

        btn.classList.add(status);
        mini.classList.add(status);
        badge.textContent = status.toUpperCase();
        mini.textContent = status.toUpperCase();
    }

    setActive(key) {
        document.querySelectorAll(".section-btn").forEach(b => b.classList.remove("active"));
        const btn = document.getElementById(`sbtn-${key}`);
        if (btn) btn.classList.add("active");
    }

    resetAll() {
        for (const s of SECTIONS_META) this.setStatus(s.key, "idle");
    }

    passCount() {
        return Object.values(this.states).filter(s => s === "pass").length;
    }
}

/* =========================================================
   StatsTracker — parses output for live stats
   ========================================================= */
class StatsTracker {
    constructor() {
        this._inventory = {};
        this._dayPhase = null;
        this._tick = 0;
        this._agentCount = 0;
    }

    /** Parse a line for stats data. */
    parseLine(line) {
        // Steps/second
        const fpsMatch = line.match(/Steps\/second\s*:\s*([\d,]+)/);
        if (fpsMatch) {
            document.getElementById("stat-fps").textContent = fpsMatch[1];
        }
        const usMatch = line.match(/us\/step\s*:\s*([\d.]+)/);
        if (usMatch) {
            document.getElementById("stat-us").textContent = usMatch[1] + " \u03BCs";
        }

        // Inventory tracking: "Picked up Wood" or "Inventory: {Wood: 2, Stone: 1}"
        const pickupMatch = line.match(/Picked up (\w+)/i);
        if (pickupMatch) {
            const item = pickupMatch[1];
            this._inventory[item] = (this._inventory[item] || 0) + 1;
            this._updateInventoryDisplay();
        }

        const invMatch = line.match(/Inventory:\s*\{([^}]+)\}/);
        if (invMatch) {
            this._inventory = {};
            const pairs = invMatch[1].split(",");
            for (const pair of pairs) {
                const [name, count] = pair.split(":").map(s => s.trim());
                if (name && count) {
                    this._inventory[name] = parseInt(count, 10);
                }
            }
            this._updateInventoryDisplay();
        }

        // Crafting: "Crafted Axe"
        const craftMatch = line.match(/Crafted\s+(\w+)/i);
        if (craftMatch) {
            const item = craftMatch[1];
            this._inventory[item] = (this._inventory[item] || 0) + 1;
            this._updateInventoryDisplay();
        }

        // Day/night phase detection
        const phaseMatch = line.match(/Phase:\s*(Dawn|Day|Dusk|Night)/i);
        if (phaseMatch) {
            this._setDayPhase(phaseMatch[1].toLowerCase());
        }
        if (/\bdawn\b/i.test(line) && /phase|cycle/i.test(line)) {
            this._setDayPhase("dawn");
        } else if (/\bday\b/i.test(line) && /phase|cycle/i.test(line)) {
            this._setDayPhase("day");
        } else if (/\bdusk\b/i.test(line) && /phase|cycle/i.test(line)) {
            this._setDayPhase("dusk");
        } else if (/\bnight\b/i.test(line) && /phase|cycle/i.test(line)) {
            this._setDayPhase("night");
        }

        // Tick tracking
        const tickMatch = line.match(/\[?(?:Step|Tick)\s*(\d+)/i);
        if (tickMatch) {
            this._tick = parseInt(tickMatch[1], 10);
            document.getElementById("footer-tick").textContent = `Tick: ${this._tick}`;
        }

        // Agent count
        const agentMatch = line.match(/(\d+)\s*agents?/i);
        if (agentMatch) {
            this._agentCount = parseInt(agentMatch[1], 10);
            document.getElementById("footer-agents").textContent = `Agents: ${this._agentCount}`;
        }
    }

    _setDayPhase(phase) {
        this._dayPhase = phase;
        const phases = ["dawn", "day", "dusk", "night"];
        for (const p of phases) {
            const el = document.getElementById(`phase-${p}`);
            if (el) {
                el.classList.toggle("active", p === phase);
            }
        }
    }

    _updateInventoryDisplay() {
        const el = document.getElementById("inventory-display");
        if (!el) return;

        const entries = Object.entries(this._inventory).filter(([, v]) => v > 0);
        if (entries.length === 0) {
            el.innerHTML = '<span class="inventory-empty">No items yet</span>';
            return;
        }

        el.innerHTML = entries.map(([name, count]) =>
            `<span class="inventory-item">${name} <span class="item-count">x${count}</span></span>`
        ).join("");
    }

    reset() {
        this._inventory = {};
        this._dayPhase = null;
        this._tick = 0;
        this._agentCount = 0;
        this._updateInventoryDisplay();
        const phases = ["dawn", "day", "dusk", "night"];
        for (const p of phases) {
            const el = document.getElementById(`phase-${p}`);
            if (el) el.classList.remove("active");
        }
        document.getElementById("footer-tick").textContent = "Tick: \u2014";
        document.getElementById("footer-agents").textContent = "Agents: \u2014";
    }
}

/* =========================================================
   Runner — SSE orchestrator
   ========================================================= */
class Runner {
    constructor(terminal, nav, stats) {
        this.terminal = terminal;
        this.nav = nav;
        this.stats = stats;
        this._abortCtrl = null;
        this._running = false;
        this._startTime = 0;
        this._timerInterval = null;
        this._completedSections = 0;
        this._totalSections = SECTIONS_META.length;
    }

    get seed() {
        return parseInt(document.getElementById("seed-input").value, 10) || 42;
    }

    get quick() {
        return document.getElementById("quick-toggle").classList.contains("on");
    }

    _setRunning(running) {
        this._running = running;
        document.getElementById("btn-run-all").disabled = running;
        document.getElementById("btn-stop").disabled = !running;
        document.getElementById("seed-input").disabled = running;
        document.querySelectorAll(".section-btn").forEach(b => {
            b.style.pointerEvents = running ? "none" : "";
        });
    }

    _startTimer() {
        this._startTime = Date.now();
        clearInterval(this._timerInterval);
        this._timerInterval = setInterval(() => {
            const sec = ((Date.now() - this._startTime) / 1000).toFixed(1);
            document.getElementById("footer-elapsed").textContent = `${sec}s`;
        }, 100);
    }

    _stopTimer() {
        clearInterval(this._timerInterval);
    }

    _setProgress(done, total) {
        const pct = total > 0 ? (done / total) * 100 : 0;
        document.getElementById("progress-bar").style.width = `${pct}%`;
        document.getElementById("progress-label").textContent =
            total > 0 ? `${done} / ${total} sections` : "Idle";
    }

    _setPrompt(txt) {
        document.getElementById("prompt-text").textContent = txt;
    }

    async runAll() {
        if (this._running) return;
        this.terminal.clear();
        this.nav.resetAll();
        this.stats.reset();
        window.worldRenderer.clearTrail();
        this._setRunning(true);
        this._startTimer();
        this._completedSections = 0;
        this._setProgress(0, this._totalSections);
        document.getElementById("current-section-label").textContent = "\u2014 All sections";

        this._abortCtrl = new AbortController();

        try {
            const resp = await fetch("/api/run-all", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ seed: this.seed, quick: this.quick }),
                signal: this._abortCtrl.signal,
            });

            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

            await this._consumeStream(resp, "all");
        } catch (err) {
            if (err.name !== "AbortError") {
                this.terminal.appendLine(`\nERROR: ${err.message}`);
            }
        } finally {
            this._finalize();
        }
    }

    async runSection(sectionKey) {
        if (this._running) return;
        const meta = SECTIONS_META.find(s => s.key === sectionKey);
        if (!meta) return;

        this.terminal.clear();
        this.stats.reset();
        window.worldRenderer.clearTrail();
        this._setRunning(true);
        this._startTimer();
        this.nav.setActive(sectionKey);
        this.nav.setStatus(sectionKey, "running");
        this._setProgress(0, 1);
        document.getElementById("current-section-label").textContent = `\u2014 ${meta.icon} ${meta.name}`;
        this._setPrompt(`running ${sectionKey}`);

        this._abortCtrl = new AbortController();

        try {
            const resp = await fetch(`/api/run/${sectionKey}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ seed: this.seed, quick: this.quick }),
                signal: this._abortCtrl.signal,
            });

            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

            await this._consumeStream(resp, sectionKey);
        } catch (err) {
            if (err.name !== "AbortError") {
                this.terminal.appendLine(`\nERROR: ${err.message}`);
                this.nav.setStatus(sectionKey, "fail");
            }
        } finally {
            this._finalize();
        }
    }

    /** Consume a text/event-stream SSE response and route lines. */
    async _consumeStream(resp, context) {
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let currentSection = context === "all" ? null : context;
        let streamDone = false;

        outer: while (!streamDone) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() ?? "";

            for (const raw of lines) {
                if (!raw.startsWith("data: ")) continue;

                const rawSlice = raw.slice(6).trim();

                if (rawSlice === "__STREAM_END__" || rawSlice === '"__STREAM_END__"') {
                    streamDone = true;
                    break outer;
                }

                let payload;
                try {
                    payload = JSON.parse(rawSlice);
                } catch (_) {
                    payload = rawSlice;
                }
                if (typeof payload === "string") payload = payload.trim();

                // Control tokens for "run-all" mode
                if (typeof payload === "string" && payload.startsWith("__SECTION_START__ ")) {
                    currentSection = payload.replace("__SECTION_START__ ", "").trim();
                    const meta = SECTIONS_META.find(s => s.key === currentSection);
                    if (meta) {
                        this.nav.setActive(currentSection);
                        this.nav.setStatus(currentSection, "running");
                        this._setPrompt(`running ${currentSection}`);
                        document.getElementById("current-section-label").textContent =
                            `\u2014 ${meta.icon} ${meta.name}`;
                    }
                    window.worldRenderer.clearTrail();
                    continue;
                }

                if (typeof payload === "string" && payload.startsWith("__SECTION_END__ ")) {
                    const sec = payload.replace("__SECTION_END__ ", "").trim();
                    if (sec === currentSection) {
                        this._completedSections++;
                        this._setProgress(this._completedSections, this._totalSections);
                        const st = this.nav.states[sec];
                        if (st === "running") this.nav.setStatus(sec, "pass");
                    }
                    continue;
                }

                if (typeof payload === "string" && payload === "__DONE__") continue;

                // Detect PASS/FAIL in output
                if (currentSection && typeof payload === "string") {
                    if (/\bPASS\b/.test(payload)) {
                        this.nav.setStatus(currentSection, "pass");
                    } else if (/\bFAIL\b/.test(payload)) {
                        this.nav.setStatus(currentSection, "fail");
                    }
                }

                this.terminal.appendLine(payload);

                // Parse live stats
                if (typeof payload === "string") {
                    this.stats.parseLine(payload);
                }
            }
        }
    }

    stop() {
        if (this._abortCtrl) this._abortCtrl.abort();
        this._running = false;
        this._finalize();
    }

    _finalize() {
        this._setRunning(false);
        this._stopTimer();
        this._setPrompt("ready");
        document.getElementById("current-section-label").textContent = "";
        const pct = this._completedSections / this._totalSections * 100;
        if (pct >= 100 || !this._running) {
            this._setProgress(this._completedSections, this._totalSections);
        }
    }
}

/* =========================================================
   Init
   ========================================================= */
let terminal, worldRenderer, nav, stats, runner;

function clearTerminal() {
    terminal.clear();
    worldRenderer._drawIdle();
    worldRenderer.clearTrail();
    stats.reset();
}

function toggleQuick() {
    const el = document.getElementById("quick-toggle");
    el.classList.toggle("on");
}

function updateSpeedLabel() {
    const slider = document.getElementById("speed-slider");
    document.getElementById("speed-label").textContent = slider.value + "x";
}

function runAll() {
    runner.runAll();
}

function stopRun() {
    runner.stop();
}

async function loadResults() {
    try {
        const r = await fetch("/api/results");
        const data = await r.json();

        document.getElementById("chip-platform").textContent =
            (data.platform || "\u2014").substring(0, 30);
        document.getElementById("chip-date").textContent = data.date || "\u2014";
        document.getElementById("chip-result").textContent = data.result || "\u2014";
        document.getElementById("stat-seed").textContent = data.seed ?? 42;
        document.getElementById("seed-input").value = data.seed ?? 42;

        if (data.performance) {
            document.getElementById("stat-fps").textContent =
                data.performance.steps_per_second || "\u2014";
            document.getElementById("stat-us").textContent =
                data.performance.us_per_step
                    ? data.performance.us_per_step + " \u03BCs"
                    : "\u2014";
        }
    } catch (_) {
        // Offline / no results yet
    }
}

document.addEventListener("DOMContentLoaded", () => {
    terminal = new ForgeTerminal(document.getElementById("terminal-output"));
    worldRenderer = new WorldRenderer(document.getElementById("world-canvas"));
    stats = new StatsTracker();
    nav = new SectionNav(
        document.getElementById("section-list"),
        document.getElementById("mini-section-list"),
    );
    runner = new Runner(terminal, nav, stats);

    // Expose globally so HTML onclick handlers work
    window.worldRenderer = worldRenderer;
    window.runner = runner;

    loadResults();
});
