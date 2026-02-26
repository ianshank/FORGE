/**
 * app.js — FORGE Demo UI logic
 *
 * Classes:
 *   ForgeTerminal  – live SSE output with colour syntax
 *   WorldRenderer  – ASCII grid → canvas
 *   StatsPanel     – numeric metric cards
 *   SectionNav     – sidebar state management
 *   Runner         – orchestrates SSE fetch calls & progress
 */

"use strict";

/* =========================================================
   Constants
   ========================================================= */
const SECTIONS_META = [
    { key: "worldgen", name: "World Generation", icon: "🌍" },
    { key: "navigation", name: "Navigation", icon: "🧭" },
    { key: "gathering", name: "Resource Gathering", icon: "🪵" },
    { key: "crafting", name: "Crafting", icon: "⚒️" },
    { key: "multiagent", name: "Multi-Agent", icon: "🤝" },
    { key: "daynight", name: "Day/Night Cycle", icon: "🌙" },
    { key: "determinism", name: "Determinism", icon: "🔁" },
    { key: "performance", name: "Performance", icon: "⚡" },
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
        // A grid line consists mostly of terrain characters
        const terrain = new Set(['.', '~', '#', 'T', 'M', 'S', 'I', 'L', 'A', 'R', 'O', ' ']);
        const ratio = [...t].filter(c => terrain.has(c)).length / t.length;
        return ratio > 0.7;
    }

    _colourize(text) {
        const escaped = this._esc(text);

        // Section headers (=== ... ===)
        if (/={3,}/.test(text)) return `<span class="c-header">${escaped}</span>`;

        // PASS / FAIL markers
        if (/\bPASS\b/.test(text)) return escaped.replace(/PASS/g, '<span class="c-pass">PASS</span>');
        if (/\bFAIL\b/.test(text)) return escaped.replace(/FAIL/g, '<span class="c-fail">FAIL</span>');

        // Step / Move info lines
        if (/^  (Move|Step|\[Step)/.test(text)) return `<span class="c-info">${escaped}</span>`;

        // Crafted / picked up
        if (/Crafted|icked up/.test(text)) return `<span class="c-pass">${escaped}</span>`;

        // Insufficient / insufficient materials
        if (/nsufficient|ERROR/.test(text)) return `<span class="c-fail">${escaped}</span>`;

        // Sub-headers (--- ... ---)
        if (/^---/.test(text)) return `<span class="c-info">${escaped}</span>`;

        // Grid lines: colorize char by char
        if (this._isGridLine(text)) return this._colourizeGrid(text);

        // Dim separators / empty
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
   WorldRenderer
   ========================================================= */
class WorldRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext("2d");
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

        const rows = lines.length;
        const cols = Math.max(...lines.map(l => l.length));
        const { canvas, ctx } = this;

        const cw = canvas.width / cols;
        const ch = canvas.height / rows;

        ctx.fillStyle = "#050c18";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < lines[r].length; c++) {
                const ch_char = lines[r][c];
                ctx.fillStyle = TERRAIN_COLORS[ch_char] || "#1a2540";
                ctx.fillRect(Math.floor(c * cw), Math.floor(r * ch),
                    Math.ceil(cw) + 1, Math.ceil(ch) + 1);
            }
        }
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
        this.states = {}; // key -> "idle"|"running"|"pass"|"fail"
        this._build();
    }

    _build() {
        this.listEl.innerHTML = "";
        this.miniListEl.innerHTML = "";

        for (const s of SECTIONS_META) {
            this.states[s.key] = "idle";

            // Sidebar button
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

            // Mini row for stats panel
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

        // Remove all status classes
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
   Runner — SSE orchestrator
   ========================================================= */
class Runner {
    constructor(terminal, nav) {
        this.terminal = terminal;
        this.nav = nav;
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
        this._setRunning(true);
        this._startTimer();
        this._completedSections = 0;
        this._setProgress(0, this._totalSections);
        document.getElementById("current-section-label").textContent = "— All sections";

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
        this._setRunning(true);
        this._startTimer();
        this.nav.setActive(sectionKey);
        this.nav.setStatus(sectionKey, "running");
        this._setProgress(0, 1);
        document.getElementById("current-section-label").textContent = `— ${meta.icon} ${meta.name}`;
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

                // Sentinel detecting — may be bare or JSON-encoded
                if (rawSlice === "__STREAM_END__" || rawSlice === '"__STREAM_END__"') {
                    streamDone = true;
                    break outer;
                }

                // Safe JSON parse — fall back to raw string; always trim whitespace
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
                            `— ${meta.icon} ${meta.name}`;
                    }
                    continue;
                }

                if (typeof payload === "string" && payload.startsWith("__SECTION_END__ ")) {
                    const sec = payload.replace("__SECTION_END__ ", "").trim();
                    // Mark section done — determine pass/fail from nav state
                    if (sec === currentSection) {
                        this._completedSections++;
                        this._setProgress(this._completedSections, this._totalSections);
                        // If no status set yet, default to pass
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

                // Parse perf stats if present
                this._extractStats(payload);
            }
        }
    }


    /** Pull performance numbers out of demo output. */
    _extractStats(line) {
        const fpsMatch = line.match(/Steps\/second\s*:\s*([\d,]+)/);
        if (fpsMatch) {
            document.getElementById("stat-fps").textContent =
                fpsMatch[1].replace(/,/g, ",");
        }
        const usMatch = line.match(/us\/step\s*:\s*([\d.]+)/);
        if (usMatch) {
            document.getElementById("stat-us").textContent = usMatch[1] + " μs";
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
let terminal, worldRenderer, nav, runner;

function clearTerminal() {
    terminal.clear();
    worldRenderer._drawIdle();
}

function toggleQuick() {
    const el = document.getElementById("quick-toggle");
    el.classList.toggle("on");
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
            (data.platform || "—").substring(0, 30);
        document.getElementById("chip-date").textContent = data.date || "—";
        document.getElementById("chip-result").textContent = data.result || "—";
        document.getElementById("stat-seed").textContent = data.seed ?? 42;
        document.getElementById("seed-input").value = data.seed ?? 42;

        if (data.performance) {
            document.getElementById("stat-fps").textContent =
                data.performance.steps_per_second || "—";
            document.getElementById("stat-us").textContent =
                data.performance.us_per_step
                    ? data.performance.us_per_step + " μs"
                    : "—";
        }
    } catch (_) {
        // Offline / no results yet — that's fine
    }
}

document.addEventListener("DOMContentLoaded", () => {
    terminal = new ForgeTerminal(document.getElementById("terminal-output"));
    worldRenderer = new WorldRenderer(document.getElementById("world-canvas"));
    nav = new SectionNav(
        document.getElementById("section-list"),
        document.getElementById("mini-section-list"),
    );
    runner = new Runner(terminal, nav);

    // Expose globally so HTML onclick handlers work
    window.worldRenderer = worldRenderer;
    window.runner = runner;

    loadResults();
});
