# FORGE Control Center

The unified web UI for FORGE — live simulation, training metrics, episode
replay, and engine demos in a single React app. This consolidates what were
previously two separate front-ends (the React `dashboard/` and the vanilla-JS
`demo_ui/`) into one professional, dev-tool-styled dashboard.

## Stack

- **React 18 + TypeScript** (strict) with **Vite**
- **Tailwind CSS** with a semantic design-token layer (CSS variables, dark theme)
- **Radix UI** primitives + a small in-house component library (`src/components/ui`)
- **React Router** with route-level code splitting
- **Recharts** for metric/replay charts (lazy-loaded)
- **Biome** (lint) + **Vitest** (unit tests)

## Layout

```
src/
  components/
    ui/          design-system primitives (button, card, badge, tabs, …)
    layout/      app shell — Sidebar, TopBar, AppShell
    *.tsx        domain panels (SimulationCanvas, MetricsDashboard, …)
  pages/         one component per route
  context/       SimulationProvider (shared WebSocket subscription)
  hooks/         useWebSocket / useSimulationState / useMetrics
  lib/           cn() + formatters, trajectory parser
  config/        runtime configuration (VITE_* env vars)
```

## Routes

| Route       | Page          | What it shows                                            |
| ----------- | ------------- | -------------------------------------------------------- |
| `/live`     | Live          | World canvas, agent inspector, decision traces, controls |
| `/training` | Training      | Reward/loss/entropy curves; run comparison (placeholder) |
| `/runs`     | Runs          | Training/eval run history (placeholder)                  |
| `/replay`   | Replay        | Scrub a `forge-replay` v2 trajectory JSON                |
| `/demos`    | Demos         | Run engine demos live via the `demo_ui` backend (SSE)    |
| `/settings` | Settings      | Resolved runtime configuration                           |

## Commands

```bash
npm install          # install dependencies
npm run dev          # start the dev server (proxies /api + /ws to :8080)
npm run build        # type-check + production build
npm run lint         # Biome lint
npm test             # Vitest unit tests
```

## Configuration

All runtime config is resolved once at startup from `VITE_*` environment
variables (see `src/config/environment.ts` and the Settings page):

| Variable                     | Default                  | Purpose                      |
| ---------------------------- | ------------------------ | ---------------------------- |
| `VITE_WS_URL`                | `ws://localhost:8080/ws` | Simulation state stream      |
| `VITE_API_BASE_URL`          | `http://localhost:8080`  | `forge-server` REST API      |
| `VITE_DEMO_API_BASE_URL`     | `http://localhost:8000`  | `demo_ui` FastAPI backend    |
| `VITE_METRICS_INTERVAL`      | `2000`                   | Metrics poll interval (ms)   |
| `VITE_CELL_SIZE`             | `8`                      | Canvas cell size (px)        |

## Data sources

- **Live state** — `forge-server` WebSocket (`/ws`) → `SimulationProvider`.
- **Server metrics** — `forge-server` REST (`/api/metrics`) → `useMetrics`.
- **Replay** — local `forge-replay` v2 trajectory JSON loaded in the browser.
- **Demos** — `demo_ui` FastAPI SSE endpoints (`/api/run/{section}`).

Panels that have no live data source yet (training history, run list) render
explicit empty states rather than fabricated data.
