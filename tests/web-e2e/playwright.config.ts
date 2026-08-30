import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright E2E for the static in-browser WASM demo (`web/`).
 *
 * Unlike `dashboard/e2e`, which mocks WebSocket/REST/SSE in-browser because an
 * unmocked fetch would escape to a server nothing runs, this suite mocks
 * nothing: the demo is fully client-side. It drives the real deploy artifact --
 * `web/index.html` + `web/app.js` + the wasm-pack `--target web` output --
 * served byte-for-byte by `serve.mjs`, which is the same bytes `gh-pages.yml`
 * uploads. Chromium only, matching `dashboard/playwright.config.ts`.
 */
const PORT = Number(process.env.WEB_E2E_PORT ?? 4174);
const HOST = "127.0.0.1";
const BASE_URL = `http://${HOST}:${PORT}`;

export default defineConfig({
  testDir: "./e2e",
  testMatch: "**/*.spec.ts",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: process.env.CI ? 2 : undefined,
  reporter: process.env.CI
    ? [["html", { open: "never" }], ["github"], ["list"]]
    : [["html", { open: "never" }], ["list"]],
  use: {
    baseURL: BASE_URL,
    trace: "on-first-retry",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        // Normally undefined, so Playwright resolves the browser it downloaded
        // (`npm run test:e2e:install`, and what CI does). The override exists
        // for hosts that already have a compatible Chromium and cannot or do
        // not want to fetch another -- a container with a preinstalled browser,
        // or an air-gapped machine.
        ...(process.env.WEB_E2E_CHROMIUM_PATH
          ? { launchOptions: { executablePath: process.env.WEB_E2E_CHROMIUM_PATH } }
          : {}),
      },
    },
  ],
  webServer: {
    // preflight.mjs builds (or verifies) web/pkg/ before the server starts.
    // It has to run here rather than in globalSetup -- Playwright starts
    // webServer as a plugin, and plugin setup is ordered ahead of global
    // setups, so a globalSetup check would fire only after the server had
    // already come up on an empty web/pkg.
    command: "node preflight.mjs && node serve.mjs",
    url: BASE_URL,
    reuseExistingServer: !process.env.CI,
    // A cold wasm-pack build (including its wasm-opt pass) comfortably exceeds
    // dashboard's 120s.
    timeout: 300_000,
    stdout: "pipe",
    stderr: "pipe",
  },
});
