import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright E2E + AQA configuration for the FORGE Control Center.
 *
 * Tests run against the production build served by `vite preview`, with all
 * backends (WebSocket / REST / SSE) mocked in-browser — see `e2e/fixtures`.
 * Chromium only by design (E2E flows + axe-core accessibility).
 */

const PORT = Number(process.env.E2E_PORT ?? 4173);
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
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: {
    command: `npm run build && npm run preview -- --host ${HOST} --port ${PORT} --strictPort`,
    url: `${BASE_URL}/live`,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
