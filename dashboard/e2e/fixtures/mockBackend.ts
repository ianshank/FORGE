import type { Page } from "@playwright/test";
import { DEMO_LINES, REMIX_OK, SERVER_METRICS, sseBody } from "./mockData";

/**
 * Register happy-path REST/SSE route handlers. Specs can override any of these
 * by calling `page.route(...)` again before navigation (last handler wins).
 */
export async function installRestMocks(page: Page): Promise<void> {
  await page.route("**/api/metrics", (route) =>
    route.fulfill({ json: SERVER_METRICS }),
  );

  await page.route("**/api/scenario/remix", (route) =>
    route.fulfill({ json: REMIX_OK }),
  );

  await page.route("**/api/run/*", (route) =>
    route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: sseBody(DEMO_LINES),
    }),
  );

  // The three history endpoints the app polls on /live, /training and
  // /runs. Without these the fetches escape to the real apiBaseUrl
  // (http://localhost:8080), which nothing serves under `vite preview`,
  // so Chromium logs `net::ERR_CONNECTION_REFUSED` and the
  // "loads the shell without console errors" spec fails whenever that
  // rejection lands before its assertion — an intermittent failure with
  // no relation to the code under test.
  //
  // Empty arrays deliberately: they reproduce today's rendered state
  // exactly, so the existing empty-state assertions stay valid. Specs
  // that want populated data re-`route` these before navigating.
  for (const path of [
    "**/api/runs",
    "**/api/training-metrics/history*",
    "**/api/decision-traces/history*",
  ]) {
    await page.route(path, (route) => route.fulfill({ json: [] }));
  }
}
