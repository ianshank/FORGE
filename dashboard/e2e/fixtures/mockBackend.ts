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
}
