import type { Page, Route } from "@playwright/test";
import { DEMO_LINES, REMIX_OK, SERVER_METRICS, sseBody } from "./mockData";

/**
 * Every backend route this E2E environment knows how to serve, mapped to
 * the response it serves.
 *
 * This map is the contract, not a convenience. `installRestMocks`
 * registers exactly these patterns and `installNetworkEscapeGuard` fails
 * any test whose page requests an off-origin URL matching none of them,
 * so both sides are derived from one declaration and cannot drift.
 * Adding a hook that polls a new endpoint therefore fails loudly and
 * immediately instead of escaping to the real `apiBaseUrl`.
 *
 * The three history endpoints answer with **empty arrays** deliberately:
 * that reproduces today's rendered state exactly, so the existing
 * empty-state assertions stay valid. Specs wanting populated data
 * re-`route` them before navigating (last handler wins).
 *
 * Patterns use Playwright's glob syntax, where `*` stops at a `/` and
 * `**` does not — see `globToRegExp` in `networkGuard.ts`.
 */
const API_MOCKS: Record<string, (route: Route) => Promise<void>> = {
  "**/api/metrics": (route) => route.fulfill({ json: SERVER_METRICS }),
  "**/api/scenario/remix": (route) => route.fulfill({ json: REMIX_OK }),
  "**/api/run/*": (route) =>
    route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: sseBody(DEMO_LINES),
    }),
  "**/api/runs": (route) => route.fulfill({ json: [] }),
  "**/api/training-metrics/history*": (route) => route.fulfill({ json: [] }),
  "**/api/decision-traces/history*": (route) => route.fulfill({ json: [] }),
};

/** The URL patterns {@link installRestMocks} serves. */
export const MOCKED_API_PATTERNS: readonly string[] = Object.keys(API_MOCKS);

/**
 * Register happy-path REST/SSE route handlers for every entry in
 * {@link MOCKED_API_PATTERNS}. Specs can override any of them by calling
 * `page.route(...)` again before navigation (last handler wins).
 */
export async function installRestMocks(page: Page): Promise<void> {
  for (const [pattern, handler] of Object.entries(API_MOCKS)) {
    await page.route(pattern, handler);
  }
}
