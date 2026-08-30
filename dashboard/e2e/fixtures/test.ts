import { test as base, expect } from "@playwright/test";
import { installRestMocks } from "./mockBackend";
import { STATE_UPDATE } from "./mockData";
import { installWebSocketMock } from "./mockWebSocket";
import { describeEscapes, installNetworkEscapeGuard } from "./networkGuard";

/**
 * Base test fixture: every test gets a page with the WebSocket stub installed
 * (auto-opens and emits an initial StateUpdate) and happy-path REST/SSE mocks
 * wired, so specs only navigate + assert. Error cases re-`route` before goto.
 *
 * Every test also carries the network-escape guard, and fails at teardown if
 * the page reached a real backend. It is on by default rather than opt-in
 * because the defect it catches — an endpoint nobody mocked — surfaces as an
 * intermittent failure in whichever spec happens to observe the rejection,
 * never in the spec that introduced it.
 */
export const test = base.extend({
  page: async ({ page, baseURL }, use) => {
    await installWebSocketMock(page, STATE_UPDATE);
    await installRestMocks(page);
    const escaped = installNetworkEscapeGuard(page, baseURL);

    await use(page);

    expect(escaped, describeEscapes(escaped)).toEqual([]);
  },
});

export { expect };
