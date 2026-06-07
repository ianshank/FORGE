import { test as base, expect } from "@playwright/test";
import { installRestMocks } from "./mockBackend";
import { STATE_UPDATE } from "./mockData";
import { installWebSocketMock } from "./mockWebSocket";

/**
 * Base test fixture: every test gets a page with the WebSocket stub installed
 * (auto-opens and emits an initial StateUpdate) and happy-path REST/SSE mocks
 * wired, so specs only navigate + assert. Error cases re-`route` before goto.
 */
export const test = base.extend({
  page: async ({ page }, use) => {
    await installWebSocketMock(page, STATE_UPDATE);
    await installRestMocks(page);
    await use(page);
  },
});

export { expect };
