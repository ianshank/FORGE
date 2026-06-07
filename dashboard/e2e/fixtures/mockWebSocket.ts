import type { Page } from "@playwright/test";

/**
 * Install a deterministic fake `window.WebSocket` via `addInitScript` (runs
 * before the app's `new WebSocket(url)`), plus page-global drivers so specs can
 * push frames / close / error on demand.
 *
 * Chosen over `page.routeWebSocket()` for determinism and version-stability;
 * the implementation mirrors the proven `src/test/mockWebSocket.ts` contract.
 */
export async function installWebSocketMock(
  page: Page,
  initialMessage?: unknown,
): Promise<void> {
  // Opt-in lifecycle logging surfaces in Playwright traces (set E2E_DEBUG=1)
  // to diagnose the WebSocket path, which is the most timing-sensitive mock.
  const debug = Boolean(process.env.E2E_DEBUG);

  await page.addInitScript(
    ({ initial, debug: dbg }) => {
      const log = (...args: unknown[]) => {
        if (dbg) console.debug("[e2e:ws]", ...args);
      };

      class MockWS {
        static readonly CONNECTING = 0;
        static readonly OPEN = 1;
        static readonly CLOSING = 2;
        static readonly CLOSED = 3;

        url: string;
        readyState = MockWS.CONNECTING;
        onopen: ((e: unknown) => void) | null = null;
        onclose: ((e: unknown) => void) | null = null;
        onerror: ((e: unknown) => void) | null = null;
        onmessage: ((e: unknown) => void) | null = null;

        constructor(url: string) {
          this.url = url;
          log("construct", url);
          const w = window as unknown as { __E2E_WS__: MockWS[] };
          w.__E2E_WS__ = w.__E2E_WS__ ?? [];
          w.__E2E_WS__.push(this);
          // Open on the next tick so the app can assign on* handlers first.
          setTimeout(() => {
            this.readyState = MockWS.OPEN;
            log("open");
            this.onopen?.({});
            if (initial !== undefined) {
              log("emit initial");
              this.onmessage?.({ data: JSON.stringify(initial) });
            }
          }, 0);
        }

        send(): void {}

        close(): void {
          log("close");
          this.readyState = MockWS.CLOSED;
          this.onclose?.({ code: 1000, reason: "closed" });
        }
      }

      (window as unknown as { WebSocket: unknown }).WebSocket = MockWS;

      const sockets = () =>
        (window as unknown as { __E2E_WS__?: MockWS[] }).__E2E_WS__ ?? [];

      // Drivers used by specs (via page.evaluate).
      (window as unknown as Record<string, unknown>).__E2E_WS_EMIT__ = (
        msg: unknown,
      ) => {
        log("emit", msg);
        for (const ws of sockets()) {
          if (ws.readyState === MockWS.OPEN) {
            ws.onmessage?.({ data: JSON.stringify(msg) });
          }
        }
      };
      (window as unknown as Record<string, unknown>).__E2E_WS_CLOSE__ = () => {
        log("close-all");
        for (const ws of sockets()) {
          ws.readyState = MockWS.CLOSED;
          ws.onclose?.({ code: 1006, reason: "lost" });
        }
      };
      (window as unknown as Record<string, unknown>).__E2E_WS_ERROR__ = () => {
        log("error-all");
        for (const ws of sockets()) ws.onerror?.({});
      };
    },
    { initial: initialMessage, debug },
  );
}

/** Push a WebSocket message into the running page. */
export async function emitWsMessage(page: Page, msg: unknown): Promise<void> {
  await page.evaluate((m) => {
    (
      window as unknown as { __E2E_WS_EMIT__?: (x: unknown) => void }
    ).__E2E_WS_EMIT__?.(m);
  }, msg);
}
