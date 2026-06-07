/**
 * Global test setup: jest-dom matchers + jsdom polyfills that the component
 * tree (Recharts' ResponsiveContainer in particular) expects.
 */
import "@testing-library/jest-dom/vitest";
import { afterEach, beforeEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";
import { MockWebSocket } from "./mockWebSocket";

// jsdom lacks ResizeObserver, which Recharts' ResponsiveContainer relies on.
class ResizeObserverStub {
  observe(): void {}
  unobserve(): void {}
  disconnect(): void {}
}

if (!("ResizeObserver" in globalThis)) {
  (globalThis as { ResizeObserver?: unknown }).ResizeObserver =
    ResizeObserverStub;
}

// Recharts reads element dimensions; give it a non-zero box in jsdom.
if (!HTMLElement.prototype.getBoundingClientRect) {
  // no-op: jsdom provides this, kept for clarity.
}

// matchMedia is referenced by some UI libraries during render.
if (!window.matchMedia) {
  window.matchMedia = vi.fn().mockImplementation((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  }));
}

// Install the WebSocket stub and a no-op fetch so hooks never touch the
// network by default. Individual tests override `global.fetch` as needed.
(globalThis as { WebSocket?: unknown }).WebSocket = MockWebSocket;

beforeEach(() => {
  MockWebSocket.reset();
  // Default fetch never resolves, so polling hooks stay quiet unless a test
  // provides its own implementation.
  global.fetch = vi.fn(() => new Promise<Response>(() => {}));
});

// Ensure the DOM is reset between tests.
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});
