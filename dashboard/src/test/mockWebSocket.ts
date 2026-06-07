/**
 * Controllable WebSocket stub for tests.
 *
 * Installed as the global `WebSocket` by the test setup so hooks never open a
 * real socket. Tests drive lifecycle events via the `emit*` helpers and inspect
 * `MockWebSocket.instances`.
 */

type Listener = ((event: unknown) => void) | null;

export class MockWebSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;

  /** Every instance created since the last {@link reset}. */
  static instances: MockWebSocket[] = [];

  static reset(): void {
    MockWebSocket.instances = [];
  }

  /** Most recently constructed instance, or undefined. */
  static last(): MockWebSocket | undefined {
    return MockWebSocket.instances[MockWebSocket.instances.length - 1];
  }

  readonly url: string;
  readyState: number = MockWebSocket.CONNECTING;
  onopen: Listener = null;
  onclose: Listener = null;
  onerror: Listener = null;
  onmessage: Listener = null;

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
  }

  send(): void {}

  close(): void {
    this.readyState = MockWebSocket.CLOSED;
    this.onclose?.({ code: 1000, reason: "closed" });
  }

  // --- test drivers -------------------------------------------------------

  emitOpen(): void {
    this.readyState = MockWebSocket.OPEN;
    this.onopen?.({});
  }

  emitMessage(data: unknown): void {
    const payload = typeof data === "string" ? data : JSON.stringify(data);
    this.onmessage?.({ data: payload });
  }

  emitError(): void {
    this.onerror?.({});
  }

  /** Simulate a server-initiated close. */
  emitClose(code = 1006, reason = "lost"): void {
    this.readyState = MockWebSocket.CLOSED;
    this.onclose?.({ code, reason });
  }
}
