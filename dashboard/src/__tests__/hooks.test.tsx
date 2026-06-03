import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useMetrics } from "../hooks/useMetrics";
import { useSimulationState } from "../hooks/useSimulationState";
import { useWebSocket } from "../hooks/useWebSocket";
import { MockWebSocket } from "../test/mockWebSocket";
import type { ServerMetrics, SimulationState } from "../types/simulation";

describe("useWebSocket", () => {
  it("connects and reports an open connection", () => {
    const { result } = renderHook(() =>
      useWebSocket({ url: "ws://test/ws" }),
    );
    expect(MockWebSocket.instances).toHaveLength(1);
    act(() => MockWebSocket.last()?.emitOpen());
    expect(result.current.status).toBe("connected");
  });

  it("delivers parsed messages to onMessage", () => {
    const onMessage = vi.fn();
    renderHook(() => useWebSocket({ url: "ws://test/ws", onMessage }));
    act(() => MockWebSocket.last()?.emitMessage({ hello: "world" }));
    expect(onMessage).toHaveBeenCalledWith({ hello: "world" });
  });

  it("ignores malformed message payloads", () => {
    const onMessage = vi.fn();
    renderHook(() => useWebSocket({ url: "ws://test/ws", onMessage }));
    act(() => MockWebSocket.last()?.onmessage?.({ data: "{not json" } as never));
    expect(onMessage).not.toHaveBeenCalled();
  });

  it("closes the socket on manual disconnect", () => {
    const { result } = renderHook(() => useWebSocket({ url: "ws://test/ws" }));
    act(() => result.current.disconnect());
    expect(result.current.status).toBe("disconnected");
  });

  it("schedules a reconnect after an unexpected close", () => {
    vi.useFakeTimers();
    try {
      renderHook(() =>
        useWebSocket({ url: "ws://test/ws", reconnectInterval: 100 }),
      );
      expect(MockWebSocket.instances).toHaveLength(1);
      act(() => MockWebSocket.last()?.emitClose());
      act(() => vi.advanceTimersByTime(100));
      expect(MockWebSocket.instances.length).toBeGreaterThanOrEqual(2);
    } finally {
      vi.useRealTimers();
    }
  });

  it("handles a constructor failure gracefully", () => {
    const original = globalThis.WebSocket;
    (globalThis as { WebSocket: unknown }).WebSocket = class {
      constructor() {
        throw new Error("boom");
      }
    };
    try {
      const { result } = renderHook(() =>
        useWebSocket({ url: "ws://bad", reconnectInterval: 100 }),
      );
      expect(result.current.status).toBe("disconnected");
    } finally {
      (globalThis as { WebSocket: unknown }).WebSocket = original;
    }
  });
});

describe("useMetrics", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  const sample: ServerMetrics = {
    simulationTicks: 5,
    stepsPerSecond: 2,
    wsConnections: 1,
    uptimeSeconds: 10,
  };

  it("fetches and exposes server metrics", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => sample,
    } as Response);
    const { result } = renderHook(() => useMetrics());
    await waitFor(() => expect(result.current.metrics).toEqual(sample));
    expect(result.current.error).toBeNull();
  });

  it("records an HTTP error", async () => {
    global.fetch = vi.fn().mockResolvedValue({ ok: false, status: 503 } as Response);
    const { result } = renderHook(() => useMetrics());
    await waitFor(() => expect(result.current.error).toContain("503"));
  });

  it("records a network failure", async () => {
    global.fetch = vi.fn().mockRejectedValue(new Error("down"));
    const { result } = renderHook(() => useMetrics());
    await waitFor(() => expect(result.current.error).toBe("down"));
  });
});

describe("useSimulationState", () => {
  it("applies StateUpdate messages to state", () => {
    const { result } = renderHook(() => useSimulationState());
    const payload: SimulationState = {
      tick: 9,
      agents: [],
      gridWidth: 8,
      gridHeight: 8,
      events: [],
      schemaVersion: 1,
    };
    act(() =>
      MockWebSocket.last()?.emitMessage({ type: "StateUpdate", payload }),
    );
    expect(result.current.state?.tick).toBe(9);
  });

  it("ignores unparseable messages and logs server errors", () => {
    const { result } = renderHook(() => useSimulationState());
    act(() => MockWebSocket.last()?.emitMessage({ nope: true }));
    act(() =>
      MockWebSocket.last()?.emitMessage({ type: "Error", payload: "boom" }),
    );
    expect(result.current.state).toBeNull();
  });
});
