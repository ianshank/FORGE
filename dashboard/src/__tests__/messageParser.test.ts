import { describe, expect, it } from "vitest";
import { parseServerMessage } from "../utils/messageParser";

describe("parseServerMessage", () => {
  it("parses a valid StateUpdate message", () => {
    const data = {
      type: "StateUpdate",
      payload: {
        tick: 42,
        agents: [{ id: 1, x: 10, y: 20, health: 100, alive: true }],
        gridWidth: 64,
        gridHeight: 64,
        events: [],
        schemaVersion: 1,
      },
    };

    const result = parseServerMessage(data);
    expect(result).not.toBeNull();
    expect(result?.type).toBe("StateUpdate");
    if (result?.type === "StateUpdate") {
      expect(result.payload.tick).toBe(42);
      expect(result.payload.agents).toHaveLength(1);
    }
  });

  it("parses a valid Metrics message", () => {
    const data = {
      type: "Metrics",
      payload: {
        simulationTicks: 100,
        stepsPerSecond: 10.5,
        wsConnections: 3,
        uptimeSeconds: 60,
      },
    };

    const result = parseServerMessage(data);
    expect(result).not.toBeNull();
    expect(result?.type).toBe("Metrics");
    if (result?.type === "Metrics") {
      expect(result.payload.simulationTicks).toBe(100);
    }
  });

  it("parses a valid Error message", () => {
    const data = { type: "Error", payload: "something went wrong" };
    const result = parseServerMessage(data);
    expect(result).not.toBeNull();
    expect(result?.type).toBe("Error");
    if (result?.type === "Error") {
      expect(result.payload).toBe("something went wrong");
    }
  });

  it("returns null for non-object input", () => {
    expect(parseServerMessage("string")).toBeNull();
    expect(parseServerMessage(42)).toBeNull();
    expect(parseServerMessage(null)).toBeNull();
    expect(parseServerMessage(undefined)).toBeNull();
  });

  it("returns null for missing type field", () => {
    expect(parseServerMessage({ payload: {} })).toBeNull();
  });

  it("returns null for missing payload field", () => {
    expect(parseServerMessage({ type: "StateUpdate" })).toBeNull();
  });

  it("returns null for invalid StateUpdate payload", () => {
    const data = { type: "StateUpdate", payload: { tick: "not_a_number" } };
    expect(parseServerMessage(data)).toBeNull();
  });

  it("returns null for invalid Metrics payload", () => {
    const data = { type: "Metrics", payload: { invalid: true } };
    expect(parseServerMessage(data)).toBeNull();
  });

  it("returns null for unknown message type", () => {
    const data = { type: "Unknown", payload: {} };
    expect(parseServerMessage(data)).toBeNull();
  });

  it("returns null for Error with non-string payload", () => {
    const data = { type: "Error", payload: 42 };
    expect(parseServerMessage(data)).toBeNull();
  });
});
