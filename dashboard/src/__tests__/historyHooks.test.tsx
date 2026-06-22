import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useDecisionTraces } from "../hooks/useDecisionTraces";
import { useRuns } from "../hooks/useRuns";
import { useTrainingHistory } from "../hooks/useTrainingHistory";
import type {
  RunSummary,
  TraceHistoryRecord,
  TrainingHistoryRecord,
} from "../types/simulation";

function jsonResponse(data: unknown): Response {
  return { ok: true, json: async () => data } as Response;
}

describe("useTrainingHistory", () => {
  afterEach(() => vi.restoreAllMocks());

  const record: TrainingHistoryRecord = {
    runId: "run-a",
    recordedAtMs: 1,
    episode: 3,
    totalSteps: 100,
    meanReward: 1.25,
    winRate: 0.5,
    curriculumDifficulty: 0.1,
    stepsPerSecond: 10,
    lossPolicy: 0.2,
    lossValue: 0.3,
    entropy: 0.4,
  };

  it("maps server records onto chart metrics (meanReward -> reward)", async () => {
    global.fetch = vi.fn().mockResolvedValue(jsonResponse([record]));
    const { result } = renderHook(() => useTrainingHistory());
    await waitFor(() => expect(result.current.history).toHaveLength(1));
    expect(result.current.history[0].reward).toBe(1.25);
    expect(result.current.history[0].episode).toBe(3);
    expect(result.current.error).toBeNull();
  });

  it("records an HTTP error", async () => {
    global.fetch = vi.fn().mockResolvedValue({ ok: false, status: 500 } as Response);
    const { result } = renderHook(() => useTrainingHistory());
    await waitFor(() => expect(result.current.error).toContain("500"));
  });

  it("records a network failure", async () => {
    global.fetch = vi.fn().mockRejectedValue(new Error("offline"));
    const { result } = renderHook(() => useTrainingHistory("run-a"));
    await waitFor(() => expect(result.current.error).toBe("offline"));
  });
});

describe("useRuns", () => {
  afterEach(() => vi.restoreAllMocks());

  const run: RunSummary = {
    runId: "run-a",
    startedAtMs: 1,
    lastSeenMs: 2,
    episodes: 4,
    latestMeanReward: 0.75,
  };

  it("exposes run summaries", async () => {
    global.fetch = vi.fn().mockResolvedValue(jsonResponse([run]));
    const { result } = renderHook(() => useRuns());
    await waitFor(() => expect(result.current.runs).toEqual([run]));
  });

  it("records an HTTP error", async () => {
    global.fetch = vi.fn().mockResolvedValue({ ok: false, status: 503 } as Response);
    const { result } = renderHook(() => useRuns());
    await waitFor(() => expect(result.current.error).toContain("503"));
  });
});

describe("useDecisionTraces", () => {
  afterEach(() => vi.restoreAllMocks());

  const trace: TraceHistoryRecord = {
    runId: "run-a",
    recordedAtMs: 1,
    agentId: 2,
    tick: 9,
    intentLabel: "explore",
    confidence: 0.9,
    searchDepth: 5,
    ucb1Score: 1.1,
    alternativesConsidered: 3,
  };

  it("maps server trace records onto panel entries", async () => {
    global.fetch = vi.fn().mockResolvedValue(jsonResponse([trace]));
    const { result } = renderHook(() => useDecisionTraces());
    await waitFor(() => expect(result.current.traces).toHaveLength(1));
    const entry = result.current.traces[0];
    expect(entry.intentLabel).toBe("explore");
    expect(entry.agentId).toBe(2);
    expect(entry.action).toBe(0); // server stores no action id
  });

  it("records a network failure", async () => {
    global.fetch = vi.fn().mockRejectedValue(new Error("down"));
    const { result } = renderHook(() => useDecisionTraces("run-a"));
    await waitFor(() => expect(result.current.error).toBe("down"));
  });
});
