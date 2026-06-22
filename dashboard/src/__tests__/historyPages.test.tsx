import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { RunsPage } from "../pages/RunsPage";
import { TrainingPage } from "../pages/TrainingPage";
import type { RunSummary, TrainingHistoryRecord } from "../types/simulation";

function jsonResponse(data: unknown): Response {
  return { ok: true, json: async () => data } as Response;
}

describe("RunsPage (populated)", () => {
  afterEach(() => vi.restoreAllMocks());

  it("renders a table row per run when the runs API returns data", async () => {
    const runs: RunSummary[] = [
      {
        runId: "run-xyz",
        startedAtMs: 1_700_000_000_000,
        lastSeenMs: 1_700_000_100_000,
        episodes: 12,
        latestMeanReward: 3.5,
      },
    ];
    global.fetch = vi.fn().mockResolvedValue(jsonResponse(runs));
    render(<RunsPage />);
    expect(await screen.findByText("run-xyz")).toBeInTheDocument();
    expect(screen.getByText("Latest Mean Reward")).toBeInTheDocument();
    // The empty state must not be shown once rows exist.
    expect(screen.queryByText("No runs recorded")).not.toBeInTheDocument();
  });
});

describe("TrainingPage (populated)", () => {
  afterEach(() => vi.restoreAllMocks());

  it("renders metric charts once history is available", async () => {
    const records: TrainingHistoryRecord[] = [
      {
        runId: "r",
        recordedAtMs: 1,
        episode: 1,
        totalSteps: 10,
        meanReward: 0.5,
        winRate: 0.4,
        curriculumDifficulty: 0.1,
        stepsPerSecond: 5,
        lossPolicy: 0.2,
        lossValue: 0.3,
        entropy: 0.6,
      },
    ];
    global.fetch = vi.fn().mockResolvedValue(jsonResponse(records));
    render(<TrainingPage />);
    // MetricsDashboard renders a "Reward" chart card once history is non-empty.
    await waitFor(() =>
      expect(screen.queryByText("No training metrics yet")).not.toBeInTheDocument(),
    );
    expect(screen.getByText("Reward")).toBeInTheDocument();
  });
});
