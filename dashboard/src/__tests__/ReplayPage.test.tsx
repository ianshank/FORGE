import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ReplayPage } from "../pages/ReplayPage";

function makeTrajectoryJson(stepCount: number): string {
  const steps = Array.from({ length: stepCount }, (_, i) => ({
    tick: i,
    obs: [0, 0],
    action_id: i % 3,
    policy_target: [0.2, 0.3, 0.5],
    value_target: 0.1 * i,
    reward: 1,
    terminated: i === stepCount - 1,
    truncated: false,
  }));
  return JSON.stringify({
    format_version: 2,
    env_id: "minecraft",
    schema_id: "abc",
    episode_id: "ep-test",
    seed: 7,
    obs_dim: 2,
    action_count: 3,
    final_reward: stepCount,
    started_at: "2026-01-01T00:00:00Z",
    ended_at: "2026-01-01T00:01:00Z",
    steps,
  });
}

function fileInput(container: HTMLElement): HTMLInputElement {
  const input = container.querySelector('input[type="file"]');
  if (!input) throw new Error("file input not found");
  return input as HTMLInputElement;
}

/**
 * jsdom's File implementation does not reliably provide `.text()`, so we hand
 * the change event a minimal file-like object that does.
 */
function fakeFile(content: string, name = "traj.json"): File {
  return {
    name,
    type: "application/json",
    text: () => Promise.resolve(content),
  } as unknown as File;
}

async function loadTrajectory(container: HTMLElement, json: string) {
  fireEvent.change(fileInput(container), {
    target: { files: [fakeFile(json)] },
  });
  await screen.findByText(/ep-test/);
}

describe("ReplayPage", () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("shows the empty state before a trajectory is loaded", () => {
    render(<ReplayPage />);
    expect(screen.getByText("Load a trajectory to replay")).toBeInTheDocument();
  });

  it("surfaces a parse error for invalid JSON", async () => {
    const { container } = render(<ReplayPage />);
    fireEvent.change(fileInput(container), {
      target: { files: [fakeFile("{bad json")] },
    });
    expect(await screen.findByRole("alert")).toHaveTextContent(/parse JSON/i);
  });

  it("rejects a trajectory with no steps", async () => {
    const { container } = render(<ReplayPage />);
    fireEvent.change(fileInput(container), {
      target: { files: [fakeFile(JSON.stringify({ env_id: "x", steps: [] }))] },
    });
    expect(await screen.findByRole("alert")).toHaveTextContent(/no steps/i);
  });

  it("loads a trajectory and renders summary + step detail", async () => {
    const { container } = render(<ReplayPage />);
    await loadTrajectory(container, makeTrajectoryJson(4));
    expect(screen.getByText("Steps")).toBeInTheDocument();
    expect(screen.getByText("1 / 4")).toBeInTheDocument();
    expect(screen.getByText("Step Detail")).toBeInTheDocument();
  });

  it("steps forward and backward through the timeline", async () => {
    const { container } = render(<ReplayPage />);
    await loadTrajectory(container, makeTrajectoryJson(4));

    fireEvent.click(screen.getByRole("button", { name: "Step forward" }));
    expect(screen.getByText("2 / 4")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Step forward" }));
    expect(screen.getByText("3 / 4")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Step back" }));
    expect(screen.getByText("2 / 4")).toBeInTheDocument();

    // A timeline slider is present for scrubbing.
    expect(screen.getByRole("slider")).toBeInTheDocument();
  });

  it("plays through steps on a timer and changes speed", async () => {
    const { container } = render(<ReplayPage />);
    await loadTrajectory(container, makeTrajectoryJson(5));

    // Bump speed (covers the speed buttons).
    fireEvent.click(screen.getByRole("button", { name: "10×" }));

    vi.useFakeTimers();
    fireEvent.click(screen.getByRole("button", { name: "Play" }));
    // 10 steps/sec → 1s of ticks runs well past the 5-step trajectory.
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(screen.getByText("5 / 5")).toBeInTheDocument();
  });
});
