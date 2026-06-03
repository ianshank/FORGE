import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { AgentInspector } from "../components/AgentInspector";
import { DecisionTracePanel } from "../components/DecisionTracePanel";
import { MetricsDashboard } from "../components/MetricsDashboard";
import { ScenarioControls } from "../components/ScenarioControls";
import { SimulationCanvas } from "../components/SimulationCanvas";
import { HEALTH_FIXED_POINT_SCALE } from "../lib/canvasRender";
import type {
  AgentState,
  DecisionTraceEntry,
  SimulationState,
  TrainingMetrics,
} from "../types/simulation";

function makeAgent(p: Partial<AgentState> & Pick<AgentState, "id" | "x" | "y">): AgentState {
  return {
    health: HEALTH_FIXED_POINT_SCALE,
    alive: true,
    teamId: 0,
    intent: null,
    visionRadius: 3,
    ...p,
  };
}

describe("AgentInspector", () => {
  it("shows an empty state when no agent is selected", () => {
    render(<AgentInspector agent={null} />);
    expect(screen.getByText("No agent selected")).toBeInTheDocument();
  });

  it("renders an alive agent's fields including intent", () => {
    render(
      <AgentInspector
        agent={makeAgent({ id: 7, x: 2, y: 3, intent: "gather", teamId: 1 })}
      />,
    );
    expect(screen.getByText("#7")).toBeInTheDocument();
    expect(screen.getByText("(2, 3)")).toBeInTheDocument();
    expect(screen.getByText("Alive")).toBeInTheDocument();
    expect(screen.getByText("gather")).toBeInTheDocument();
  });

  it("renders a dead agent with no team", () => {
    render(
      <AgentInspector agent={makeAgent({ id: 1, x: 0, y: 0, alive: false, teamId: null })} />,
    );
    expect(screen.getByText("Dead")).toBeInTheDocument();
    expect(screen.getByText("None")).toBeInTheDocument();
  });
});

describe("DecisionTracePanel", () => {
  it("shows an empty state with no traces", () => {
    render(<DecisionTracePanel traces={[]} />);
    expect(screen.getByText("No traces yet")).toBeInTheDocument();
  });

  it("renders the most recent traces, capped by maxEntries", () => {
    const traces: DecisionTraceEntry[] = Array.from({ length: 5 }, (_, i) => ({
      tick: i,
      agentId: i % 2,
      action: i,
      confidence: 0.5,
      searchDepth: 2,
      ucb1Score: 0.1,
      intentLabel: `intent-${i}`,
    }));
    render(<DecisionTracePanel traces={traces} maxEntries={2} />);
    // Newest first; only the last 2 entries shown.
    expect(screen.getByText("intent-4")).toBeInTheDocument();
    expect(screen.getByText("intent-3")).toBeInTheDocument();
    expect(screen.queryByText("intent-0")).not.toBeInTheDocument();
    // Total count badge reflects all traces.
    expect(screen.getByText("5")).toBeInTheDocument();
  });
});

describe("MetricsDashboard", () => {
  it("shows an empty state with no history", () => {
    render(<MetricsDashboard history={[]} />);
    expect(screen.getByText("No training metrics yet")).toBeInTheDocument();
  });

  it("renders one chart per metric with the latest value", () => {
    const history: TrainingMetrics[] = [
      { episode: 1, reward: 1, winRate: 0.2, stepsPerSecond: 10, lossPolicy: 0.1, lossValue: 0.2, entropy: 0.5 },
      { episode: 2, reward: 2, winRate: 0.4, stepsPerSecond: 12, lossPolicy: 0.1, lossValue: 0.2, entropy: 0.4 },
    ];
    render(<MetricsDashboard history={history} />);
    expect(screen.getByText("Reward")).toBeInTheDocument();
    expect(screen.getByText("Policy Entropy")).toBeInTheDocument();
    // Latest reward 2.00 is shown in the header.
    expect(screen.getByText("2.00")).toBeInTheDocument();
  });
});

describe("ScenarioControls", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("posts a remix and invokes onRemix on success", async () => {
    global.fetch = vi.fn().mockResolvedValue({ ok: true, status: 200 } as Response);
    const onRemix = vi.fn();
    render(<ScenarioControls onRemix={onRemix} />);

    fireEvent.change(screen.getByLabelText("Random seed"), {
      target: { value: "99" },
    });
    fireEvent.click(screen.getByRole("button", { name: /Remix/ }));

    await waitFor(() => expect(onRemix).toHaveBeenCalledWith(99));
  });

  it("surfaces an HTTP error", async () => {
    global.fetch = vi.fn().mockResolvedValue({ ok: false, status: 500 } as Response);
    render(<ScenarioControls />);
    fireEvent.click(screen.getByRole("button", { name: /Remix/ }));
    expect(await screen.findByRole("alert")).toHaveTextContent("HTTP 500");
  });

  it("surfaces a network failure", async () => {
    global.fetch = vi.fn().mockRejectedValue(new Error("offline"));
    render(<ScenarioControls />);
    fireEvent.click(screen.getByRole("button", { name: /Remix/ }));
    expect(await screen.findByRole("alert")).toHaveTextContent("offline");
  });
});

describe("SimulationCanvas", () => {
  function fakeContext() {
    return {
      fillStyle: "",
      strokeStyle: "",
      lineWidth: 0,
      font: "",
      textAlign: "" as CanvasTextAlign,
      fillRect: vi.fn(),
      beginPath: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      stroke: vi.fn(),
      arc: vi.fn(),
      fill: vi.fn(),
      fillText: vi.fn(),
    };
  }

  const state: SimulationState = {
    tick: 1,
    gridWidth: 4,
    gridHeight: 4,
    events: [],
    schemaVersion: 1,
    agents: [
      makeAgent({ id: 0, x: 1, y: 1, intent: "move", teamId: 0 }),
      makeAgent({ id: 1, x: 3, y: 3, alive: false, health: 0 }),
      makeAgent({ id: 2, x: 2, y: 2, health: HEALTH_FIXED_POINT_SCALE / 4 }),
    ],
  };

  beforeEach(() => {
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(
      fakeContext() as unknown as CanvasRenderingContext2D,
    );
    vi.spyOn(HTMLCanvasElement.prototype, "getBoundingClientRect").mockReturnValue({
      left: 0,
      top: 0,
      width: 32,
      height: 32,
      right: 32,
      bottom: 32,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    } as DOMRect);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders an empty state with no simulation", () => {
    render(<SimulationCanvas state={null} />);
    expect(screen.getByText("Waiting for simulation")).toBeInTheDocument();
  });

  it("draws the grid and agents, and selects on click", () => {
    const onSelectAgent = vi.fn();
    const { container } = render(
      <SimulationCanvas
        state={state}
        selectedAgentId={0}
        onSelectAgent={onSelectAgent}
      />,
    );
    const canvas = container.querySelector("canvas");
    expect(canvas).toBeTruthy();
    // Click over agent 0's cell centre (cellSize 8 → centre of (1,1) = 12,12).
    fireEvent.click(canvas as HTMLCanvasElement, { clientX: 12, clientY: 12 });
    expect(onSelectAgent).toHaveBeenCalled();
    expect(onSelectAgent.mock.calls[0][0].id).toBe(0);
  });
});
