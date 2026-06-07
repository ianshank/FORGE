import { render, screen, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { MemoryRouter } from "react-router-dom";
import { act } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { AppShell } from "../components/layout/AppShell";
import { Sidebar } from "../components/layout/Sidebar";
import { TopBar } from "../components/layout/TopBar";
import { SimulationProvider } from "../context/SimulationContext";
import { MockWebSocket } from "../test/mockWebSocket";

function withProviders(ui: ReactNode, initial = "/live") {
  return (
    <MemoryRouter initialEntries={[initial]}>
      <SimulationProvider>{ui}</SimulationProvider>
    </MemoryRouter>
  );
}

describe("Sidebar", () => {
  it("renders all navigation entries", () => {
    render(<MemoryRouter><Sidebar /></MemoryRouter>);
    for (const label of ["Live", "Training", "Runs", "Replay", "Demos", "Settings"]) {
      expect(screen.getByRole("link", { name: label })).toBeInTheDocument();
    }
    expect(screen.getByText("FORGE")).toBeInTheDocument();
  });

  it("marks the active route", () => {
    render(
      <MemoryRouter initialEntries={["/training"]}>
        <Sidebar />
      </MemoryRouter>,
    );
    expect(screen.getByRole("link", { name: "Training" })).toHaveClass(
      "bg-accent",
    );
  });
});

describe("TopBar", () => {
  it("renders title and subtitle with a connection indicator", () => {
    render(withProviders(<TopBar title="Live" subtitle="Realtime" />));
    expect(screen.getByText("Live")).toBeInTheDocument();
    expect(screen.getByText("Realtime")).toBeInTheDocument();
    expect(screen.getByText("Connecting")).toBeInTheDocument();
  });

  it("shows tick/agent counts once connected", () => {
    render(withProviders(<TopBar title="Live" />));
    act(() => {
      const ws = MockWebSocket.last();
      ws?.emitOpen();
      ws?.emitMessage({
        type: "StateUpdate",
        payload: {
          tick: 12,
          agents: [{ id: 0, x: 0, y: 0, health: 1, alive: true, teamId: 0, intent: null, visionRadius: 1 }],
          gridWidth: 4,
          gridHeight: 4,
          events: [],
          schemaVersion: 1,
        },
      });
    });
    expect(screen.getByText("Connected")).toBeInTheDocument();
    expect(screen.getByText(/agents/)).toBeInTheDocument();
  });
});

describe("AppShell", () => {
  it("composes sidebar, top bar and content", () => {
    render(
      withProviders(
        <AppShell title="Demos" subtitle="run">
          <p>shell-body</p>
        </AppShell>,
      ),
    );
    const nav = screen.getByRole("navigation");
    expect(within(nav).getByRole("link", { name: "Live" })).toBeInTheDocument();
    expect(screen.getByText("shell-body")).toBeInTheDocument();
  });
});
