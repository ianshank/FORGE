import { fireEvent, render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { SimulationProvider } from "../context/SimulationContext";
import { LivePage } from "../pages/LivePage";
import { RunsPage } from "../pages/RunsPage";
import { SettingsPage } from "../pages/SettingsPage";
import { TrainingPage } from "../pages/TrainingPage";

function withProviders(ui: ReactNode) {
  return (
    <MemoryRouter>
      <SimulationProvider>{ui}</SimulationProvider>
    </MemoryRouter>
  );
}

describe("TrainingPage", () => {
  it("shows the curves tab empty state by default", () => {
    render(<TrainingPage />);
    expect(screen.getByText("No training metrics yet")).toBeInTheDocument();
  });

  it("switches to the compare tab", () => {
    render(<TrainingPage />);
    const compare = screen.getByRole("tab", { name: "Compare runs" });
    fireEvent.mouseDown(compare);
    fireEvent.click(compare);
    expect(screen.getByText("Select runs to compare")).toBeInTheDocument();
  });
});

describe("RunsPage", () => {
  it("renders the empty runs state", () => {
    render(<RunsPage />);
    expect(screen.getByText("No runs recorded")).toBeInTheDocument();
  });
});

describe("SettingsPage", () => {
  it("renders the resolved runtime configuration", () => {
    render(<SettingsPage />);
    expect(screen.getByText("Runtime Configuration")).toBeInTheDocument();
    expect(screen.getByText("VITE_WS_URL")).toBeInTheDocument();
    expect(screen.getByText("VITE_DEMO_API_BASE_URL")).toBeInTheDocument();
  });
});

describe("LivePage", () => {
  it("renders KPI cards, controls and the waiting canvas", () => {
    render(withProviders(<LivePage />));
    expect(screen.getByText("Tick")).toBeInTheDocument();
    expect(screen.getByText("Agents Alive")).toBeInTheDocument();
    expect(screen.getByText("Server Uptime")).toBeInTheDocument();
    expect(screen.getByText("Scenario")).toBeInTheDocument();
    expect(screen.getByText("Waiting for simulation")).toBeInTheDocument();
    expect(screen.getByText("No agent selected")).toBeInTheDocument();
  });
});
