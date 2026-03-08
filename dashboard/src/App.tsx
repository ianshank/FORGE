import { useState } from "react";
import { SimulationCanvas } from "./components/SimulationCanvas";
import { DecisionTracePanel } from "./components/DecisionTracePanel";
import { MetricsDashboard } from "./components/MetricsDashboard";
import { ScenarioControls } from "./components/ScenarioControls";
import { AgentInspector } from "./components/AgentInspector";
import { useSimulationState } from "./hooks/useSimulationState";
import type {
  AgentState,
  TrainingMetrics,
} from "./types/simulation";

/** Root application component — FORGE Dashboard. */
export function App() {
  const { state, connectionStatus, traces } = useSimulationState();
  const [selectedAgent, setSelectedAgent] = useState<AgentState | null>(null);
  const [metricsHistory] = useState<TrainingMetrics[]>([]);

  return (
    <div className="min-h-screen bg-gray-950 text-white flex flex-col">
      {/* Header */}
      <header className="border-b border-gray-800 px-4 py-2 flex items-center justify-between">
        <h1 className="text-lg font-bold">FORGE Dashboard</h1>
        <div className="flex items-center gap-3 text-sm">
          <span
            className={`inline-block w-2 h-2 rounded-full ${
              connectionStatus === "connected"
                ? "bg-green-500"
                : connectionStatus === "connecting"
                  ? "bg-yellow-500"
                  : "bg-red-500"
            }`}
          />
          <span className="text-gray-400">
            {connectionStatus === "connected"
              ? `Tick ${state?.tick ?? 0} | ${state?.agents.length ?? 0} agents`
              : connectionStatus}
          </span>
        </div>
      </header>

      {/* Controls */}
      <div className="px-4 py-2">
        <ScenarioControls />
      </div>

      {/* Main content */}
      <div className="flex-1 flex gap-4 px-4 pb-4 overflow-hidden">
        {/* Left: Canvas + Agent Inspector */}
        <div className="flex-1 flex flex-col gap-4 min-w-0">
          <SimulationCanvas state={state} />
          <AgentInspector agent={selectedAgent} />
        </div>

        {/* Right: Decision Traces */}
        <div className="w-80 flex-shrink-0">
          <DecisionTracePanel traces={traces} />
        </div>
      </div>

      {/* Bottom: Metrics */}
      <div className="px-4 pb-4">
        <MetricsDashboard history={metricsHistory} />
      </div>
    </div>
  );
}
