import { Cpu, Users, Zap } from "lucide-react";
import { useState } from "react";
import { AgentInspector } from "../components/AgentInspector";
import { DecisionTracePanel } from "../components/DecisionTracePanel";
import { ScenarioControls } from "../components/ScenarioControls";
import { SimulationCanvas } from "../components/SimulationCanvas";
import { StatCard } from "../components/ui/stat-card";
import { useSimulation } from "../context/SimulationContext";
import { useMetrics } from "../hooks/useMetrics";
import { formatNumber } from "../lib/utils";
import type { AgentState, DecisionTraceEntry } from "../types/simulation";

/** Live simulation view: world canvas, agent inspector, traces, controls. */
export function LivePage() {
  const { state } = useSimulation();
  const { metrics } = useMetrics();
  const [selectedAgent, setSelectedAgent] = useState<AgentState | null>(null);
  // Live decision traces are not yet streamed over the wire; show empty state.
  const traces: DecisionTraceEntry[] = [];

  const aliveAgents = state?.agents.filter((a) => a.alive).length ?? 0;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <StatCard
          label="Tick"
          value={formatNumber(state?.tick ?? 0)}
          icon={Cpu}
          tone="primary"
        />
        <StatCard
          label="Agents Alive"
          value={`${aliveAgents}`}
          unit={`/ ${state?.agents.length ?? 0}`}
          icon={Users}
          tone="success"
        />
        <StatCard
          label="Steps / sec"
          value={formatNumber(metrics?.stepsPerSecond ?? 0, 1)}
          icon={Zap}
          tone="warning"
        />
        <StatCard
          label="WS Clients"
          value={formatNumber(metrics?.wsConnections ?? 0)}
          icon={Users}
        />
      </div>

      <ScenarioControls />

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
        <div className="space-y-6 xl:col-span-2">
          <SimulationCanvas
            state={state}
            selectedAgentId={selectedAgent?.id ?? null}
            onSelectAgent={setSelectedAgent}
          />
          <AgentInspector agent={selectedAgent} />
        </div>
        <div className="min-h-[24rem] xl:h-[calc(100vh-22rem)]">
          <DecisionTracePanel traces={traces} />
        </div>
      </div>
    </div>
  );
}
