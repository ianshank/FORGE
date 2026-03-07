import type { AgentState } from "../types/simulation";

interface AgentInspectorProps {
  agent: AgentState | null;
}

/** Detailed inspector panel for a single agent. */
export function AgentInspector({ agent }: AgentInspectorProps) {
  if (!agent) {
    return (
      <div className="bg-gray-900 border border-gray-700 rounded p-3">
        <p className="text-gray-500 text-sm italic">
          Click an agent to inspect
        </p>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 border border-gray-700 rounded p-3 text-sm">
      <h3 className="font-bold text-gray-300 mb-2">
        Agent {agent.id}
        <span
          className={`ml-2 text-xs px-1 rounded ${
            agent.alive ? "bg-green-900 text-green-300" : "bg-red-900 text-red-300"
          }`}
        >
          {agent.alive ? "ALIVE" : "DEAD"}
        </span>
      </h3>
      <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-gray-400">
        <dt>Position</dt>
        <dd className="text-white">
          ({agent.x}, {agent.y})
        </dd>
        <dt>Health</dt>
        <dd className="text-white">{agent.health}</dd>
        <dt>Team</dt>
        <dd className="text-white">{agent.teamId ?? "None"}</dd>
        <dt>Vision</dt>
        <dd className="text-white">{agent.visionRadius}</dd>
        {agent.intent && (
          <>
            <dt>Intent</dt>
            <dd className="text-yellow-400">{agent.intent}</dd>
          </>
        )}
      </dl>
    </div>
  );
}
