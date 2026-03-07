import type { DecisionTraceEntry } from "../types/simulation";

interface DecisionTracePanelProps {
  traces: DecisionTraceEntry[];
  maxEntries?: number;
}

/** Panel displaying live decision trace entries from agents. */
export function DecisionTracePanel({
  traces,
  maxEntries = 100,
}: DecisionTracePanelProps) {
  const displayed = traces.slice(-maxEntries);

  return (
    <div className="bg-gray-900 border border-gray-700 rounded p-3 h-full overflow-hidden flex flex-col">
      <h3 className="text-sm font-bold text-gray-300 mb-2">
        Decision Traces ({traces.length})
      </h3>
      <div className="flex-1 overflow-y-auto text-xs font-mono space-y-1">
        {displayed.length === 0 ? (
          <p className="text-gray-500 italic">No traces yet</p>
        ) : (
          displayed.map((trace, i) => (
            <div
              key={`${trace.tick}-${trace.agentId}-${i}`}
              className="flex items-center gap-2 text-gray-400 hover:bg-gray-800 px-1 rounded"
            >
              <span className="text-gray-600 w-12">t={trace.tick}</span>
              <span className="text-blue-400 w-8">A{trace.agentId}</span>
              <span className="text-yellow-400 w-16 truncate">
                {trace.intentLabel}
              </span>
              <span className="text-green-400 w-12">
                {(trace.confidence * 100).toFixed(0)}%
              </span>
              <span className="text-gray-500 w-10">
                d={trace.searchDepth}
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
