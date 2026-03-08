import { useCallback, useRef, useState } from "react";
import type {
  DecisionTraceEntry,
  ServerMetrics,
  SimulationState,
} from "../types/simulation";
import { useWebSocket } from "./useWebSocket";
import { getConfig } from "../config/environment";

interface SimulationHookResult {
  /** Current simulation state (latest from WebSocket). */
  state: SimulationState | null;
  /** WebSocket connection status. */
  connectionStatus: string;
  /** Accumulated decision trace entries. */
  traces: DecisionTraceEntry[];
  /** Server metrics (latest). */
  serverMetrics: ServerMetrics | null;
}

/** Hook for subscribing to simulation state updates via WebSocket. */
export function useSimulationState(): SimulationHookResult {
  const config = getConfig();
  const [state, setState] = useState<SimulationState | null>(null);
  const [serverMetrics, setServerMetrics] = useState<ServerMetrics | null>(null);
  const [traces, setTraces] = useState<DecisionTraceEntry[]>([]);
  const maxTraces = useRef(config.maxTraceEntries);

  const onMessage = useCallback((data: unknown) => {
    const msg = data as {
      type?: string;
      payload?: SimulationState | ServerMetrics;
    };
    if (msg.type === "StateUpdate" && msg.payload) {
      setState(msg.payload as SimulationState);
    } else if (msg.type === "Metrics" && msg.payload) {
      setServerMetrics(msg.payload as ServerMetrics);
    }
  }, []);

  const ws = useWebSocket({ url: config.wsUrl, onMessage });

  return {
    state,
    connectionStatus: ws.status,
    traces,
    serverMetrics,
  };
}
