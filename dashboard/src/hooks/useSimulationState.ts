import { useCallback, useRef, useState } from "react";
import type {
  DecisionTraceEntry,
  ServerMetrics,
  SimulationState,
} from "../types/simulation";
import { createLogger } from "../utils/logger";
import { parseServerMessage } from "../utils/messageParser";
import { useWebSocket } from "./useWebSocket";
import { getConfig } from "../config/environment";

const log = createLogger("useSimulationState");

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
    const msg = parseServerMessage(data);
    if (!msg) {
      log.debug("Skipping unparseable message");
      return;
    }

    switch (msg.type) {
      case "StateUpdate":
        setState(msg.payload);
        break;
      case "Metrics":
        setServerMetrics(msg.payload);
        break;
      case "Error":
        log.error("Server error:", msg.payload);
        break;
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
