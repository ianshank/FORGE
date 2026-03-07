import { useCallback, useState } from "react";
import type { SimulationState } from "../types/simulation";
import { useWebSocket } from "./useWebSocket";
import { getConfig } from "../config/environment";

/** Hook for subscribing to simulation state updates via WebSocket. */
export function useSimulationState() {
  const config = getConfig();
  const [state, setState] = useState<SimulationState | null>(null);

  const onMessage = useCallback((data: unknown) => {
    const msg = data as { type?: string; payload?: SimulationState };
    if (msg.type === "StateUpdate" && msg.payload) {
      setState(msg.payload);
    }
  }, []);

  const ws = useWebSocket({ url: config.wsUrl, onMessage });

  return { state, connectionStatus: ws.status };
}
