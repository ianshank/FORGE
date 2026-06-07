import { createContext, useContext, type ReactNode } from "react";
import { useSimulationState } from "../hooks/useSimulationState";
import type { ConnectionStatus } from "../hooks/useWebSocket";
import type { SimulationState } from "../types/simulation";

interface SimulationContextValue {
  /** Latest simulation state from the WebSocket, or null before first message. */
  state: SimulationState | null;
  /** Current WebSocket connection status. */
  connectionStatus: ConnectionStatus;
}

const SimulationContext = createContext<SimulationContextValue | null>(null);

/**
 * Provides a single shared simulation WebSocket subscription to the whole
 * app, so the top bar and live view read from one connection.
 */
export function SimulationProvider({ children }: { children: ReactNode }) {
  const value = useSimulationState();
  return (
    <SimulationContext.Provider value={value}>
      {children}
    </SimulationContext.Provider>
  );
}

/** Access the shared simulation state. Must be used within a provider. */
export function useSimulation(): SimulationContextValue {
  const ctx = useContext(SimulationContext);
  if (!ctx) {
    throw new Error("useSimulation must be used within a SimulationProvider");
  }
  return ctx;
}
