import { renderHook } from "@testing-library/react";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";
import {
  SimulationProvider,
  useSimulation,
} from "../context/SimulationContext";

describe("SimulationContext", () => {
  it("throws when used outside a provider", () => {
    // Silence the expected React error boundary logging.
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    expect(() => renderHook(() => useSimulation())).toThrow(
      /must be used within a SimulationProvider/,
    );
    spy.mockRestore();
  });

  it("provides state and connection status within a provider", () => {
    const wrapper = ({ children }: { children: ReactNode }) => (
      <SimulationProvider>{children}</SimulationProvider>
    );
    const { result } = renderHook(() => useSimulation(), { wrapper });
    expect(result.current.state).toBeNull();
    expect(result.current.connectionStatus).toBe("connecting");
  });
});
