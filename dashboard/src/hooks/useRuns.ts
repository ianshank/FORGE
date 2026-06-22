import { useCallback, useEffect, useState } from "react";
import { getConfig } from "../config/environment";
import type { RunSummary } from "../types/simulation";
import { createLogger } from "../utils/logger";

const log = createLogger("useRuns");

/**
 * Poll `GET /api/runs` and expose the run summaries. Mirrors {@link useMetrics}:
 * config-driven interval, REST polling, error state.
 */
export function useRuns() {
  const config = getConfig();
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  const fetchRuns = useCallback(async () => {
    try {
      const res = await fetch(`${config.apiBaseUrl}/api/runs`);
      if (!res.ok) {
        const msg = `Runs fetch failed: HTTP ${res.status}`;
        log.warn(msg);
        setError(msg);
        return;
      }
      const data = (await res.json()) as RunSummary[];
      setRuns(data);
      setError(null);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Failed to fetch runs";
      log.debug("Runs fetch error:", msg);
      setError(msg);
    }
  }, [config.apiBaseUrl]);

  useEffect(() => {
    void fetchRuns();
    const interval = setInterval(() => void fetchRuns(), config.runsInterval);
    return () => clearInterval(interval);
  }, [fetchRuns, config.runsInterval]);

  return { runs, error };
}
