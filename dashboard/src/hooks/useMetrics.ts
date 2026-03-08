import { useCallback, useEffect, useState } from "react";
import type { ServerMetrics } from "../types/simulation";
import { getConfig } from "../config/environment";
import { createLogger } from "../utils/logger";

const log = createLogger("useMetrics");

/** Hook for polling server metrics via REST API. */
export function useMetrics() {
  const config = getConfig();
  const [metrics, setMetrics] = useState<ServerMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchMetrics = useCallback(async () => {
    try {
      const res = await fetch(`${config.apiBaseUrl}/api/metrics`);
      if (!res.ok) {
        const msg = `Metrics fetch failed: HTTP ${res.status}`;
        log.warn(msg);
        setError(msg);
        return;
      }
      const data = (await res.json()) as ServerMetrics;
      setMetrics(data);
      setError(null);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Failed to fetch metrics";
      log.debug("Metrics fetch error:", msg);
      setError(msg);
    }
  }, [config.apiBaseUrl]);

  useEffect(() => {
    void fetchMetrics();
    const interval = setInterval(
      () => void fetchMetrics(),
      config.metricsPollingInterval,
    );
    return () => clearInterval(interval);
  }, [fetchMetrics, config.metricsPollingInterval]);

  return { metrics, error };
}
