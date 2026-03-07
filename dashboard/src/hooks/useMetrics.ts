import { useCallback, useEffect, useState } from "react";
import type { ServerMetrics } from "../types/simulation";
import { getConfig } from "../config/environment";

/** Hook for polling server metrics via REST API. */
export function useMetrics() {
  const config = getConfig();
  const [metrics, setMetrics] = useState<ServerMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchMetrics = useCallback(async () => {
    try {
      const res = await fetch(`${config.apiBaseUrl}/api/metrics`);
      if (res.ok) {
        const data = (await res.json()) as ServerMetrics;
        setMetrics(data);
        setError(null);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to fetch metrics");
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
