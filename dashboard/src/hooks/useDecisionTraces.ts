import { useCallback, useEffect, useState } from "react";
import { getConfig } from "../config/environment";
import type { DecisionTraceEntry, TraceHistoryRecord } from "../types/simulation";
import { createLogger } from "../utils/logger";

const log = createLogger("useDecisionTraces");

/**
 * Map a persisted server trace record onto the panel-facing
 * `DecisionTraceEntry`. The server does not store the selected `action` id, so
 * it defaults to 0; all other fields map directly.
 */
function toEntry(rec: TraceHistoryRecord): DecisionTraceEntry {
  return {
    tick: rec.tick,
    agentId: rec.agentId,
    action: 0,
    confidence: rec.confidence,
    searchDepth: rec.searchDepth,
    ucb1Score: rec.ucb1Score,
    intentLabel: rec.intentLabel,
  };
}

/**
 * Poll `GET /api/decision-traces/history` and expose recent decision traces for
 * the trace panel. Mirrors {@link useMetrics}: config-driven interval, REST
 * polling, error state.
 *
 * @param runId Optional run filter.
 */
export function useDecisionTraces(runId?: string) {
  const config = getConfig();
  const [traces, setTraces] = useState<DecisionTraceEntry[]>([]);
  const [error, setError] = useState<string | null>(null);

  const fetchTraces = useCallback(async () => {
    try {
      const params = new URLSearchParams({ limit: String(config.historyLimit) });
      if (runId) params.set("runId", runId);
      const res = await fetch(
        `${config.apiBaseUrl}/api/decision-traces/history?${params.toString()}`,
      );
      if (!res.ok) {
        const msg = `Decision traces fetch failed: HTTP ${res.status}`;
        log.warn(msg);
        setError(msg);
        return;
      }
      const data = (await res.json()) as TraceHistoryRecord[];
      setTraces(data.map(toEntry));
      setError(null);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Failed to fetch decision traces";
      log.debug("Decision traces fetch error:", msg);
      setError(msg);
    }
  }, [config.apiBaseUrl, config.historyLimit, runId]);

  useEffect(() => {
    void fetchTraces();
    const interval = setInterval(() => void fetchTraces(), config.metricsPollingInterval);
    return () => clearInterval(interval);
  }, [fetchTraces, config.metricsPollingInterval]);

  return { traces, error };
}
