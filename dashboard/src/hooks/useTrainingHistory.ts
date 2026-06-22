import { useCallback, useEffect, useState } from "react";
import { getConfig } from "../config/environment";
import type { TrainingHistoryRecord, TrainingMetrics } from "../types/simulation";
import { createLogger } from "../utils/logger";

const log = createLogger("useTrainingHistory");

/**
 * Map a persisted server record onto the chart-facing `TrainingMetrics` shape
 * (the server stores `meanReward`; the dashboard charts read `reward`).
 */
function toMetrics(rec: TrainingHistoryRecord): TrainingMetrics {
  return {
    episode: rec.episode,
    reward: rec.meanReward,
    winRate: rec.winRate,
    stepsPerSecond: rec.stepsPerSecond,
    lossPolicy: rec.lossPolicy,
    lossValue: rec.lossValue,
    entropy: rec.entropy,
  };
}

/**
 * Poll `GET /api/training-metrics/history` and expose the chart-ready history.
 * Mirrors {@link useMetrics}: config-driven interval, REST polling, error state.
 *
 * @param runId Optional run filter; when set, only that run's samples load.
 */
export function useTrainingHistory(runId?: string) {
  const config = getConfig();
  const [history, setHistory] = useState<TrainingMetrics[]>([]);
  const [error, setError] = useState<string | null>(null);

  const fetchHistory = useCallback(async () => {
    try {
      const params = new URLSearchParams({ limit: String(config.historyLimit) });
      if (runId) params.set("runId", runId);
      const res = await fetch(
        `${config.apiBaseUrl}/api/training-metrics/history?${params.toString()}`,
      );
      if (!res.ok) {
        const msg = `Training history fetch failed: HTTP ${res.status}`;
        log.warn(msg);
        setError(msg);
        return;
      }
      const data = (await res.json()) as TrainingHistoryRecord[];
      setHistory(data.map(toMetrics));
      setError(null);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Failed to fetch training history";
      log.debug("Training history fetch error:", msg);
      setError(msg);
    }
  }, [config.apiBaseUrl, config.historyLimit, runId]);

  useEffect(() => {
    void fetchHistory();
    const interval = setInterval(() => void fetchHistory(), config.trainingHistoryInterval);
    return () => clearInterval(interval);
  }, [fetchHistory, config.trainingHistoryInterval]);

  return { history, error };
}
