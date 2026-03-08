import { useCallback, useState } from "react";
import { getConfig } from "../config/environment";
import { createLogger } from "../utils/logger";

const log = createLogger("ScenarioControls");

interface ScenarioControlsProps {
  onRemix?: (seed: number) => void;
}

/** Controls for scenario remixing and parameter adjustment. */
export function ScenarioControls({ onRemix }: ScenarioControlsProps) {
  const config = getConfig();
  const [seed, setSeed] = useState(42);
  const [gridSize, setGridSize] = useState(64);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleRemix = useCallback(async () => {
    setLoading(true);
    setError(null);
    log.info("Remixing scenario with seed=%d, gridSize=%d", seed, gridSize);

    try {
      const res = await fetch(`${config.apiBaseUrl}/api/scenario/remix`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ seed, gridSize }),
      });

      if (!res.ok) {
        const msg = `Remix failed: HTTP ${res.status}`;
        log.error(msg);
        setError(msg);
        return;
      }

      log.info("Remix successful");
      onRemix?.(seed);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Server unreachable";
      log.warn("Remix request failed:", msg);
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, [config.apiBaseUrl, seed, gridSize, onRemix]);

  return (
    <div className="bg-gray-900 border border-gray-700 rounded p-3 flex items-center gap-4">
      <label className="text-sm text-gray-400 flex items-center gap-2">
        Seed:
        <input
          type="number"
          value={seed}
          onChange={(e) => setSeed(Number(e.target.value))}
          aria-label="Random seed"
          className="w-20 bg-gray-800 border border-gray-600 rounded px-2 py-1 text-white text-sm"
        />
      </label>
      <label className="text-sm text-gray-400 flex items-center gap-2">
        Grid:
        <input
          type="range"
          min={16}
          max={256}
          step={16}
          value={gridSize}
          onChange={(e) => setGridSize(Number(e.target.value))}
          aria-label="Grid size"
          className="w-24"
        />
        <span className="text-white text-sm w-8">{gridSize}</span>
      </label>
      <button
        onClick={() => void handleRemix()}
        disabled={loading}
        className="bg-blue-600 hover:bg-blue-500 disabled:bg-gray-600 text-white text-sm px-4 py-1 rounded transition-colors"
      >
        {loading ? "Remixing..." : "Remix"}
      </button>
      {error && (
        <span className="text-red-400 text-xs" role="alert">
          {error}
        </span>
      )}
    </div>
  );
}
