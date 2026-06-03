import { Dice5, Shuffle } from "lucide-react";
import { useCallback, useState } from "react";
import { getConfig } from "../config/environment";
import { createLogger } from "../utils/logger";
import { Button } from "./ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Input } from "./ui/input";
import { Slider } from "./ui/slider";

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
    <Card>
      <CardHeader>
        <CardTitle>Scenario</CardTitle>
        {error ? (
          <span className="text-xs text-danger" role="alert">
            {error}
          </span>
        ) : null}
      </CardHeader>
      <CardContent className="flex flex-wrap items-end gap-5">
        <div className="space-y-1.5">
          <label
            htmlFor="scenario-seed"
            className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground"
          >
            <Dice5 className="size-3.5" /> Seed
          </label>
          <Input
            id="scenario-seed"
            type="number"
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value))}
            aria-label="Random seed"
            className="w-28 font-mono"
          />
        </div>

        <div className="min-w-48 flex-1 space-y-1.5">
          <div className="flex items-center justify-between text-xs font-medium text-muted-foreground">
            <span>Grid size</span>
            <span className="font-mono text-foreground">{gridSize}</span>
          </div>
          <Slider
            min={16}
            max={256}
            step={16}
            value={[gridSize]}
            onValueChange={([v]) => setGridSize(v)}
            aria-label="Grid size"
          />
        </div>

        <Button
          type="button"
          onClick={() => void handleRemix()}
          disabled={loading}
        >
          <Shuffle />
          {loading ? "Remixing…" : "Remix"}
        </Button>
      </CardContent>
    </Card>
  );
}
