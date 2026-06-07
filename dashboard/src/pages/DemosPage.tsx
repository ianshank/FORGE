import { Loader2, Play, Square, Terminal } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { EmptyState } from "../components/ui/empty-state";
import { getConfig } from "../config/environment";
import { isStreamEnd, parseSseBuffer } from "../lib/sse";
import { cn } from "../lib/utils";
import { createLogger } from "../utils/logger";

const log = createLogger("DemosPage");

/** Default parameters sent when launching a demo run. */
const DEMO_RUN_DEFAULTS = { seed: 42, quick: true } as const;

interface DemoSection {
  key: string;
  name: string;
}

/** The eight demo sections exposed by the demo_ui backend. */
const SECTIONS: DemoSection[] = [
  { key: "worldgen", name: "World Generation" },
  { key: "navigation", name: "Navigation" },
  { key: "gathering", name: "Resource Gathering" },
  { key: "crafting", name: "Crafting" },
  { key: "multiagent", name: "Multi-Agent" },
  { key: "daynight", name: "Day / Night Cycle" },
  { key: "determinism", name: "Determinism" },
  { key: "performance", name: "Performance" },
];

/**
 * Demos view: run FORGE engine demos through the demo_ui FastAPI backend
 * and stream their output live into a terminal-style panel (folds the
 * standalone demo_ui into the unified dashboard).
 */
export function DemosPage() {
  const config = getConfig();
  const [active, setActive] = useState<string | null>(null);
  const [lines, setLines] = useState<string[]>([]);
  const [running, setRunning] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const outputRef = useRef<HTMLDivElement>(null);

  // Auto-scroll the terminal as new lines arrive. `lines` drives the effect
  // even though only the ref is read inside.
  // biome-ignore lint/correctness/useExhaustiveDependencies: re-run on each new line batch.
  useEffect(() => {
    const el = outputRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [lines]);

  // Tidy up any in-flight stream on unmount.
  useEffect(() => () => abortRef.current?.abort(), []);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setRunning(false);
  }, []);

  const run = useCallback(
    async (section: DemoSection) => {
      stop();
      const controller = new AbortController();
      abortRef.current = controller;
      setActive(section.key);
      setLines([]);
      setRunning(true);

      try {
        const res = await fetch(`${config.demoApiBaseUrl}/api/run/${section.key}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(DEMO_RUN_DEFAULTS),
          signal: controller.signal,
        });

        if (!res.ok || !res.body) {
          setLines([`Error: demo backend responded ${res.status}.`]);
          setRunning(false);
          return;
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          const { events, rest } = parseSseBuffer(buffer);
          buffer = rest;
          for (const payload of events) {
            if (isStreamEnd(payload)) {
              setRunning(false);
              continue;
            }
            setLines((prev) => [...prev, payload]);
          }
        }
      } catch (e) {
        if (e instanceof Error && e.name === "AbortError") return;
        const msg = e instanceof Error ? e.message : "stream failed";
        log.warn("Demo stream error:", msg);
        setLines((prev) => [
          ...prev,
          `Error: ${msg}`,
          `Is the demo backend running at ${config.demoApiBaseUrl}?`,
        ]);
      } finally {
        setRunning(false);
      }
    },
    [config.demoApiBaseUrl, stop],
  );

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-[280px_1fr]">
      <Card className="h-fit">
        <CardHeader>
          <CardTitle>Demo Sections</CardTitle>
        </CardHeader>
        <CardContent className="space-y-1 p-2">
          {SECTIONS.map((section, i) => {
            const isActive = section.key === active;
            return (
              <button
                key={section.key}
                type="button"
                disabled={running}
                onClick={() => void run(section)}
                className={cn(
                  "flex w-full items-center gap-3 rounded-md px-3 py-2 text-left text-sm transition-colors disabled:opacity-60",
                  isActive
                    ? "bg-accent text-foreground"
                    : "text-muted-foreground hover:bg-accent/50 hover:text-foreground",
                )}
              >
                <span className="font-mono text-xs text-muted-foreground">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <span className="flex-1">{section.name}</span>
                {isActive && running ? (
                  <Loader2 className="size-4 animate-spin text-primary" />
                ) : (
                  <Play className="size-3.5 opacity-0 group-hover:opacity-100" />
                )}
              </button>
            );
          })}
        </CardContent>
      </Card>

      <Card className="flex min-h-[28rem] flex-col">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Terminal className="size-4" />
            {active
              ? SECTIONS.find((s) => s.key === active)?.name
              : "Output"}
          </CardTitle>
          <div className="flex items-center gap-2">
            {running ? (
              <>
                <Badge variant="primary">
                  <Loader2 className="size-3 animate-spin" /> running
                </Badge>
                <Button variant="ghost" size="sm" onClick={stop}>
                  <Square /> Stop
                </Button>
              </>
            ) : active ? (
              <Badge variant="success">done</Badge>
            ) : null}
          </div>
        </CardHeader>
        <CardContent className="flex-1 overflow-hidden p-0">
          {lines.length === 0 && !running ? (
            <EmptyState
              icon={Terminal}
              title="Select a demo to run"
              description="Output streams here live from the FORGE engine via the demo backend."
            />
          ) : (
            <div
              ref={outputRef}
              data-testid="demo-output"
              className="h-full overflow-y-auto bg-background/60 p-4 font-mono text-xs leading-relaxed text-foreground/90"
            >
              {lines.map((line, i) => (
                // biome-ignore lint/suspicious/noArrayIndexKey: append-only log; lines are never reordered.
                <div key={i} className="whitespace-pre-wrap">
                  {line || " "}
                </div>
              ))}
              {running ? (
                <span className="inline-block h-3.5 w-2 animate-pulse-dot bg-primary align-middle" />
              ) : null}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
