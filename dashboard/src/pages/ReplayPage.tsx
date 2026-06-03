import { Pause, Play, PlayCircle, SkipBack, SkipForward, Upload } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Area,
  AreaChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { EmptyState } from "../components/ui/empty-state";
import { Slider } from "../components/ui/slider";
import { StatCard } from "../components/ui/stat-card";
import {
  cumulativeReward,
  parseTrajectory,
  type Trajectory,
} from "../lib/trajectory";
import { createLogger } from "../utils/logger";

const log = createLogger("ReplayPage");

/** Playback speeds in steps-per-second. */
const SPEEDS = [1, 2, 5, 10, 30] as const;

/**
 * Episode replay: load a `forge-replay` v2 trajectory JSON and scrub
 * through its transitions with a reward timeline and per-step detail.
 */
export function ReplayPage() {
  const [trajectory, setTrajectory] = useState<Trajectory | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cursor, setCursor] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<(typeof SPEEDS)[number]>(5);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stepCount = trajectory?.steps.length ?? 0;

  const handleFile = useCallback(async (file: File) => {
    setError(null);
    setPlaying(false);
    try {
      const text = await file.text();
      const result = parseTrajectory(JSON.parse(text));
      if (!result.ok) {
        setError(result.error);
        return;
      }
      if (result.trajectory.steps.length === 0) {
        setError("Trajectory contains no steps.");
        return;
      }
      setTrajectory(result.trajectory);
      setCursor(0);
      log.info("Loaded trajectory with %d steps", result.trajectory.steps.length);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Failed to read file";
      setError(`Could not parse JSON: ${msg}`);
    }
  }, []);

  // Drive playback.
  useEffect(() => {
    if (!playing || stepCount === 0) return;
    timerRef.current = setInterval(() => {
      setCursor((c) => {
        if (c >= stepCount - 1) {
          setPlaying(false);
          return c;
        }
        return c + 1;
      });
    }, 1000 / speed);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [playing, speed, stepCount]);

  const rewardSeries = useMemo(() => {
    if (!trajectory) return [];
    const cumulative = cumulativeReward(trajectory.steps);
    return trajectory.steps.map((s, i) => ({
      index: i,
      tick: s.tick,
      reward: s.reward,
      cumulative: cumulative[i],
    }));
  }, [trajectory]);

  if (!trajectory) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Episode Replay</CardTitle>
        </CardHeader>
        <CardContent>
          <EmptyState
            icon={PlayCircle}
            title="Load a trajectory to replay"
            description="Open a forge-replay v2 trajectory JSON (as written by forge-mc-runner) to scrub through its transitions."
            action={<FilePicker onFile={handleFile} />}
          />
          {error ? (
            <p className="mt-3 text-center text-xs text-danger" role="alert">
              {error}
            </p>
          ) : null}
        </CardContent>
      </Card>
    );
  }

  const current = trajectory.steps[cursor];

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>
            Episode{" "}
            <span className="font-mono text-muted-foreground">
              {trajectory.episodeId}
            </span>
          </CardTitle>
          <div className="flex items-center gap-2">
            <Badge variant="primary">{trajectory.envId}</Badge>
            <FilePicker onFile={handleFile} label="Load another" />
          </div>
        </CardHeader>
        <CardContent className="space-y-5">
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <StatCard label="Steps" value={`${stepCount}`} />
            <StatCard
              label="Final Reward"
              value={trajectory.finalReward.toFixed(2)}
              tone="success"
            />
            <StatCard label="Obs Dim" value={`${trajectory.obsDim}`} />
            <StatCard label="Actions" value={`${trajectory.actionCount}`} />
          </div>

          <ResponsiveContainer width="100%" height={180}>
            <AreaChart data={rewardSeries} margin={{ top: 4, right: 8, bottom: 0, left: -12 }}>
              <defs>
                <linearGradient id="replay-grad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#39ff7e" stopOpacity={0.35} />
                  <stop offset="100%" stopColor="#39ff7e" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis
                dataKey="index"
                tick={{ fontSize: 10, fill: "hsl(215 18% 58%)" }}
                stroke="hsl(217 33% 18%)"
              />
              <YAxis
                tick={{ fontSize: 10, fill: "hsl(215 18% 58%)" }}
                stroke="hsl(217 33% 18%)"
                width={44}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(222 44% 6%)",
                  border: "1px solid hsl(217 33% 15%)",
                  borderRadius: 8,
                  fontSize: 12,
                }}
                labelStyle={{ color: "hsl(215 18% 58%)" }}
              />
              <Area
                type="monotone"
                dataKey="cumulative"
                stroke="#39ff7e"
                strokeWidth={2}
                fill="url(#replay-grad)"
                dot={false}
              />
              <ReferenceLine x={cursor} stroke="#38bdf8" strokeWidth={1.5} />
            </AreaChart>
          </ResponsiveContainer>

          {/* Transport controls. */}
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-1">
              <Button
                variant="ghost"
                size="icon"
                aria-label="Step back"
                onClick={() => setCursor((c) => Math.max(0, c - 1))}
              >
                <SkipBack />
              </Button>
              <Button
                variant="secondary"
                size="icon"
                aria-label={playing ? "Pause" : "Play"}
                onClick={() => setPlaying((p) => !p)}
              >
                {playing ? <Pause /> : <Play />}
              </Button>
              <Button
                variant="ghost"
                size="icon"
                aria-label="Step forward"
                onClick={() => setCursor((c) => Math.min(stepCount - 1, c + 1))}
              >
                <SkipForward />
              </Button>
            </div>

            <Slider
              className="flex-1"
              min={0}
              max={stepCount - 1}
              step={1}
              value={[cursor]}
              onValueChange={([v]) => {
                setPlaying(false);
                setCursor(v);
              }}
              aria-label="Timeline"
            />

            <span className="w-24 shrink-0 text-right font-mono text-xs text-muted-foreground">
              {cursor + 1} / {stepCount}
            </span>

            <div className="flex items-center gap-1">
              {SPEEDS.map((s) => (
                <Button
                  key={s}
                  size="sm"
                  variant={s === speed ? "default" : "ghost"}
                  onClick={() => setSpeed(s)}
                >
                  {s}×
                </Button>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Step Detail</CardTitle>
          <Badge variant="outline" className="font-mono">
            tick {current.tick}
          </Badge>
        </CardHeader>
        <CardContent>
          <dl className="grid grid-cols-2 gap-x-8 gap-y-2 font-mono text-xs md:grid-cols-3">
            <Detail label="Action" value={`#${current.actionId}`} />
            <Detail label="Reward" value={current.reward.toFixed(4)} />
            <Detail label="Value Target" value={current.valueTarget.toFixed(4)} />
            <Detail label="Policy Dim" value={`${current.policyTargetLen}`} />
            <Detail
              label="Terminated"
              value={current.terminated ? "yes" : "no"}
            />
            <Detail label="Truncated" value={current.truncated ? "yes" : "no"} />
          </dl>
        </CardContent>
      </Card>
    </div>
  );
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between border-b border-border/60 py-1.5">
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="text-foreground">{value}</dd>
    </div>
  );
}

function FilePicker({
  onFile,
  label = "Load trajectory",
}: {
  onFile: (file: File) => void;
  label?: string;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  return (
    <>
      <input
        ref={inputRef}
        type="file"
        accept="application/json,.json"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onFile(file);
          e.target.value = "";
        }}
      />
      <Button
        variant="secondary"
        size="sm"
        onClick={() => inputRef.current?.click()}
      >
        <Upload /> {label}
      </Button>
    </>
  );
}
