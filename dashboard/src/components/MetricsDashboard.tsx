import { LineChartIcon } from "lucide-react";
import { CHART_SERIES } from "../lib/chartTheme";
import type { TrainingMetrics } from "../types/simulation";
import { AreaTrend } from "./ui/area-trend";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { EmptyState } from "./ui/empty-state";

interface MetricsDashboardProps {
  history: TrainingMetrics[];
}

interface MetricSpec {
  title: string;
  dataKey: keyof TrainingMetrics;
  color: string;
}

const METRICS: MetricSpec[] = [
  { title: "Reward", dataKey: "reward", color: CHART_SERIES.reward },
  { title: "Win Rate", dataKey: "winRate", color: CHART_SERIES.winRate },
  {
    title: "Steps / Second",
    dataKey: "stepsPerSecond",
    color: CHART_SERIES.stepsPerSecond,
  },
  { title: "Policy Entropy", dataKey: "entropy", color: CHART_SERIES.entropy },
];

/** Grid of training-metric charts over episodes. */
export function MetricsDashboard({ history }: MetricsDashboardProps) {
  if (history.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Training Metrics</CardTitle>
        </CardHeader>
        <CardContent>
          <EmptyState
            icon={LineChartIcon}
            title="No training metrics yet"
            description="Metrics appear here once a training run posts to /api/training-metrics."
          />
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
      {METRICS.map((spec) => (
        <MetricChart key={spec.dataKey} data={history} spec={spec} />
      ))}
    </div>
  );
}

function MetricChart({
  data,
  spec,
}: {
  data: TrainingMetrics[];
  spec: MetricSpec;
}) {
  const latest = data[data.length - 1]?.[spec.dataKey];
  return (
    <Card>
      <CardHeader>
        <CardTitle>{spec.title}</CardTitle>
        <span
          className="font-mono text-sm tabular-nums"
          style={{ color: spec.color }}
        >
          {typeof latest === "number" ? latest.toFixed(2) : "—"}
        </span>
      </CardHeader>
      <CardContent>
        <AreaTrend
          data={data as unknown as Array<Record<string, number>>}
          xKey="episode"
          dataKey={spec.dataKey}
          color={spec.color}
        />
      </CardContent>
    </Card>
  );
}
