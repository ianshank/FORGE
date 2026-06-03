import { LineChartIcon } from "lucide-react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { TrainingMetrics } from "../types/simulation";
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
  { title: "Reward", dataKey: "reward", color: "#39ff7e" },
  { title: "Win Rate", dataKey: "winRate", color: "#38bdf8" },
  { title: "Steps / Second", dataKey: "stepsPerSecond", color: "#ffb347" },
  { title: "Policy Entropy", dataKey: "entropy", color: "#a855f7" },
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
  const gradientId = `grad-${spec.dataKey}`;
  return (
    <Card>
      <CardHeader>
        <CardTitle>{spec.title}</CardTitle>
        <span className="font-mono text-sm tabular-nums" style={{ color: spec.color }}>
          {typeof latest === "number" ? latest.toFixed(2) : "—"}
        </span>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={140}>
          <AreaChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -16 }}>
            <defs>
              <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={spec.color} stopOpacity={0.35} />
                <stop offset="100%" stopColor={spec.color} stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(217 33% 15%)" />
            <XAxis
              dataKey="episode"
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
              dataKey={spec.dataKey}
              stroke={spec.color}
              strokeWidth={2}
              fill={`url(#${gradientId})`}
              dot={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
