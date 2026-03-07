import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { TrainingMetrics } from "../types/simulation";

interface MetricsDashboardProps {
  history: TrainingMetrics[];
}

/** Dashboard showing training metrics over time using Recharts. */
export function MetricsDashboard({ history }: MetricsDashboardProps) {
  if (history.length === 0) {
    return (
      <div className="bg-gray-900 border border-gray-700 rounded p-4">
        <p className="text-gray-500 text-sm italic">
          No training metrics available
        </p>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 border border-gray-700 rounded p-4 grid grid-cols-2 gap-4">
      <MetricChart
        title="Reward"
        data={history}
        dataKey="reward"
        color="#22c55e"
      />
      <MetricChart
        title="Win Rate"
        data={history}
        dataKey="winRate"
        color="#3b82f6"
      />
      <MetricChart
        title="Steps/Second"
        data={history}
        dataKey="stepsPerSecond"
        color="#f59e0b"
      />
      <MetricChart
        title="Entropy"
        data={history}
        dataKey="entropy"
        color="#a855f7"
      />
    </div>
  );
}

interface MetricChartProps {
  title: string;
  data: TrainingMetrics[];
  dataKey: keyof TrainingMetrics;
  color: string;
}

function MetricChart({ title, data, dataKey, color }: MetricChartProps) {
  return (
    <div>
      <h4 className="text-xs font-bold text-gray-400 mb-1">{title}</h4>
      <ResponsiveContainer width="100%" height={120}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#333" />
          <XAxis dataKey="episode" tick={{ fontSize: 10 }} stroke="#666" />
          <YAxis tick={{ fontSize: 10 }} stroke="#666" />
          <Tooltip
            contentStyle={{ backgroundColor: "#1a1a2e", border: "1px solid #333" }}
          />
          <Line
            type="monotone"
            dataKey={dataKey}
            stroke={color}
            dot={false}
            strokeWidth={1.5}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
