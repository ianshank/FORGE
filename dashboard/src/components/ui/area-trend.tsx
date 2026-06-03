import { useId, useMemo } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { resolveChartTheme } from "../../lib/chartTheme";

export interface AreaTrendProps {
  /** Row data. Each row must contain `xKey` and `dataKey` numeric fields. */
  data: Array<Record<string, number>>;
  /** Field plotted on the Y axis. */
  dataKey: string;
  /** Field plotted on the X axis. */
  xKey: string;
  /** Series color (hex). */
  color: string;
  /** Chart height in px. */
  height?: number;
  /** Optional X position for a vertical reference line (e.g. replay cursor). */
  referenceX?: number;
  /** Whether to draw the cartesian grid. */
  showGrid?: boolean;
}

/**
 * Reusable themed Recharts area chart.
 *
 * Centralises the gradient + axis + tooltip styling that was previously
 * duplicated across the metrics and replay views. Colors come from the shared
 * {@link resolveChartTheme} token resolver rather than inline literals.
 */
export function AreaTrend({
  data,
  dataKey,
  xKey,
  color,
  height = 140,
  referenceX,
  showGrid = true,
}: AreaTrendProps) {
  const theme = useMemo(() => resolveChartTheme(), []);
  // Stable, collision-free gradient id (multiple charts can share a page).
  const gradientId = `area-grad-${useId().replace(/:/g, "")}`;

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -16 }}>
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.35} />
            <stop offset="100%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>
        {showGrid ? (
          <CartesianGrid strokeDasharray="3 3" stroke={theme.grid} />
        ) : null}
        <XAxis
          dataKey={xKey}
          tick={{ fontSize: 10, fill: theme.axisText }}
          stroke={theme.axis}
        />
        <YAxis
          tick={{ fontSize: 10, fill: theme.axisText }}
          stroke={theme.axis}
          width={44}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: theme.tooltipBg,
            border: `1px solid ${theme.tooltipBorder}`,
            borderRadius: 8,
            fontSize: 12,
          }}
          labelStyle={{ color: theme.tooltipText }}
        />
        <Area
          type="monotone"
          dataKey={dataKey}
          stroke={color}
          strokeWidth={2}
          fill={`url(#${gradientId})`}
          dot={false}
        />
        {typeof referenceX === "number" ? (
          <ReferenceLine x={referenceX} stroke="#38bdf8" strokeWidth={1.5} />
        ) : null}
      </AreaChart>
    </ResponsiveContainer>
  );
}
