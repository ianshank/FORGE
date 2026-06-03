import type { LucideIcon } from "lucide-react";
import { cn } from "../../lib/utils";

interface StatCardProps {
  /** Short metric label. */
  label: string;
  /** Formatted metric value. */
  value: string;
  /** Optional unit / suffix shown after the value. */
  unit?: string;
  /** Optional leading icon. */
  icon?: LucideIcon;
  /** Accent tint for the icon + value. */
  tone?: "default" | "primary" | "success" | "warning" | "danger";
  className?: string;
}

const TONE_TEXT: Record<NonNullable<StatCardProps["tone"]>, string> = {
  default: "text-foreground",
  primary: "text-primary",
  success: "text-success",
  warning: "text-warning",
  danger: "text-danger",
};

/** Compact KPI tile with label, value and optional icon. */
export function StatCard({
  label,
  value,
  unit,
  icon: Icon,
  tone = "default",
  className,
}: StatCardProps) {
  return (
    <div
      className={cn(
        "rounded-lg border border-border bg-card px-4 py-3",
        className,
      )}
    >
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
          {label}
        </span>
        {Icon ? (
          <Icon className={cn("size-4", TONE_TEXT[tone])} />
        ) : null}
      </div>
      <div className="mt-1.5 flex items-baseline gap-1">
        <span
          className={cn(
            "font-mono text-2xl font-semibold tabular-nums",
            TONE_TEXT[tone],
          )}
        >
          {value}
        </span>
        {unit ? (
          <span className="text-xs text-muted-foreground">{unit}</span>
        ) : null}
      </div>
    </div>
  );
}
