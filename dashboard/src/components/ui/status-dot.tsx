import type { ConnectionStatus } from "../../hooks/useWebSocket";
import { cn } from "../../lib/utils";

interface StatusDotProps {
  status: ConnectionStatus;
  className?: string;
}

const STATUS_STYLES: Record<ConnectionStatus, string> = {
  connected: "bg-success",
  connecting: "bg-warning animate-pulse-dot",
  disconnected: "bg-danger",
};

const STATUS_LABEL: Record<ConnectionStatus, string> = {
  connected: "Connected",
  connecting: "Connecting",
  disconnected: "Disconnected",
};

/** Colored connection-status indicator dot. */
export function StatusDot({ status, className }: StatusDotProps) {
  return (
    <span
      className={cn("inline-block size-2 rounded-full", STATUS_STYLES[status], className)}
      role="img"
      aria-label={STATUS_LABEL[status]}
      title={STATUS_LABEL[status]}
    />
  );
}

export { STATUS_LABEL };
