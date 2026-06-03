import { useSimulation } from "../../context/SimulationContext";
import { StatusDot, STATUS_LABEL } from "../ui/status-dot";

interface TopBarProps {
  /** Page title shown on the left. */
  title: string;
  /** Optional short description under the title. */
  subtitle?: string;
}

/** Sticky page header with title and live connection status. */
export function TopBar({ title, subtitle }: TopBarProps) {
  const { state, connectionStatus } = useSimulation();

  return (
    <header className="flex h-14 shrink-0 items-center justify-between border-b border-border bg-background/80 px-6 backdrop-blur">
      <div>
        <h1 className="text-base font-semibold tracking-tight">{title}</h1>
        {subtitle ? (
          <p className="text-xs text-muted-foreground">{subtitle}</p>
        ) : null}
      </div>

      <div className="flex items-center gap-4 text-xs">
        {connectionStatus === "connected" ? (
          <span className="font-mono text-muted-foreground">
            tick{" "}
            <span className="text-foreground">{state?.tick ?? 0}</span>
            <span className="mx-1.5 text-border">·</span>
            <span className="text-foreground">{state?.agents.length ?? 0}</span>{" "}
            agents
          </span>
        ) : null}
        <div className="flex items-center gap-2 rounded-full border border-border bg-card px-2.5 py-1">
          <StatusDot status={connectionStatus} />
          <span className="text-muted-foreground">
            {STATUS_LABEL[connectionStatus]}
          </span>
        </div>
      </div>
    </header>
  );
}
