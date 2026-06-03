import { ListTree } from "lucide-react";
import type { DecisionTraceEntry } from "../types/simulation";
import { Badge } from "./ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { EmptyState } from "./ui/empty-state";

interface DecisionTracePanelProps {
  traces: DecisionTraceEntry[];
  maxEntries?: number;
}

/** Panel displaying live decision trace entries from agents. */
export function DecisionTracePanel({
  traces,
  maxEntries = 100,
}: DecisionTracePanelProps) {
  const displayed = traces.slice(-maxEntries).reverse();

  return (
    <Card className="flex h-full flex-col">
      <CardHeader>
        <CardTitle>Decision Traces</CardTitle>
        <Badge variant="outline" className="font-mono">
          {traces.length}
        </Badge>
      </CardHeader>
      <CardContent className="flex-1 overflow-y-auto p-0">
        {displayed.length === 0 ? (
          <EmptyState
            icon={ListTree}
            title="No traces yet"
            description="MCTS decision traces stream here as agents act."
          />
        ) : (
          <ul className="divide-y divide-border/60 font-mono text-xs">
            {displayed.map((trace, i) => (
              <li
                key={`${trace.tick}-${trace.agentId}-${i}`}
                className="flex items-center gap-3 px-4 py-1.5 transition-colors hover:bg-accent/40"
              >
                <span className="w-14 shrink-0 text-muted-foreground">
                  t{trace.tick}
                </span>
                <span className="w-8 shrink-0 text-primary">
                  A{trace.agentId}
                </span>
                <span className="flex-1 truncate text-warning">
                  {trace.intentLabel}
                </span>
                <span className="w-10 shrink-0 text-right text-success">
                  {(trace.confidence * 100).toFixed(0)}%
                </span>
                <span className="w-10 shrink-0 text-right text-muted-foreground">
                  d{trace.searchDepth}
                </span>
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}
