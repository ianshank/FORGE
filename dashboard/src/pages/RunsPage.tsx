import { History } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { EmptyState } from "../components/ui/empty-state";
import { useRuns } from "../hooks/useRuns";
import { formatNumber } from "../lib/utils";

/** Format an epoch-millis timestamp for display, tolerating 0/missing. */
function formatTimestamp(ms: number): string {
  if (!ms) return "—";
  return new Date(ms).toLocaleString();
}

/** Runs view: history of training / evaluation runs from `GET /api/runs`. */
export function RunsPage() {
  const { runs } = useRuns();

  return (
    <Card>
      <CardHeader>
        <CardTitle>Runs</CardTitle>
      </CardHeader>
      <CardContent>
        {runs.length === 0 ? (
          <EmptyState
            icon={History}
            title="No runs recorded"
            description="Training and evaluation runs will be listed here with their reward and sample count once a training job reports metrics."
          />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="text-muted-foreground">
                <tr className="border-b">
                  <th className="py-2 pr-4 font-medium">Run ID</th>
                  <th className="py-2 pr-4 font-medium">Started</th>
                  <th className="py-2 pr-4 font-medium">Last Seen</th>
                  <th className="py-2 pr-4 font-medium">Episodes</th>
                  <th className="py-2 pr-4 font-medium">Latest Mean Reward</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => (
                  <tr key={run.runId} className="border-b last:border-0">
                    <td className="py-2 pr-4 font-mono">{run.runId}</td>
                    <td className="py-2 pr-4">{formatTimestamp(run.startedAtMs)}</td>
                    <td className="py-2 pr-4">{formatTimestamp(run.lastSeenMs)}</td>
                    <td className="py-2 pr-4">{formatNumber(run.episodes)}</td>
                    <td className="py-2 pr-4">{formatNumber(run.latestMeanReward, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
