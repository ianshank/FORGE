import { MetricsDashboard } from "../components/MetricsDashboard";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { EmptyState } from "../components/ui/empty-state";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs";
import { useTrainingHistory } from "../hooks/useTrainingHistory";
import { GitCompare } from "lucide-react";

/**
 * Training view: metric curves over episodes plus a placeholder for
 * multi-run comparison (wired once the runs API lands).
 */
export function TrainingPage() {
  // Training metrics are persisted server-side by training jobs and polled via
  // the history endpoint; MetricsDashboard renders an empty state when none.
  const { history } = useTrainingHistory();

  return (
    <Tabs defaultValue="curves" className="space-y-2">
      <TabsList>
        <TabsTrigger value="curves">Curves</TabsTrigger>
        <TabsTrigger value="compare">Compare runs</TabsTrigger>
      </TabsList>

      <TabsContent value="curves">
        <MetricsDashboard history={history} />
      </TabsContent>

      <TabsContent value="compare">
        <Card>
          <CardHeader>
            <CardTitle>Run Comparison</CardTitle>
          </CardHeader>
          <CardContent>
            <EmptyState
              icon={GitCompare}
              title="Select runs to compare"
              description="Once two or more completed runs are available, overlay their reward and loss curves here."
            />
          </CardContent>
        </Card>
      </TabsContent>
    </Tabs>
  );
}
