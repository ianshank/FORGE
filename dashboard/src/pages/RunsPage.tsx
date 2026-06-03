import { History } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { EmptyState } from "../components/ui/empty-state";

/** Runs view: history of training / evaluation runs (placeholder). */
export function RunsPage() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Runs</CardTitle>
      </CardHeader>
      <CardContent>
        <EmptyState
          icon={History}
          title="No runs recorded"
          description="Training and evaluation runs will be listed here with their status, reward, and model version once the runs API is connected."
        />
      </CardContent>
    </Card>
  );
}
