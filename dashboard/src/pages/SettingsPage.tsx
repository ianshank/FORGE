import { getConfig } from "../config/environment";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";

/** Settings view: surfaces the resolved runtime configuration. */
export function SettingsPage() {
  const config = getConfig();

  const rows: Array<{ label: string; value: string; env: string }> = [
    { label: "WebSocket URL", value: config.wsUrl, env: "VITE_WS_URL" },
    { label: "API base URL", value: config.apiBaseUrl, env: "VITE_API_BASE_URL" },
    {
      label: "Demo API base URL",
      value: config.demoApiBaseUrl,
      env: "VITE_DEMO_API_BASE_URL",
    },
    {
      label: "Metrics polling",
      value: `${config.metricsPollingInterval} ms`,
      env: "VITE_METRICS_INTERVAL",
    },
    {
      label: "Max trace entries",
      value: `${config.maxTraceEntries}`,
      env: "VITE_MAX_TRACES",
    },
    {
      label: "Show grid lines",
      value: config.showGridLines ? "true" : "false",
      env: "VITE_SHOW_GRID",
    },
    { label: "Cell size", value: `${config.cellSize}px`, env: "VITE_CELL_SIZE" },
    {
      label: "Max reconnect attempts",
      value: `${config.maxReconnectAttempts}`,
      env: "VITE_MAX_RECONNECT_ATTEMPTS",
    },
  ];

  return (
    <div className="max-w-3xl space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Runtime Configuration</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border text-left text-xs uppercase tracking-wide text-muted-foreground">
                <th className="px-4 py-2 font-medium">Setting</th>
                <th className="px-4 py-2 font-medium">Value</th>
                <th className="px-4 py-2 font-medium">Env Var</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr
                  key={row.label}
                  className="border-b border-border/60 last:border-0"
                >
                  <td className="px-4 py-2.5 text-muted-foreground">
                    {row.label}
                  </td>
                  <td className="px-4 py-2.5 font-mono text-foreground">
                    {row.value}
                  </td>
                  <td className="px-4 py-2.5 font-mono text-xs text-muted-foreground">
                    {row.env}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>

      <p className="text-xs text-muted-foreground">
        Values are resolved once at startup from <code className="font-mono">VITE_*</code>{" "}
        environment variables, with the defaults shown above. Restart the dev
        server after changing them.
      </p>
    </div>
  );
}
