import { lazy, Suspense, type ReactNode } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/layout/AppShell";
import { Skeleton } from "./components/ui/skeleton";
import { SimulationProvider } from "./context/SimulationContext";

// Route-level code splitting keeps heavy deps (Recharts) off the initial bundle.
const LivePage = lazy(() =>
  import("./pages/LivePage").then((m) => ({ default: m.LivePage })),
);
const TrainingPage = lazy(() =>
  import("./pages/TrainingPage").then((m) => ({ default: m.TrainingPage })),
);
const RunsPage = lazy(() =>
  import("./pages/RunsPage").then((m) => ({ default: m.RunsPage })),
);
const ReplayPage = lazy(() =>
  import("./pages/ReplayPage").then((m) => ({ default: m.ReplayPage })),
);
const DemosPage = lazy(() =>
  import("./pages/DemosPage").then((m) => ({ default: m.DemosPage })),
);
const SettingsPage = lazy(() =>
  import("./pages/SettingsPage").then((m) => ({ default: m.SettingsPage })),
);

interface PageMeta {
  title: string;
  subtitle: string;
}

/** Per-route top-bar metadata. */
const PAGE_META = {
  live: { title: "Live", subtitle: "Real-time simulation state and agents" },
  training: { title: "Training", subtitle: "Metric curves and run comparison" },
  runs: { title: "Runs", subtitle: "Training and evaluation history" },
  replay: { title: "Replay", subtitle: "Scrub recorded episode trajectories" },
  demos: { title: "Demos", subtitle: "Run FORGE engine demos live" },
  settings: { title: "Settings", subtitle: "Dashboard configuration" },
} satisfies Record<string, PageMeta>;

/** Loading fallback shown while a route chunk is fetched. */
function PageFallback() {
  return (
    <div className="space-y-4">
      <Skeleton className="h-24 w-full" />
      <Skeleton className="h-64 w-full" />
    </div>
  );
}

/** Wrap a page in the app shell with its title metadata. */
function Page({
  id,
  children,
}: {
  id: keyof typeof PAGE_META;
  children: ReactNode;
}) {
  const meta = PAGE_META[id];
  return (
    <AppShell title={meta.title} subtitle={meta.subtitle}>
      <Suspense fallback={<PageFallback />}>{children}</Suspense>
    </AppShell>
  );
}

/** Root application component — FORGE Control Center. */
export function App() {
  return (
    <SimulationProvider>
      <Routes>
        <Route path="/" element={<Navigate to="/live" replace />} />
        <Route path="/live" element={<Page id="live"><LivePage /></Page>} />
        <Route
          path="/training"
          element={<Page id="training"><TrainingPage /></Page>}
        />
        <Route path="/runs" element={<Page id="runs"><RunsPage /></Page>} />
        <Route
          path="/replay"
          element={<Page id="replay"><ReplayPage /></Page>}
        />
        <Route path="/demos" element={<Page id="demos"><DemosPage /></Page>} />
        <Route
          path="/settings"
          element={<Page id="settings"><SettingsPage /></Page>}
        />
        <Route path="*" element={<Navigate to="/live" replace />} />
      </Routes>
    </SimulationProvider>
  );
}
