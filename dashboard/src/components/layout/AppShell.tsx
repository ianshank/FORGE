import type { ReactNode } from "react";
import { Sidebar } from "./Sidebar";
import { TopBar } from "./TopBar";

interface AppShellProps {
  /** Page title for the top bar. */
  title: string;
  /** Optional subtitle for the top bar. */
  subtitle?: string;
  /** Page content. */
  children: ReactNode;
}

/** Application chrome: sidebar + top bar + scrollable content region. */
export function AppShell({ title, subtitle, children }: AppShellProps) {
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar title={title} subtitle={subtitle} />
        <main className="flex-1 animate-fade-in overflow-y-auto p-6">
          {children}
        </main>
      </div>
    </div>
  );
}
