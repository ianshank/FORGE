import { test, expect, DashboardPage } from "../fixtures/test-fixtures";

test.describe("Dashboard — Smoke Tests", () => {
  let dashboard: DashboardPage;

  test.beforeEach(async ({ page }) => {
    dashboard = new DashboardPage(page);
    await dashboard.goto();
  });

  test("page loads with correct title", async ({ page }) => {
    await expect(page).toHaveTitle(/FORGE/i);
  });

  test("header renders with FORGE Dashboard text", async () => {
    await expect(dashboard.header).toBeVisible();
    await expect(dashboard.header).toHaveText("FORGE Dashboard");
  });

  test("connection status indicator is visible", async () => {
    await expect(dashboard.connectionDot).toBeVisible();
    // Without the Rust backend, it should show disconnected (red dot)
    await expect(dashboard.connectionDot).toHaveClass(/bg-red-500|bg-yellow-500/);
  });

  test("connection label shows status text", async () => {
    await expect(dashboard.connectionLabel).toBeVisible();
    // Without backend it will show "disconnected" or "connecting"
    await expect(dashboard.connectionLabel).toHaveText(
      /disconnected|connecting/i,
    );
  });

  test("main layout sections are present", async ({ page }) => {
    // Header
    await expect(page.locator("header")).toBeVisible();
    // Controls bar
    await expect(dashboard.scenarioControls).toBeVisible();
    // Canvas area
    await expect(dashboard.simulationCanvas).toBeVisible();
    // Decision traces panel
    await expect(dashboard.decisionTracePanel).toBeVisible();
    // Metrics area
    await expect(dashboard.metricsDashboard).toBeVisible();
    // Agent inspector
    await expect(dashboard.agentInspector).toBeVisible();
  });

  test("page has no console errors on load", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (err) => errors.push(err.message));

    await page.goto("/");
    await page.waitForLoadState("networkidle");

    // Filter out expected WebSocket connection errors (no backend)
    const unexpected = errors.filter(
      (e) => !e.includes("WebSocket") && !e.includes("ECONNREFUSED"),
    );
    expect(unexpected).toHaveLength(0);
  });
});
