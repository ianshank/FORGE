import { test, expect, DashboardPage } from "../fixtures/test-fixtures";

test.describe("Dashboard — Component Rendering", () => {
  let dashboard: DashboardPage;

  test.beforeEach(async ({ page }) => {
    dashboard = new DashboardPage(page);
    await dashboard.goto();
  });

  test("SimulationCanvas renders a canvas element", async () => {
    await expect(dashboard.simulationCanvas).toBeVisible();
    // Canvas should have pixelated rendering style
    await expect(dashboard.simulationCanvas).toHaveCSS(
      "image-rendering",
      /pixelated|crisp-edges/,
    );
  });

  test("ScenarioControls renders seed input, grid slider, and remix button", async () => {
    await expect(dashboard.scenarioControls).toBeVisible();

    // Seed input
    await expect(dashboard.seedInput).toBeVisible();
    await expect(dashboard.seedInput).toHaveValue("42");

    // Grid size slider
    await expect(dashboard.gridSlider).toBeVisible();

    // Remix button
    await expect(dashboard.remixButton).toBeVisible();
    await expect(dashboard.remixButton).toHaveText(/Remix/);
  });

  test("ScenarioControls seed input accepts new values", async () => {
    await dashboard.seedInput.fill("123");
    await expect(dashboard.seedInput).toHaveValue("123");
  });

  test("DecisionTracePanel renders with header and empty state", async ({
    page,
  }) => {
    await expect(dashboard.decisionTracePanel).toBeVisible();
    await expect(dashboard.decisionTracePanel).toContainText(
      "Decision Traces",
    );

    // Empty state message
    const emptyMsg = page.locator("text=No traces yet");
    await expect(emptyMsg).toBeVisible();
  });

  test("MetricsDashboard renders empty state", async () => {
    await expect(dashboard.metricsDashboard).toBeVisible();
    await expect(dashboard.metricsDashboard).toContainText(
      "No training metrics available",
    );
  });

  test("AgentInspector renders empty state", async () => {
    await expect(dashboard.agentInspector).toBeVisible();
    await expect(dashboard.agentInspector).toContainText(
      "Click an agent to inspect",
    );
  });
});
