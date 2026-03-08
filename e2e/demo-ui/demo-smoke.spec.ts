import {
  test,
  expect,
  DemoPage,
  DEMO_SECTIONS,
} from "../fixtures/test-fixtures";

test.describe("Demo UI — Smoke Tests", () => {
  let demo: DemoPage;

  test.beforeEach(async ({ page }) => {
    demo = new DemoPage(page);
    await demo.goto();
  });

  test("page loads with correct title", async ({ page }) => {
    await expect(page).toHaveTitle(/FORGE Demo/i);
  });

  test("header shows FORGE logo text", async () => {
    await expect(demo.header).toBeVisible();
    await expect(demo.header).toHaveText("FORGE");
  });

  test("header shows tagline", async ({ page }) => {
    const tagline = page.locator(".logo-tagline");
    await expect(tagline).toBeVisible();
    await expect(tagline).toContainText("Fast Open-source Runtime");
  });

  test("header meta chips are present", async ({ page }) => {
    await expect(page.locator("#chip-platform")).toBeVisible();
    await expect(page.locator("#chip-date")).toBeVisible();
    await expect(page.locator("#chip-result")).toBeVisible();
  });

  test("sidebar renders all 8 section buttons", async () => {
    await expect(demo.sidebar).toBeVisible();

    for (const key of DEMO_SECTIONS) {
      const btn = demo.sectionButton(key);
      await expect(btn).toBeVisible();
    }
  });

  test("section buttons have correct names", async ({ page }) => {
    const expectedNames = [
      "World Generation",
      "Navigation",
      "Resource Gathering",
      "Crafting",
      "Multi-Agent",
      "Day/Night Cycle",
      "Determinism",
      "Performance",
    ];

    for (const name of expectedNames) {
      await expect(page.locator(".section-name", { hasText: name })).toBeVisible();
    }
  });

  test("all section badges start as IDLE", async () => {
    for (const key of DEMO_SECTIONS) {
      await expect(demo.sectionBadge(key)).toHaveText("IDLE");
    }
  });

  test("footer controls render correctly", async () => {
    // Seed input
    await expect(demo.seedInput).toBeVisible();
    await expect(demo.seedInput).toHaveValue("42");

    // Quick toggle
    await expect(demo.quickToggle).toBeVisible();

    // Run All button
    await expect(demo.runAllButton).toBeVisible();
    await expect(demo.runAllButton).toHaveText("Run All");
    await expect(demo.runAllButton).toBeEnabled();

    // Stop button (disabled initially)
    await expect(demo.stopButton).toBeVisible();
    await expect(demo.stopButton).toBeDisabled();
  });

  test("stats panel renders key elements", async ({ page }) => {
    // Seed display
    await expect(page.locator("#stat-seed")).toBeVisible();

    // Steps/second
    await expect(page.locator("#stat-fps")).toBeVisible();

    // World canvas
    await expect(demo.worldCanvas).toBeVisible();

    // Inventory display
    await expect(demo.inventoryDisplay).toBeVisible();
    await expect(demo.inventoryDisplay).toContainText("No items yet");

    // Progress bar area
    await expect(demo.progressLabel).toBeVisible();
    await expect(demo.progressLabel).toHaveText("Idle");
  });

  test("terminal shows initial placeholder text", async () => {
    await expect(demo.terminalOutput).toContainText("Waiting for FORGE");
  });

  test("terminal prompt shows ready state", async () => {
    await expect(demo.terminalPrompt).toHaveText("ready");
  });

  test("day/night phase indicators are present", async ({ page }) => {
    const phases = ["dawn", "day", "dusk", "night"];
    for (const phase of phases) {
      await expect(page.locator(`#phase-${phase}`)).toBeVisible();
    }
  });

  test("footer shows elapsed time and agent/tick counters", async () => {
    await expect(demo.footerElapsed).toBeVisible();
    await expect(demo.footerTick).toHaveText(/Tick/);
    await expect(demo.footerAgents).toHaveText(/Agents/);
  });

  test("mini section results list is present", async ({ page }) => {
    const miniList = page.locator("#mini-section-list");
    await expect(miniList).toBeVisible();

    for (const key of DEMO_SECTIONS) {
      await expect(demo.miniStatus(key)).toBeVisible();
      await expect(demo.miniStatus(key)).toHaveText("IDLE");
    }
  });
});
