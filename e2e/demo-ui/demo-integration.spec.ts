import {
  test,
  expect,
  DemoPage,
  DEMO_SECTIONS,
} from "../fixtures/test-fixtures";

test.describe("Demo UI — Integration Tests", () => {
  test("GET /health returns ok", async ({ request }) => {
    const resp = await request.get("/health");
    expect(resp.ok()).toBe(true);

    const body = await resp.json();
    expect(body).toEqual({ status: "ok" });
  });

  test("GET /api/sections returns all 8 sections", async ({ request }) => {
    const resp = await request.get("/api/sections");
    expect(resp.ok()).toBe(true);

    const sections = await resp.json();
    expect(sections).toHaveLength(8);

    // Verify structure
    for (const section of sections) {
      expect(section).toHaveProperty("key");
      expect(section).toHaveProperty("name");
      expect(section).toHaveProperty("index");
    }

    // Verify all expected keys are present
    const keys = sections.map((s: { key: string }) => s.key);
    for (const expected of DEMO_SECTIONS) {
      expect(keys).toContain(expected);
    }
  });

  test("GET /api/sections returns correct order and names", async ({
    request,
  }) => {
    const resp = await request.get("/api/sections");
    const sections = await resp.json();

    const expectedOrder = [
      { key: "worldgen", name: "World Generation" },
      { key: "navigation", name: "Navigation" },
      { key: "gathering", name: "Resource Gathering" },
      { key: "crafting", name: "Crafting" },
      { key: "multiagent", name: "Multi-Agent" },
      { key: "daynight", name: "Day/Night Cycle" },
      { key: "determinism", name: "Determinism" },
      { key: "performance", name: "Performance" },
    ];

    for (let i = 0; i < expectedOrder.length; i++) {
      expect(sections[i].key).toBe(expectedOrder[i].key);
      expect(sections[i].name).toBe(expectedOrder[i].name);
      expect(sections[i].index).toBe(i + 1);
    }
  });

  test("POST /api/run/{section} returns SSE stream", async ({ request }) => {
    const resp = await request.post("/api/run/worldgen", {
      data: { seed: 42, quick: true },
    });

    expect(resp.ok()).toBe(true);
    expect(resp.headers()["content-type"]).toContain("text/event-stream");

    const body = await resp.text();
    // SSE format: lines starting with "data: "
    expect(body).toContain("data: ");
    // Stream should end with the sentinel
    expect(body).toContain("__STREAM_END__");
  });

  test("POST /api/run/{section} rejects unknown sections", async ({
    request,
  }) => {
    const resp = await request.post("/api/run/nonexistent", {
      data: { seed: 42, quick: true },
    });

    expect(resp.status()).toBe(404);
  });

  test("GET /api/results returns valid JSON structure", async ({ request }) => {
    const resp = await request.get("/api/results");
    expect(resp.ok()).toBe(true);

    const data = await resp.json();
    // The results endpoint returns parsed demo_results.md
    // It may be empty if no results exist yet, but should be a valid object
    expect(typeof data).toBe("object");
  });

  test("run section via UI updates terminal and badge", async ({ page }) => {
    const demo = new DemoPage(page);
    await demo.goto();

    // Mock the SSE endpoint to return a controlled stream
    await page.route("**/api/run/worldgen", (route) =>
      route.fulfill({
        status: 200,
        contentType: "text/event-stream",
        headers: {
          "Cache-Control": "no-cache",
          "X-Accel-Buffering": "no",
        },
        body: [
          'data: "=== World Generation ==="\n\n',
          'data: "Generating 16x16 grid with seed 42"\n\n',
          'data: "PASS"\n\n',
          'data: "__STREAM_END__"\n\n',
        ].join(""),
      }),
    );

    // Click the worldgen section button
    await demo.sectionButton("worldgen").click();

    // Wait for terminal to show output
    await expect(demo.terminalOutput).toContainText("World Generation", {
      timeout: 10_000,
    });

    // Badge should update to PASS
    await expect(demo.sectionBadge("worldgen")).toHaveText("PASS", {
      timeout: 5_000,
    });
  });

  test("run all via UI triggers progress updates", async ({ page }) => {
    const demo = new DemoPage(page);
    await demo.goto();

    // Mock the run-all SSE endpoint
    await page.route("**/api/run-all", (route) =>
      route.fulfill({
        status: 200,
        contentType: "text/event-stream",
        headers: {
          "Cache-Control": "no-cache",
          "X-Accel-Buffering": "no",
        },
        body: [
          'data: "__SECTION_START__ worldgen"\n\n',
          'data: "=== World Generation ==="\n\n',
          'data: "PASS"\n\n',
          'data: "__SECTION_END__ worldgen"\n\n',
          'data: "__SECTION_START__ navigation"\n\n',
          'data: "=== Navigation ==="\n\n',
          'data: "PASS"\n\n',
          'data: "__SECTION_END__ navigation"\n\n',
          'data: "__STREAM_END__"\n\n',
        ].join(""),
      }),
    );

    // Click Run All
    await demo.runAllButton.click();

    // Terminal should show output
    await expect(demo.terminalOutput).toContainText("World Generation", {
      timeout: 10_000,
    });

    // Badges should update
    await expect(demo.sectionBadge("worldgen")).toHaveText("PASS", {
      timeout: 5_000,
    });
    await expect(demo.sectionBadge("navigation")).toHaveText("PASS", {
      timeout: 5_000,
    });

    // Progress label should show completed sections
    await expect(demo.progressLabel).toContainText(/2\s*\/\s*8/);
  });

  test("stop button aborts a running stream", async ({ page }) => {
    const demo = new DemoPage(page);
    await demo.goto();

    // Create a slow SSE stream that takes time
    await page.route("**/api/run-all", async (route) => {
      // Return a response that streams slowly — just fulfill immediately
      // to simulate a stream that starts but the UI will abort
      await route.fulfill({
        status: 200,
        contentType: "text/event-stream",
        headers: { "Cache-Control": "no-cache" },
        body: 'data: "__SECTION_START__ worldgen"\n\ndata: "Starting..."\n\n',
      });
    });

    // Start run
    await demo.runAllButton.click();

    // Stop should become enabled
    await expect(demo.stopButton).toBeEnabled({ timeout: 5_000 });

    // Click stop
    await demo.stopButton.click();

    // Run All should be re-enabled
    await expect(demo.runAllButton).toBeEnabled({ timeout: 5_000 });

    // Prompt should show ready
    await expect(demo.terminalPrompt).toHaveText("ready");
  });

  test("results are loaded on page init", async ({ page }) => {
    const demo = new DemoPage(page);

    // Mock results endpoint
    await page.route("**/api/results", (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          platform: "Linux x86_64",
          date: "2026-03-08",
          result: "8/8 PASS",
          seed: 99,
          performance: {
            steps_per_second: "1,234,567",
            us_per_step: "0.81",
          },
        }),
      }),
    );

    await demo.goto();

    // Verify the results are displayed
    await expect(page.locator("#chip-platform")).toHaveText("Linux x86_64");
    await expect(page.locator("#chip-date")).toHaveText("2026-03-08");
    await expect(page.locator("#chip-result")).toHaveText("8/8 PASS");
    await expect(page.locator("#stat-seed")).toHaveText("99");
    await expect(page.locator("#stat-fps")).toHaveText("1,234,567");
  });
});
