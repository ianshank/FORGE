import {
  test,
  expect,
  DemoPage,
  DEMO_SECTIONS,
} from "../fixtures/test-fixtures";

test.describe("Demo UI — Interaction Tests", () => {
  let demo: DemoPage;

  test.beforeEach(async ({ page }) => {
    demo = new DemoPage(page);
    await demo.goto();
  });

  test("seed input accepts numeric values", async () => {
    await demo.seedInput.fill("12345");
    await expect(demo.seedInput).toHaveValue("12345");
  });

  test("seed input rejects non-numeric values", async () => {
    await demo.seedInput.fill("abc");
    // Number inputs ignore non-numeric input, value stays empty or previous
    const value = await demo.seedInput.inputValue();
    expect(value === "" || value === "42").toBe(true);
  });

  test("quick toggle toggles on/off", async ({ page }) => {
    // Initially should have 'on' class
    await expect(demo.quickToggle).toHaveClass(/\bon\b/);

    // Click the toggle wrapper
    const toggleWrap = page.locator(".toggle-wrap");
    await toggleWrap.click();

    // Should no longer have 'on' class
    await expect(demo.quickToggle).not.toHaveClass(/\bon\b/);

    // Click again to re-enable
    await toggleWrap.click();
    await expect(demo.quickToggle).toHaveClass(/\bon\b/);
  });

  test("clear button clears terminal output", async () => {
    // Terminal has initial placeholder text
    await expect(demo.terminalOutput).toContainText("Waiting for FORGE");

    // Click clear
    await demo.clearButton.click();

    // Terminal should be empty (no text content)
    const text = await demo.terminalOutput.textContent();
    expect(text?.trim()).toBe("");
  });

  test("speed slider updates label", async ({ page }) => {
    const slider = page.locator("#speed-slider");
    const label = page.locator("#speed-label");

    // Default value
    await expect(label).toHaveText("5x");

    // Change slider value
    await slider.fill("8");
    await slider.dispatchEvent("input");
    await expect(label).toHaveText("8x");
  });

  test("section buttons are clickable and exist for all sections", async () => {
    for (const key of DEMO_SECTIONS) {
      const btn = demo.sectionButton(key);
      await expect(btn).toBeVisible();
      await expect(btn).toBeEnabled();
    }
  });

  test("run all button is enabled, stop button is disabled initially", async () => {
    await expect(demo.runAllButton).toBeEnabled();
    await expect(demo.stopButton).toBeDisabled();
  });

  test("world canvas has correct dimensions", async () => {
    const width = await demo.worldCanvas.getAttribute("width");
    const height = await demo.worldCanvas.getAttribute("height");
    expect(width).toBe("280");
    expect(height).toBe("280");
  });

  test("terrain legend is visible", async ({ page }) => {
    const legend = page.locator(".terrain-legend");
    await expect(legend).toBeVisible();

    const items = [
      "Ground",
      "Water",
      "Forest",
      "Sand",
      "Ice",
      "Lava",
      "Mountain",
      "Agent",
      "Resource",
      "Object",
    ];
    for (const item of items) {
      await expect(legend.locator(".legend-item", { hasText: item })).toBeVisible();
    }
  });
});
