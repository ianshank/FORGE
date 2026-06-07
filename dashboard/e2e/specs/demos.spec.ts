import { DemosPage } from "../pages/DemosPage";
import { expect, test } from "../fixtures/test";

test.describe("Demos view", () => {
  test("runs a section and streams output to completion", async ({ page }) => {
    await page.goto("/demos");
    const demos = new DemosPage(page);
    await demos.sectionButton(/World Generation/).click();

    await expect(demos.output()).toContainText("Generating 8x8 world");
    await expect(demos.output()).toContainText("Simulation complete");
    await expect(demos.doneBadge()).toBeVisible();
  });

  test("reports a backend error", async ({ page }) => {
    await page.route("**/api/run/*", (route) =>
      route.fulfill({ status: 502, body: "" }),
    );
    await page.goto("/demos");
    const demos = new DemosPage(page);
    await demos.sectionButton(/Navigation/).click();
    await expect(demos.output()).toContainText("502");
  });
});
