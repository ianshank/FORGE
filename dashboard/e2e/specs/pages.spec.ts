import { AppNav } from "../pages/AppNav";
import { expect, test } from "../fixtures/test";

test.describe("Training view", () => {
  test("switches between Curves and Compare tabs", async ({ page }) => {
    await new AppNav(page).goto("/training");
    await expect(page.getByText("No training metrics yet")).toBeVisible();

    await page.getByRole("tab", { name: "Compare runs" }).click();
    await expect(page.getByText("Select runs to compare")).toBeVisible();
  });
});

test.describe("Runs view", () => {
  test("renders the empty runs state", async ({ page }) => {
    await new AppNav(page).goto("/runs");
    await expect(page.getByText("No runs recorded")).toBeVisible();
  });
});

test.describe("Settings view", () => {
  test("renders the runtime configuration table", async ({ page }) => {
    await new AppNav(page).goto("/settings");
    await expect(page.getByText("Runtime Configuration")).toBeVisible();
    await expect(page.getByText("VITE_WS_URL")).toBeVisible();
    await expect(page.getByText("VITE_DEMO_API_BASE_URL")).toBeVisible();
  });
});
