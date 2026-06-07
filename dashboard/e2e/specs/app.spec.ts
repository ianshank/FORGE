import { AppNav } from "../pages/AppNav";
import { expect, test } from "../fixtures/test";

test.describe("App routing & navigation", () => {
  test("redirects / to /live", async ({ page }) => {
    const nav = new AppNav(page);
    await nav.goto("/");
    await expect(page).toHaveURL(/\/live$/);
    await expect(nav.title()).toHaveText("Live");
  });

  test("redirects an unknown route to /live", async ({ page }) => {
    await page.goto("/does-not-exist");
    await expect(page).toHaveURL(/\/live$/);
  });

  test("navigates between routes via the sidebar", async ({ page }) => {
    const nav = new AppNav(page);
    await nav.goto("/live");

    await nav.navLink("Settings").click();
    await expect(page).toHaveURL(/\/settings$/);
    await expect(nav.title()).toHaveText("Settings");

    await nav.navLink("Replay").click();
    await expect(page).toHaveURL(/\/replay$/);
    await expect(nav.title()).toHaveText("Replay");
  });

  test("loads the shell without console errors", async ({ page }) => {
    const errors: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error" && !msg.text().includes("favicon")) {
        errors.push(msg.text());
      }
    });
    await new AppNav(page).goto("/live");
    await expect(new AppNav(page).brand()).toBeVisible();
    expect(errors).toEqual([]);
  });
});
