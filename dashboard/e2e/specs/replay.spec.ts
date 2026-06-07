import path from "node:path";
import { ReplayPage } from "../pages/ReplayPage";
import { expect, test } from "../fixtures/test";

const dataFile = (name: string) =>
  path.join(process.cwd(), "e2e", "test-data", name);

test.describe("Replay view", () => {
  test("uploads a trajectory and renders summary + step detail", async ({
    page,
  }) => {
    await page.goto("/replay");
    const replay = new ReplayPage(page);
    await replay.upload(dataFile("trajectory.valid.json"));

    await expect(page.getByText("ep-e2e")).toBeVisible();
    await expect(replay.position()).toHaveText("1 / 4");
    await expect(page.getByText("Step Detail")).toBeVisible();
  });

  test("steps forward and backward", async ({ page }) => {
    await page.goto("/replay");
    const replay = new ReplayPage(page);
    await replay.upload(dataFile("trajectory.valid.json"));

    await replay.stepForward().click();
    await expect(replay.position()).toHaveText("2 / 4");
    await replay.stepBack().click();
    await expect(replay.position()).toHaveText("1 / 4");
  });

  test("plays through and auto-stops at the last step", async ({ page }) => {
    await page.goto("/replay");
    const replay = new ReplayPage(page);
    await replay.upload(dataFile("trajectory.valid.json"));

    await replay.speedButton("30×").click();
    await replay.playPause().click();
    await expect(replay.position()).toHaveText("4 / 4");
  });

  test("rejects a malformed trajectory", async ({ page }) => {
    await page.goto("/replay");
    const replay = new ReplayPage(page);
    await replay.upload(dataFile("trajectory.malformed.json"));
    await expect(replay.alert()).toContainText("steps");
  });

  test("rejects an empty trajectory", async ({ page }) => {
    await page.goto("/replay");
    const replay = new ReplayPage(page);
    await replay.upload(dataFile("trajectory.empty.json"));
    await expect(replay.alert()).toContainText("no steps");
  });
});
