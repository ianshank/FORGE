import { AppNav } from "../pages/AppNav";
import { LivePage } from "../pages/LivePage";
import { emitWsMessage } from "../fixtures/mockWebSocket";
import { expect, test } from "../fixtures/test";

test.describe("Live view", () => {
  test("connects and shows live tick + agent counts", async ({ page }) => {
    const nav = new AppNav(page);
    await nav.goto("/live");
    await expect(nav.connectionStatus()).toHaveText("Connected");

    const live = new LivePage(page);
    await expect(live.statCard("Tick")).toContainText("42");
    // 1 alive of 2 agents in the seeded StateUpdate.
    await expect(live.statCard("Agents Alive")).toContainText("/ 2");
  });

  test("stat cards reflect mocked server metrics", async ({ page }) => {
    await new AppNav(page).goto("/live");
    const live = new LivePage(page);
    await expect(live.statCard("Steps / sec")).toContainText("25");
    // uptimeSeconds 120 → formatDuration "2m 0s".
    await expect(live.statCard("Server Uptime")).toContainText("2m");
  });

  test("remix succeeds without an error", async ({ page }) => {
    await new AppNav(page).goto("/live");
    const live = new LivePage(page);
    await live.seedInput().fill("99");
    await live.remixButton().click();
    await expect(live.alert()).toHaveCount(0);
  });

  test("remix surfaces an HTTP error", async ({ page }) => {
    await page.route("**/api/scenario/remix", (route) =>
      route.fulfill({ status: 500, json: { success: false } }),
    );
    await new AppNav(page).goto("/live");
    const live = new LivePage(page);
    await live.remixButton().click();
    await expect(live.alert()).toContainText("HTTP 500");
  });

  test("reflects live WebSocket state updates", async ({ page }) => {
    const nav = new AppNav(page);
    await nav.goto("/live");
    const live = new LivePage(page);
    await expect(live.statCard("Tick")).toContainText("42");

    // Push a fresh frame over the mocked socket; the UI must update live.
    await emitWsMessage(page, {
      type: "StateUpdate",
      payload: {
        tick: 1000,
        agents: [],
        gridWidth: 8,
        gridHeight: 8,
        events: [],
        schemaVersion: 1,
      },
    });

    await expect(live.statCard("Tick")).toContainText("1,000");
    await expect(live.statCard("Agents Alive")).toContainText("/ 0");
  });

  test("selects an agent on canvas click @canvas", async ({ page }) => {
    await new AppNav(page).goto("/live");
    const live = new LivePage(page);
    await live.clickCell(1, 1, 8); // agent #0 sits at grid (1,1)
    await expect(page.getByText("#0")).toBeVisible();
  });
});
