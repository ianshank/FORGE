import type { Locator, Page } from "@playwright/test";

/** Page object for the Live view (/live). */
export class LivePage {
  constructor(private readonly page: Page) {}

  /** A KPI tile by its label (Tick, Agents Alive, Steps / sec, Server Uptime). */
  statCard(label: string): Locator {
    return this.page.locator(`[data-testid="stat-card"][data-stat-label="${label}"]`);
  }

  worldCanvas(): Locator {
    return this.page.getByTestId("world-canvas");
  }

  seedInput(): Locator {
    return this.page.getByLabel("Random seed");
  }

  remixButton(): Locator {
    return this.page.getByRole("button", { name: /Remix/ });
  }

  alert(): Locator {
    return this.page.getByRole("alert");
  }

  /** Click a grid cell centre given cell coords and the configured cell size. */
  async clickCell(x: number, y: number, cellSize = 8): Promise<void> {
    const px = x * cellSize + cellSize / 2;
    const py = y * cellSize + cellSize / 2;
    await this.worldCanvas().click({ position: { x: px, y: py } });
  }
}
