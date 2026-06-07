import type { Locator, Page } from "@playwright/test";

/** Page object for the Replay view (/replay). */
export class ReplayPage {
  constructor(private readonly page: Page) {}

  /** Upload a trajectory JSON into the hidden file input. */
  async upload(filePath: string): Promise<void> {
    await this.page.locator('input[type="file"]').setInputFiles(filePath);
  }

  alert(): Locator {
    return this.page.getByRole("alert");
  }

  stepForward(): Locator {
    return this.page.getByRole("button", { name: "Step forward" });
  }

  stepBack(): Locator {
    return this.page.getByRole("button", { name: "Step back" });
  }

  playPause(): Locator {
    return this.page.getByRole("button", { name: /Play|Pause/ });
  }

  speedButton(label: string): Locator {
    return this.page.getByRole("button", { name: label });
  }

  /** The "N / M" cursor position text. */
  position(): Locator {
    return this.page.getByText(/^\d+ \/ \d+$/);
  }
}
