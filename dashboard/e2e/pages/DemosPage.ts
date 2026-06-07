import type { Locator, Page } from "@playwright/test";

/** Page object for the Demos view (/demos). */
export class DemosPage {
  constructor(private readonly page: Page) {}

  sectionButton(name: string | RegExp): Locator {
    return this.page.getByRole("button", { name });
  }

  output(): Locator {
    return this.page.getByTestId("demo-output");
  }

  runningBadge(): Locator {
    return this.page.getByText("running", { exact: true });
  }

  doneBadge(): Locator {
    return this.page.getByText("done", { exact: true });
  }
}
