import type { Locator, Page } from "@playwright/test";

/** Shared chrome: sidebar navigation + top-bar title/connection status. */
export class AppNav {
  constructor(private readonly page: Page) {}

  async goto(path = "/live"): Promise<void> {
    await this.page.goto(path);
  }

  /** Sidebar nav link by visible name (Live, Training, Runs, Replay, Demos, Settings). */
  navLink(name: string): Locator {
    return this.page.getByRole("link", { name });
  }

  /** The top-bar page title (h1). */
  title(): Locator {
    return this.page.getByRole("heading", { level: 1 });
  }

  /** Connection status text in the top bar (Connecting/Connected/Disconnected). */
  connectionStatus(): Locator {
    // The TopBar renders the status label text next to the StatusDot.
    return this.page.getByText(/^(Connecting|Connected|Disconnected)$/);
  }

  /** The FORGE wordmark in the sidebar. */
  brand(): Locator {
    return this.page.getByText("FORGE", { exact: true });
  }
}
