import { test as base, type Page, type Locator } from "@playwright/test";

// ---------------------------------------------------------------------------
// Dashboard page-object helper
// ---------------------------------------------------------------------------

export class DashboardPage {
  readonly page: Page;
  readonly header: Locator;
  readonly connectionDot: Locator;
  readonly connectionLabel: Locator;
  readonly scenarioControls: Locator;
  readonly simulationCanvas: Locator;
  readonly decisionTracePanel: Locator;
  readonly metricsDashboard: Locator;
  readonly agentInspector: Locator;
  readonly seedInput: Locator;
  readonly gridSlider: Locator;
  readonly remixButton: Locator;

  constructor(page: Page) {
    this.page = page;
    this.header = page.locator("h1");
    this.connectionDot = page.locator("header span.rounded-full");
    this.connectionLabel = page.locator("header span.text-gray-400");
    this.scenarioControls = page.locator(
      "div.bg-gray-900.border.border-gray-700.rounded.p-3.flex.items-center",
    );
    this.simulationCanvas = page.locator("canvas");
    this.decisionTracePanel = page.locator("h3", {
      hasText: "Decision Traces",
    });
    this.metricsDashboard = page.locator("text=No training metrics available");
    this.agentInspector = page.locator("text=Click an agent to inspect");
    this.seedInput = page.locator('input[type="number"]');
    this.gridSlider = page.locator('input[type="range"]');
    this.remixButton = page.getByRole("button", { name: /Remix/i });
  }

  async goto() {
    await this.page.goto("/");
    await this.page.waitForLoadState("domcontentloaded");
  }
}

// ---------------------------------------------------------------------------
// Demo UI page-object helper
// ---------------------------------------------------------------------------

/** All 8 demo section keys in order. */
export const DEMO_SECTIONS = [
  "worldgen",
  "navigation",
  "gathering",
  "crafting",
  "multiagent",
  "daynight",
  "determinism",
  "performance",
] as const;

export class DemoPage {
  readonly page: Page;
  readonly header: Locator;
  readonly sidebar: Locator;
  readonly terminalOutput: Locator;
  readonly terminalPrompt: Locator;
  readonly seedInput: Locator;
  readonly quickToggle: Locator;
  readonly runAllButton: Locator;
  readonly stopButton: Locator;
  readonly clearButton: Locator;
  readonly progressBar: Locator;
  readonly progressLabel: Locator;
  readonly worldCanvas: Locator;
  readonly inventoryDisplay: Locator;
  readonly footerElapsed: Locator;
  readonly footerTick: Locator;
  readonly footerAgents: Locator;

  constructor(page: Page) {
    this.page = page;
    this.header = page.locator(".logo-glow");
    this.sidebar = page.locator("#sidebar");
    this.terminalOutput = page.locator("#terminal-output");
    this.terminalPrompt = page.locator("#prompt-text");
    this.seedInput = page.locator("#seed-input");
    this.quickToggle = page.locator("#quick-toggle");
    this.runAllButton = page.locator("#btn-run-all");
    this.stopButton = page.locator("#btn-stop");
    this.clearButton = page.locator("#btn-clear");
    this.progressBar = page.locator("#progress-bar");
    this.progressLabel = page.locator("#progress-label");
    this.worldCanvas = page.locator("#world-canvas");
    this.inventoryDisplay = page.locator("#inventory-display");
    this.footerElapsed = page.locator("#footer-elapsed");
    this.footerTick = page.locator("#footer-tick");
    this.footerAgents = page.locator("#footer-agents");
  }

  async goto() {
    await this.page.goto("/");
    await this.page.waitForLoadState("domcontentloaded");
  }

  sectionButton(key: string): Locator {
    return this.page.locator(`#sbtn-${key}`);
  }

  sectionBadge(key: string): Locator {
    return this.page.locator(`#badge-${key}`);
  }

  miniStatus(key: string): Locator {
    return this.page.locator(`#mini-${key}`);
  }
}

// ---------------------------------------------------------------------------
// Extended test fixture that exposes both helpers
// ---------------------------------------------------------------------------

type Fixtures = {
  dashboardPage: DashboardPage;
  demoPage: DemoPage;
};

export const test = base.extend<Fixtures>({
  dashboardPage: async ({ page }, use) => {
    await use(new DashboardPage(page));
  },
  demoPage: async ({ page }, use) => {
    await use(new DemoPage(page));
  },
});

export { expect } from "@playwright/test";
