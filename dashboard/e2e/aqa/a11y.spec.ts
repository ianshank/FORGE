import AxeBuilder from "@axe-core/playwright";
import { A11Y_DISABLED_RULES, A11Y_EXCLUDES } from "./allowlist";
import { expect, test } from "../fixtures/test";

/** Routes scanned for WCAG 2 A/AA accessibility violations. */
const ROUTES = ["/live", "/training", "/runs", "/replay", "/demos", "/settings"];

test.describe("Accessibility (axe-core, WCAG 2 A/AA)", () => {
  for (const route of ROUTES) {
    test(`has no violations: ${route}`, async ({ page }) => {
      await page.goto(route);
      // Wait for the page chrome to settle before scanning.
      await page.getByRole("heading", { level: 1 }).waitFor();

      let builder = new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa"])
        .disableRules(A11Y_DISABLED_RULES);
      for (const selector of A11Y_EXCLUDES) {
        builder = builder.exclude(selector);
      }

      const { violations } = await builder.analyze();
      expect(
        violations,
        violations.map((v) => `${v.id}: ${v.help}`).join("\n"),
      ).toEqual([]);
    });
  }
});
