import { MOCKED_API_PATTERNS } from "../fixtures/mockBackend";
import {
  ALLOWED_EXTERNAL_ORIGINS,
  globToRegExp,
} from "../fixtures/networkGuard";
import { expect, test } from "../fixtures/test";

/**
 * Routes scanned for network escapes. Mirrors `a11y.spec.ts` — every
 * route in the app, so a hook added to any page is covered.
 */
const ROUTES = ["/live", "/training", "/runs", "/replay", "/demos", "/settings"];

/**
 * Long enough to out-wait the app's slowest poll. `useDecisionTraces`
 * and `useTrainingHistory` poll on a 2s interval, so a single settle
 * would miss an endpoint that is only requested on the second tick —
 * which is exactly how the original escape hid.
 */
const POLL_SETTLE_MS = 2_500;

test.describe("Network isolation (no request escapes the mocks)", () => {
  for (const route of ROUTES) {
    test(`stays inside the mocks: ${route}`, async ({ page }) => {
      await page.goto(route);
      await page.getByRole("heading", { level: 1 }).waitFor();
      // Let at least one poll interval elapse before the fixture's
      // teardown assertion runs.
      await page.waitForTimeout(POLL_SETTLE_MS);
    });
  }

  test("every mocked pattern is reachable as a matcher", () => {
    // Guards the guard: a pattern that compiles to a RegExp matching
    // nothing would silently stop covering its endpoint, and the escape
    // it exists to catch would come back unnoticed.
    for (const pattern of MOCKED_API_PATTERNS) {
      const matcher = globToRegExp(pattern);
      const concrete = pattern
        .replace("**", "http://localhost:8080")
        .replaceAll("*", "x");
      expect(
        matcher.test(concrete),
        `${pattern} does not match its own concrete form ${concrete}`,
      ).toBe(true);
    }
  });

  test("glob translation honours Playwright's separator rules", () => {
    // `**` crosses `/`, a single `*` does not. Getting this backwards
    // would make `**/api/run/*` match `/api/run/a/b`, hiding a genuinely
    // unmocked nested endpoint.
    const single = globToRegExp("**/api/run/*");
    expect(single.test("http://localhost:8000/api/run/demo")).toBe(true);
    expect(single.test("http://localhost:8000/api/run/demo/extra")).toBe(false);

    // Regex metacharacters in a pattern must stay literal.
    expect(globToRegExp("**/api/a.c").test("http://x/api/a.c")).toBe(true);
    expect(globToRegExp("**/api/a.c").test("http://x/api/abc")).toBe(false);
  });

  test("external origins are an explicit, reviewable list", () => {
    // An empty allowlist would mean the font imports fail every spec; a
    // wildcard would mean the guard checks nothing. Both are mistakes
    // this pins against.
    expect(ALLOWED_EXTERNAL_ORIGINS.length).toBeGreaterThan(0);
    for (const origin of ALLOWED_EXTERNAL_ORIGINS) {
      expect(() => new URL(origin)).not.toThrow();
      expect(new URL(origin).origin, `${origin} must be a bare origin`).toBe(
        origin,
      );
    }
  });
});
