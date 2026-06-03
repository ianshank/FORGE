import { afterEach, describe, expect, it, vi } from "vitest";
import { CHART_SERIES, resolveChartTheme } from "../lib/chartTheme";

describe("CHART_SERIES", () => {
  it("exposes a stable, non-empty color per series", () => {
    for (const color of Object.values(CHART_SERIES)) {
      expect(color).toMatch(/^#[0-9a-fA-F]{6}$/);
    }
  });
});

describe("resolveChartTheme", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("falls back to token defaults when custom properties are unset", () => {
    // jsdom's getComputedStyle returns empty strings for custom properties.
    const theme = resolveChartTheme(document.documentElement);
    expect(theme.grid).toBe("hsl(217 33% 15%)");
    expect(theme.axis).toBe("hsl(217 33% 18%)");
    expect(theme.tooltipBg).toBe("hsl(222 44% 6%)");
    expect(theme.axisText).toBe("hsl(215 18% 58%)");
  });

  it("reads resolved custom properties from the provided root", () => {
    const fakeRoot = {} as Element;
    vi.spyOn(window, "getComputedStyle").mockReturnValue({
      getPropertyValue: (name: string) =>
        name === "--chart-grid" ? " 10 20% 30% " : "1 2% 3%",
    } as unknown as CSSStyleDeclaration);

    const theme = resolveChartTheme(fakeRoot);
    // Whitespace is trimmed; the resolved channel is wrapped in hsl().
    expect(theme.grid).toBe("hsl(10 20% 30%)");
    expect(theme.axis).toBe("hsl(1 2% 3%)");
  });

  it("does not throw when no document is available", () => {
    vi.spyOn(window, "getComputedStyle").mockImplementation(() => {
      throw new Error("no layout");
    });
    const theme = resolveChartTheme(document.documentElement);
    // Every field is still a valid hsl() string from the fallbacks.
    for (const value of Object.values(theme)) {
      expect(value).toMatch(/^hsl\(/);
    }
  });
});
