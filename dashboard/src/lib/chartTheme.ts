/**
 * Centralised Recharts theming.
 *
 * Recharts renders SVG presentation attributes (e.g. `stroke`) where CSS
 * `var()` references do not resolve, so chart colors cannot simply be Tailwind
 * classes. This module is the single source of truth: it reads the design
 * tokens from the live DOM at runtime (so a future theme switch is honoured)
 * and falls back to the token defaults when the DOM is unavailable (SSR /
 * jsdom / before first paint).
 */

import { createLogger } from "../utils/logger";

const log = createLogger("chartTheme");

/**
 * Semantic series colors for charts. Single source of truth shared by every
 * chart so a palette change happens in exactly one place.
 */
export const CHART_SERIES = {
  reward: "#39ff7e",
  winRate: "#38bdf8",
  stepsPerSecond: "#ffb347",
  entropy: "#a855f7",
  /** Cumulative-reward trend used by the replay timeline. */
  cumulative: "#39ff7e",
} as const;

/** Names of the CSS custom properties (HSL channel triplets) we resolve. */
const TOKEN_VARS = {
  grid: "--chart-grid",
  axis: "--chart-axis",
  axisText: "--muted-foreground",
  tooltipBg: "--popover",
  tooltipBorder: "--border",
  tooltipText: "--muted-foreground",
  cursor: "--primary",
} as const;

/** Token defaults — last-resort fallbacks mirroring `index.css`. */
const FALLBACK_CHANNELS: Record<keyof typeof TOKEN_VARS, string> = {
  grid: "217 33% 15%",
  axis: "217 33% 18%",
  axisText: "215 18% 58%",
  tooltipBg: "222 44% 6%",
  tooltipBorder: "217 33% 15%",
  tooltipText: "215 18% 58%",
  cursor: "196 90% 55%",
};

/** Resolved, ready-to-use color strings for chart primitives. */
export interface ChartTheme {
  grid: string;
  axis: string;
  axisText: string;
  tooltipBg: string;
  tooltipBorder: string;
  tooltipText: string;
  /** Accent for cursor/reference markers. */
  cursor: string;
}

/**
 * Read a CSS custom property (an HSL channel triplet) from `root`, returning
 * the trimmed value or `null` when it is unset/unavailable.
 */
function readChannel(root: Element, name: string): string | null {
  try {
    const value = getComputedStyle(root).getPropertyValue(name).trim();
    return value.length > 0 ? value : null;
  } catch {
    return null;
  }
}

/**
 * Resolve the chart theme from the document's design tokens.
 *
 * @param root - Element to read custom properties from. Defaults to
 *   `document.documentElement`. Pass an explicit element in tests.
 */
export function resolveChartTheme(root?: Element | null): ChartTheme {
  const el =
    root ??
    (typeof document !== "undefined" ? document.documentElement : null);

  const channel = (key: keyof typeof TOKEN_VARS): string => {
    const resolved = el ? readChannel(el, TOKEN_VARS[key]) : null;
    if (resolved === null) {
      log.debug("Falling back to default for %s", TOKEN_VARS[key]);
      return FALLBACK_CHANNELS[key];
    }
    return resolved;
  };

  return {
    grid: `hsl(${channel("grid")})`,
    axis: `hsl(${channel("axis")})`,
    axisText: `hsl(${channel("axisText")})`,
    tooltipBg: `hsl(${channel("tooltipBg")})`,
    tooltipBorder: `hsl(${channel("tooltipBorder")})`,
    tooltipText: `hsl(${channel("tooltipText")})`,
    cursor: `hsl(${channel("cursor")})`,
  };
}
