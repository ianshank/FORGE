/**
 * Accessibility triage allowlist for the axe-core scans.
 *
 * Anything listed here is an explicit, reviewable exception rather than a
 * silent pass. Prefer fixing the source over extending these lists.
 */

/** Selectors excluded from axe analysis (no accessible representation). */
export const A11Y_EXCLUDES: string[] = [
  // The world view is a <canvas>; agent state is mirrored non-visually in the
  // Agent Inspector. Documented exemption (see SimulationCanvas.tsx).
  '[data-testid="world-canvas"]',
];

/**
 * axe rules disabled pending follow-up. Keep this list short and tracked.
 */
export const A11Y_DISABLED_RULES: string[] = [
  // Dark-theme colour contrast needs a dedicated design pass before it can be
  // enforced; tracked as a follow-up. Re-enable once tokens are tuned.
  "color-contrast",
];
