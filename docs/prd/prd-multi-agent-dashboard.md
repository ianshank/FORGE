# PRD — E13: Multi-Agent Dashboard UI

**Epic Slug**: `multi-agent-dashboard`
**Priority**: P2
**Sprint**: 7
**Size**: M
**Depends on**: E12 (REST Env API — impl in `python/forge_env/api.py ✅`)

---

## User Story

> **As a** researcher running multi-agent FORGE experiments,
> **I want to** see each agent's position, reward curve, and communication tokens rendered live in the demo UI,
> **So that** I can visually inspect emergent coordination without leaving the browser.

---

## Problem Statement

The current demo UI renders a monolithic ASCII world. With ≥2 agents, users cannot distinguish between them, track per-agent reward accumulation, or observe agent-level communication signals. A dedicated multi-agent view unlocks intuitive debugging and shareable screenshots for papers.

---

## Acceptance Criteria

| # | Given | When | Then |
|---|---|---|---|
| AC1 | User selects `num_agents ≥ 2` in the session config panel | Session is created | Each agent gets a unique color badge and legend entry |
| AC2 | The world canvas is active | A step is taken | Each agent's position is highlighted with its assigned color |
| AC3 | Episode is running | Any agent receives a reward | Each agent's reward curve (sparkline) updates in real-time |
| AC4 | The environment exposes `info["comm_tokens"]` | Step call returns | Comm-token badge row appears below each agent's sparkline with token counts |
| AC5 | User clicks an agent badge | Agent is selected | Canvas overlays that agent's trajectory path as a polyline for the last 50 steps |
| AC6 | Page is resized to mobile width (<768px) | Layout reflows | Agent panel collapses to a horizontal scrollable tab strip |
| AC7 | `num_agents = 1` | Session is created | Dashboard renders identical to single-agent mode (no regressions) |

---

## Out of Scope

- Agent communication message decoding (post-beta)
- Video export of the multi-agent session (see E11)
- Network-synchronized multi-user viewing

---

## Success Metrics

- Agent panels render within 16 ms of each step response (60 fps target)
- Zero JS console errors with 1–8 agents
- Playwright E2E tests cover AC1–AC7

---

## Open Questions

1. Should reward sparklines use canvas or SVG? (SVG preferred for accessibility)
2. What color palette? Recommend WCAG-AA-accessible set (e.g., Oklab rotation)
3. Should comm-token display be gated behind a feature flag until the env exposes `info["comm_tokens"]`?

---

## Implementation Notes

### Backend (no changes needed)

- `python/forge_env/api.py` already returns `info` dict from `/step` — extend format by consuming `info["comm_tokens"]` if present.

### Frontend changes (`demo_ui/frontend/`)

- `app.js`: add `AgentPanel` component class; parse `info.comm_tokens` from step response
- `styles.css`: `.agent-panel`, `.sparkline`, `.comm-token-badge` CSS tokens
- Color palette: define as CSS custom properties `--agent-0-color` … `--agent-7-color`
- Trajectory overlay: `<canvas>` with `requestAnimationFrame` polyline draw
- Responsive: media query `@media (max-width: 768px)` tab strip

### Tests

- `tests/python/test_api.py`: assert `info` key in step response (already exists)
- `demo_ui/tests/test_e2e_dashboard.py`: Playwright AC1–AC7 coverage
- `demo_ui/tests/test_backend.py`: unit test `main.py` endpoints

### CI

- Add `test_e2e_dashboard.py` to `demo_ui_ci.yml`
