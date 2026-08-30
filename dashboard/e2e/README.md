# Dashboard E2E + AQA (Playwright)

End-to-end UI tests and automated accessibility checks for the FORGE Control
Center. Tests run against the **production build** (`vite preview`) in Chromium
with **all backends mocked in-browser** — no forge-server or demo backend
required.

## Layout

```
playwright.config.ts     # webServer (build+preview on :4173), chromium project
e2e/
  fixtures/              # test.ts (composed mocks), mockWebSocket, mockBackend, mockData
  pages/                # Page Object Models
  specs/                # functional flows (app, live, replay, demos, pages)
  aqa/                  # a11y.spec.ts (axe-core) + allowlist.ts (triage)
  test-data/            # trajectory JSON fixtures for the replay upload
```

## How mocking works

- **WebSocket**: a fake `window.WebSocket` is injected via `addInitScript`
  before the app mounts (`fixtures/mockWebSocket.ts`). It auto-opens and emits a
  seeded `StateUpdate`; specs push more frames with `emitWsMessage`.
- **REST/SSE**: `page.route(...)` fulfils `/api/metrics`, `/api/scenario/remix`,
  the demo `/api/run/*` SSE stream, and the three history endpoints
  (`/api/runs`, `/api/training-metrics/history`,
  `/api/decision-traces/history`) that the app polls (`fixtures/mockBackend.ts`).
  Every endpoint the app calls must be mocked: an unmocked fetch escapes to the
  real `apiBaseUrl`, which nothing serves here, and the resulting console error
  fails the "loads the shell without console errors" spec. Error cases
  re-`route` before navigating (last handler wins).

## Running

```bash
npm ci
npm run test:e2e:install   # one-time: downloads Chromium (+ OS deps)
npm run test:e2e           # build → preview → run specs + a11y
npm run test:e2e:ui        # interactive debugging
npm run test:a11y          # accessibility scans only
npm run typecheck:e2e      # type-check e2e/ without emitting
```

> **Sandbox note**: some restricted environments block the Playwright browser
> download, so `test:e2e:install` (and therefore `test:e2e`) cannot run locally
> there. Use the official image instead —
> `docker run --rm -v "$PWD":/work -w /work/dashboard mcr.microsoft.com/playwright:v1.60.0-jammy npm run test:e2e`
> — or rely on the `dashboard-e2e` CI job, which installs Chromium and runs the
> full suite on every push.

## CI

The `dashboard-e2e` job (`.github/workflows/ci.yml`) installs Chromium (cached),
builds, and runs the suite. It is **non-blocking** initially; failures upload an
HTML report + traces as artifacts (open with `npx playwright show-trace`).

## Accessibility triage

`aqa/allowlist.ts` holds the explicit, reviewable exceptions: the `<canvas>` is
excluded (no accessible representation; state mirrored in the Agent Inspector),
and `color-contrast` is temporarily disabled pending a dark-theme design pass.
