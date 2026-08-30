# WASM Demo E2E (Playwright)

End-to-end tests for the static in-browser WebAssembly demo in [`web/`](../../web).

Unlike [`dashboard/e2e`](../../dashboard/e2e), **nothing is mocked**. The demo is
fully client-side, so these specs drive the real deploy artifact —
`web/index.html` + `web/app.js` + the wasm-pack `--target web` output in
`web/pkg/` — which is byte-for-byte what `gh-pages.yml` uploads and
`hf-space.yml` mirrors.

## Layout

```text
playwright.config.ts   # webServer (preflight + serve.mjs on :4174), chromium project
preflight.mjs          # builds/verifies web/pkg/ before the server starts
serve.mjs              # dependency-free static server for web/
e2e/
  tsconfig.json        # strict typecheck for the specs
  specs/demo.spec.ts   # the suite
```

## Why it lives here and not in `web/`

`gh-pages.yml` uploads `path: web` **wholesale**. A `package.json` and
`node_modules/` under `web/` would be published to GitHub Pages along with the
demo. Keeping the harness outside leaves `web/` free of a Node project and needs
no change to either deploy workflow.

## Why `serve.mjs` rather than a static-server package

wasm-pack's `--target web` glue calls `WebAssembly.instantiateStreaming`, which
requires the `.wasm` response to carry `application/wasm`. On any other content
type it falls back to `arrayBuffer()` + `instantiate` and emits a
`console.warn` — *not* an error — so a wrong MIME type would silently degrade
the demo while the "no console errors" spec stayed green. `serve.mjs` sets the
type explicitly, in ~70 lines of Node built-ins, with no dependency to keep
current.

## Running

```bash
npm ci
npm run test:e2e:install     # one-time: download Chromium
npm run test:e2e
```

`webServer` rebuilds `web/pkg/` on every run via `preflight.mjs`. Two escape
hatches, neither set in CI:

- `WEB_E2E_SKIP_BUILD=1` — skip the rebuild and just verify the bundle exists.
  Useful for a fast edit/run loop, or on a host that cannot reach the binaryen
  release `wasm-opt` downloads.
- `WEB_E2E_CHROMIUM_PATH=/path/to/chrome` — use an existing browser instead of
  Playwright's own. Useful in containers that ship one already.

`WEB_E2E_PORT` (default 4174) and `WEB_E2E_HOST` override the server address.

## What the suite proves

| Spec | What would break it |
|---|---|
| loads the module and renders a grid | `init()`, the constructor, or worldgen failing under wasm |
| action-space size | `app.js` drifting from `action_space_json()` |
| Step advances the tick | the step call or the DOM wiring |
| Play advances, Pause stops | the timer loop not driving real wasm steps |
| **same seed reproduces the same world** | CHARTER Invariant 6, checked through the browser and across a page reload |
| **BigInt seed contract** | `reset` taking a `number` again — unreachable from Rust, since the coercion lives only in the generated JS glue |
| readable config error | a panic crossing the boundary as an opaque wasm trap |
| no console or page errors | any Rust panic, which arrives as a `pageerror` |

The demo picks actions with `Math.random()`, so the determinism spec deliberately
compares only the **post-reset** grid, which the seed alone determines. Action-
sequence determinism is covered on the wasm target by
[`crates/forge-wasm/tests/wasm_bindings.rs`](../../crates/forge-wasm/tests/wasm_bindings.rs).
