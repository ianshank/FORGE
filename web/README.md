# FORGE — In-Browser WASM Demo

A fully static, server-free demo that runs the FORGE simulation client-side via
WebAssembly. Published to GitHub Pages by `.github/workflows/gh-pages.yml`.

## What it is

- `index.html` + `app.js` — a minimal UI (reset / step / play-pause) that drives
  the `ForgeWasmEnv` reset/step/render surface from `crates/forge-wasm`.
- `pkg/` — the wasm-pack build output (`forge_wasm.js` + `.wasm`). **Generated,
  git-ignored**; produced by the Pages workflow.

## Build & run locally

```bash
# One-time: wasm32 target (rust-toolchain.toml lists it, so rustup installs it
# for you) + the pinned wasm-pack release (scripts/install_wasm_pack.sh; the
# version is set by the caller -- see CONTRIBUTING.md -- so there's no second
# copy of the pin to drift).
rustup target add wasm32-unknown-unknown
WASM_PACK_VERSION=0.15.0 scripts/install_wasm_pack.sh

# Build the bindings into web/pkg/. Wraps wasm-pack with an absolute --out-dir:
# wasm-pack resolves a relative one against the *crate* directory, so a raw
# `--out-dir web/pkg` would silently emit to crates/forge-wasm/web/pkg.
make wasm            # or: scripts/build_wasm_demo.sh

# Serve statically. ES modules need http://, not file://, and the .wasm must be
# served as `application/wasm` or `WebAssembly.instantiateStreaming` silently
# falls back (a console.warn, not an error) -- a generic static server isn't
# guaranteed to set that, so use the harness's own server:
node tests/web-e2e/serve.mjs
# open http://localhost:4174
```

## Configuring the world

`app.js` passes `DEMO_CONFIG` (an object) to the env constructor. An empty object
makes the Rust side use `ForgeConfig::default()`; override fields there to
customise the demo world — no simulation parameters are hard-coded in the JS.
