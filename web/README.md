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
# One-time: wasm target + wasm-pack
rustup target add wasm32-unknown-unknown
cargo install wasm-pack

# Build the bindings into web/pkg/
wasm-pack build crates/forge-wasm --target web --out-dir "$PWD/web/pkg" --no-typescript

# Serve statically (any static server works; ES modules need http://, not file://)
python3 -m http.server -d web 8000
# open http://localhost:8000
```

## Configuring the world

`app.js` passes `DEMO_CONFIG` (an object) to the env constructor. An empty object
makes the Rust side use `ForgeConfig::default()`; override fields there to
customise the demo world — no simulation parameters are hard-coded in the JS.
