# PRD — E5/E6: WASM Live Demo (GitHub Pages)

**Epic Slug**: `wasm-live-demo`  
**Priority**: P0  
**Sprint**: 3  
**Size**: L

---

## User Story

> **As a** researcher or potential adopter browsing the FORGE repository,  
> **I want to** interact with a live FORGE simulation running in my browser without installing anything,  
> **So that** I can immediately evaluate whether FORGE meets my use case.

---

## Problem Statement

The current demo requires running a local Python server. This creates friction for public-facing demos, conference presentations, and repository sharing. The `forge-wasm` crate already exists — deploying it to GitHub Pages enables a zero-installation shareable demo.

---

## Acceptance Criteria

| # | Given | When | Then |
|---|---|---|---|
| AC1 | A visitor navigates to the GH Pages URL | Page loads | The FORGE demo page loads in < 3 seconds (cached WASM) |
| AC2 | User clicks "Reset World" with seed 42 | After WASM loads | ASCII grid renders correctly matching Python reference output |
| AC3 | User clicks "Step" 10 times | Each click | Grid updates, step count increments, no JS console errors |
| AC4 | User modifies the URL hash to `#seed=99` | Page loads | World is initialized with seed 99 |
| AC5 | WASM module throws a panic | During execution | Error is caught, shown in UI, does not crash the page |
| AC6 | A new release is tagged | Via CI | GH Pages is automatically rebuilt and deployed |

---

## Out of Scope

- Full 8-section animated demo in WASM (SSE-based demo UI remains separate)
- Training an agent in-browser (too slow without SIMD/threads)
- Mobile-optimized layout (desktop-first for beta)

---

## Success Metrics

- WASM bundle size < 5 MB (post `wasm-opt`)
- Page load to interactive: < 3 seconds on broadband
- Zero console errors during normal operation
- Successfully deployed to `https://ianshank.github.io/FORGE`

---

## Open Questions

1. Should the GH Pages demo be a separate `docs/` folder or the current `demo_ui/frontend/`?
2. Do we need `wasm-bindgen-rayon` for parallel WASM, or is single-threaded sufficient for demo?
3. How do we handle the JSON I/O overhead? Profile before deciding on binary encoding.

---

## Implementation Notes

- Add `wasm.yml` CI workflow: `wasm-pack build crates/forge-wasm --target web`
- Run `wasm-opt -O3` on output
- Deploy via `peaceiris/actions-gh-pages` action
- New static page: `docs/demo/index.html` with embedded WASM calls replacing SSE
- URL hash routing: `window.location.hash` → seed parameter
