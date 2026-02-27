# PRD — E11: Replay / Record Mode

**Epic Slug**: `replay-record`  
**Priority**: P2  
**Sprint**: 6  
**Size**: L

---

## User Story

> **As a** FORGE user running agent experiments,  
> **I want to** record a full episode as a JSON replay file and replay it later at configurable speed,  
> **So that** I can reproduce bugs, create shareable demonstrations, and export animated GIFs for papers and presentations.

---

## Problem Statement

Currently FORGE is deterministic given seed + action sequence, but there is no built-in mechanism to persist that sequence. Users manually log actions in their training loops. A first-class Record/Replay feature unlocks reproducibility, debugging, and presentation capabilities.

---

## Acceptance Criteria

| # | Given | When | Then |
|---|---|---|---|
| AC1 | User wraps env with `RecordEpisodeWrapper(env, output_path="replay.json")` | Episode ends | `replay.json` exists and contains `{seed, config, actions, observations}` |
| AC2 | User loads a replay file | `python -m forge_env.replay replay.json` | ASI grid is rendered step-by-step in terminal, rate-limited to `--fps` |
| AC3 | Demo UI has a "Load Replay" button | User selects a replay JSON | World canvas plays back the episode at configurable speed |
| AC4 | User clicks "Export GIF" in demo UI | After replay completes | A `.gif` file is downloaded containing the world canvas animation |
| AC5 | Replay file is loaded from a different version of FORGE | File has `"forge_version"` field | A warning is printed if versions differ, but replay still proceeds |
| AC6 | Replay file contains 10,000 actions | Replay plays at 60fps | Memory usage stays below 200 MB |

---

## Out of Scope

- Video (MP4) export (post-beta)
- Network-synchronized replays (multi-agent live replay)
- Replay file compression (post-beta)

---

## Success Metrics

- `RecordEpisodeWrapper` adds < 1% overhead per step
- Replay JSON is human-readable (pretty-printed)
- GIF export works for episodes up to 2,000 frames
- All ACs covered by automated tests

---

## Open Questions

1. Should replay JSON store full observations or just seed + actions?
2. For multi-agent: how do we handle concurrent actions in the replay format?
3. GIF export: use `Pillow` (Python) or `canvas → MediaRecorder` (browser)?

---

## Implementation Notes

- `python/forge_env/wrappers.py`: add `RecordEpisodeWrapper` (stores `seed`, `actions[]`, `obs[]`)
- `python/forge_env/replay.py`: `load_replay(path) -> ReplayData`, `play_replay(data, fps=10)`
- Demo UI: add "📼 Load Replay" button to `SectionNav`; progress bar driven by frame index
- GIF export: `canvas.toDataURL()` frames collected via `requestAnimationFrame`, encoded via `gif.js`
- Schema: `{"forge_version": "0.2.0", "seed": 42, "config": {...}, "actions": [1,2,3,...], "timestamps_ms": [...]}`
