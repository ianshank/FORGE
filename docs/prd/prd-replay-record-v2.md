# PRD — E11: Replay / Record Mode (updated)

**Epic Slug**: `replay-record`
**Priority**: P2
**Sprint**: 6
**Size**: L
**Depends on**: `forge_env.wrappers` (existing), Demo UI frontend

> **Note**: Original PRD exists at `docs/prd/prd-replay-record.md`. This file supersedes it with updated implementation notes following Sprint 5 architecture decisions.

---

## User Story

> **As a** FORGE user running agent experiments,
> **I want to** record a full episode as a `.forge` replay file and replay it in the browser or terminal at configurable speed,
> **So that** I can reproduce bugs, share demonstrations, and embed animated GIFs in papers.

---

## Problem Statement

FORGE is deterministic given seed + action sequence, but there is no first-class mechanism to persist that sequence. A Record/Replay feature with a defined `.forge` file format unlocks reproducibility, debugging, and presentation.

---

## `.forge` File Format (v1)

```json
{
  "forge_version": "0.2.0",
  "format_version": 1,
  "seed": 42,
  "config": { "world": { "width": 32, "height": 32 }, "agents": { "num_agents": 2 } },
  "actions": [1, 2, 0, 3, ...],
  "observations": [[...], [...], ...],
  "rewards": [0.1, -0.5, ...],
  "terminated_at": 487,
  "timestamps_ms": [0, 16, 33, ...]
}
```

**Design decisions:**

- Observations stored alongside actions (larger file, no re-run needed for replay)
- `observations` field is *optional* — replay can reconstruct from env if omitted
- `timestamps_ms` for pace-accurate GIF export

---

## Acceptance Criteria

| # | Given | When | Then |
|---|---|---|---|
| AC1 | User wraps env with `RecordEpisodeWrapper(env, output_path="out.forge")` | Episode ends (terminated or truncated) | `out.forge` is written with valid JSON matching the schema |
| AC2 | Valid `.forge` file exists | `python -m forge_env.replay out.forge` | World renders step-by-step in terminal at `--fps` rate (default 10) |
| AC3 | Demo UI is open | User clicks "📼 Load Replay" and selects `.forge` file | Canvas plays back episode; step counter and agent rewards update live |
| AC4 | Replay is playing in the UI | User clicks "Export GIF" | A `.gif` file downloads containing the world canvas animation (≤2000 frames) |
| AC5 | `.forge` file from older FORGE version | File `forge_version` differs | Warning is emitted but replay proceeds |
| AC6 | Replay episode has 10,000 steps | File is loaded | Memory stays ≤ 200 MB; playback start time ≤ 2s |
| AC7 | `RecordEpisodeWrapper` is active | Step is called | Per-step overhead ≤ 1% vs unwrapped env (benchmarked) |

---

## Out of Scope

- MP4 / WebM export
- Network-synchronized multi-user replay
- `.forge` file compression (post-beta)

---

## Success Metrics

- `RecordEpisodeWrapper` passes benchmark gate (see AC7)
- GIF export produces valid `.gif` for episodes ≤ 2000 frames
- All 7 ACs covered by automated tests

---

## Open Questions

1. Should `observations` default to stored or reconstructed? Recommend **stored** (simpler, reproducible).
2. GIF encoding: `Pillow` (Python CLI) vs `canvas + gif.js` (browser)? Recommend both paths.
3. Multi-agent: record per-agent actions as `{ "agent_0": [1,2,...], "agent_1": [0,3,...] }`?

---

## Implementation Notes

### Python (`python/forge_env/`)

- `wrappers.py`: `RecordEpisodeWrapper(env, output_path)` — stores `seed`, `config`, `actions[]`, `observations[]`, `rewards[]`
- `replay.py` *(new)*: `load_replay(path) -> ReplayData`, `play_replay(data, fps, stream)`, `export_gif(data, out_path)` (via `Pillow`)

### Demo UI (`demo_ui/frontend/`)

- `app.js`: "📼 Load Replay" `<input type="file" accept=".forge">` with `FileReader` → JSON parse → frame loop
- `styles.css`: replay progress bar, GIF export button styles

### Tests

- `tests/python/test_replay.py`: unit tests for `RecordEpisodeWrapper` and `load_replay/play_replay`
- `demo_ui/tests/aqa-replay-record.spec.ts` (Playwright): AC2–AC5 E2E coverage

### CLI entry-point

Add to `pyproject.toml` `[project.scripts]`:

```toml
forge-replay = "forge_env.replay:_cli"
```
