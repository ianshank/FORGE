# CompactReplay golden corpus

Bit-identity fixtures for `forge-replay` format version 2. The PR CI
`cargo test --workspace` job loads these JSON files and fails on any byte
drift (config hash, actions, frozen timestamp).

## Regenerating

Only after an intentional format change:

```bash
UPDATE_GOLDEN_REPLAYS=1 cargo test -p forge-replay --test golden_replay
```

Then append a row to [`docs/results/replay_flip_log.md`](../../docs/results/replay_flip_log.md)
with the new `config_hash` and the reason. Do not silently refresh goldens
to paper over a determinism bug.

The timestamp on every golden CompactReplay is pinned to
`1970-01-01T00:00:00+00:00` (`GOLDEN_TIMESTAMP`) so wall-clock recording
cannot flake the gate.
