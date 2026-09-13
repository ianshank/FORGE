# CompactReplay golden flip log

Record every intentional change to `tests/golden/replays/`. Unexplained
mismatches in `cargo test -p forge-replay --test golden_replay` are bugs,
not documentation updates.

| Date | Format | Config hash (sha256 hex) | Reason |
| --- | --- | --- | --- |
| 2026-09-12 | 2 | `a74fee1d5eaea7910e1b41309f2500ded8bf9c744b540a270c225adce165c0f0` | Initial v2 corpus: portable SHA-256 config hash, seed applied before `WorldState::new`, unknown action ids hard-error |
| 2026-09-12 | 2 | `e831b2c14031f0d10d4ebd3fff8259ec66bcc0727335d183c5b994f73eb27071` | Workstream E: `WorldConfig` geofence knobs + `DroneConfig` charger/spawn_home/battery_action_floor (defaults off; physics unchanged) |
