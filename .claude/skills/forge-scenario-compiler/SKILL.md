---
name: forge-scenario-compiler
description: Compile a high-level FORGE `[scenario]` TOML into ForgeConfig plus scenario_tasks, wire orchard coverage / agri-drone graders, and know when CompactReplay goldens must be regenerated. Use when adding or editing configs/scenarios/*.toml, charger/geofence/drone knobs, forge-eval task success, or the dual Rust/Python compiler.
---

High-level `configs/scenarios/*.toml` uses a `[scenario]` schema (map, drone,
agri, objectives) that is **not** `ForgeConfig`. Two compilers produce the
same `scenario_tasks` artifact:

1. Rust: `crates/forge-types/src/scenario.rs`
   (`compile_high_level_scenario` / `compile_high_level_path`). `forge-eval`
   `Scenario::load_file` falls back to this compiler.
2. Python: `python/forge/mangomas/collector/scenario.py`
   (`resolve_forge_scenarios`). MangoMAS / Gymnasium collection uses this
   path.

Do **not** fold this into `forge-skills-catalog` (that skill is in-engine
HRL options over `Action`s). Do **not** package Anthropic Agent Skills
until eval-gated HRL winners exist (CHARTER).

## Source of truth

- Defaults live in `crates/forge-types/src/constants.rs`
  (`DEFAULT_SURVEY_THRESHOLD`, `DEFAULT_COVERAGE_BATTERY_THRESHOLD`,
  `DEFAULT_SPAWN_HOME_*`, `DEFAULT_SCENARIO_NUM_AERIAL`). Python mirrors
  them as `_DEFAULT_*` in `scenario.py`; pin both in
  `tests/python/test_scenario_compiler_xlang_pin.py` and Rust
  `xlang_*_pinned_to_known_good`.
- Objective mapping:
  - `survey` → `FieldSurveyed`
  - `collect` → `SoilDataCollected`
  - `coverage` / `orchard` → `And([FieldSurveyed, BatteryAbove, AgentAt(home)])`
  - `sequence` steps `survey` / `spray` / `relay_soil` / `generate_report`
  - unmapped kinds (`patrol`, `escort`, SAR) → empty `scenario_tasks`
- Process constraints (config-driven `Noop`s, same-seed replay):
  geofence, charger-gated recharge, battery-action floor, ascend-at-cap.
  Knobs: `DroneConfig::{restrict_recharge_to_chargers, charger_tiles,
  spawn_home, battery_action_floor}` and
  `WorldConfig::{geofence_enabled, geofence_margin}`. All `Default` +
  `serde(default)`.
- Eval success (`forge-eval` harness): controlled agent alive **and** every
  attached task completed **and** not truncated. `terminated` / `truncated`
  stay separate fields. Empty `tasks` is vacuously complete.

## Adding or editing a scenario (checklist)

1. Edit `configs/scenarios/<name>.toml` with `[scenario]` tables only.
   Extra keys (waypoints, fog) are ignored; do not copy them onto
   `ForgeConfig` (`deny_unknown_fields`).
2. Keep Rust and Python compilers in lockstep. After changing mapping,
   run:
   - `cargo test -p forge-types --lib scenario`
   - `pytest tests/python/test_scenario_compiler_xlang_pin.py -q --no-cov`
3. If you add `ForgeConfig` / `DroneConfig` / `WorldConfig` fields, they
   change CompactReplay `config_hash`. In a **separate** cargo invocation:
   `UPDATE_GOLDEN_REPLAYS=1 cargo test -p forge-replay --test golden_replay`
   then append a row to `docs/results/replay_flip_log.md`.
4. Attach tasks on world init only when `task.enabled`. Agri-off + empty
   tasks stay on the zero-alloc hot path.
5. Logging: `debug!`/`trace!` on process-constraint Noops; do not
   `println!`.

## Orchard coverage grader

- Scenario: `configs/scenarios/orchard_coverage.toml` (16×16, geofence
  margin 1, home/charger `(0,0)`, restrict recharge).
- Lawnmower: `crates/forge-core/src/baselines.rs` and
  `python/forge/baselines/coverage.py` (import `FORGE_BASE_ACTIONS` /
  `FORGE_DRONE_ACTION_COUNT` from `python/forge/actions.py`; do not
  re-literal 40/19).
- SAC hook: `configs/training/sac_orchard.toml`,
  `examples/train_sac_cleanrl.py --scenario`,
  `examples/run_orchard_coverage_baselines.py`.

## Out of scope

- Unifying the two compilers onto PyO3 in this skill (larger refactor).
- Faking Minecraft `evidential_episodes >= 3`. Use
  `scripts/mc_evidential_capture.sh --dry-run` in CI; live capture needs
  Docker.
- Putting OpenEnv on the PyO3 hot path. Sidecar:
  `python/forge_env/openenv_env.py`.
- Adding `check_golden_replays.sh` to `make verify` (workspace `cargo test`
  already runs `golden_replay`).
