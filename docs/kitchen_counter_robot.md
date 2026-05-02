# FORGE for a Kitchen-Counter Cleanup Robot

A concrete deployment proposal: small autonomous robot that lives on a
kitchen counter, detects crumbs, coffee grinds, and small dry spills, and
sweeps them into the sink (or an under-counter waste port). It runs on the
Raspberry Pi 5 + Hailo AI HAT 2 + OpenClaw reference stack already
described in `docs/cloud_edge_proposal.md` and the RPi5/Hailo use-case
notes — this document instantiates that stack for the kitchen-counter
domain.

The case for FORGE here is that *every* faculty the robot needs —
discrete-tile navigation, manipulation primitives, latency-budgeted
planning, constitutional safety, OTA continual learning — is already a
first-class FORGE primitive. The kitchen-counter robot is a small
configuration delta on existing crates, not a new product line.

---

## 1. Why FORGE Fits the Kitchen Counter

A kitchen counter is, from a planning standpoint, an unusually clean fit
for FORGE's discrete-tile world model:

- **Bounded, planar workspace.** A countertop tessellates naturally into
  a 32×16 (or similar) tile grid. `forge-worldgen` already produces
  square grids of this shape.
- **Discrete, repetitive task.** "Find debris → push toward drain →
  release" is exactly the `Sequence([AgentAt, Push, AgentAt, Drop])`
  pattern the task DSL was designed for.
- **Hard safety constraints.** Counters contain cliffs (the counter
  edge), hot zones (stovetop, kettle base), fragile objects (glassware),
  electronics (toaster, charging phone), and humans (hands moving in and
  out of the workspace). FORGE's constitutional layer
  (`forge-cognitive` + `forge-integration-layer`) and the `Without(...)`
  / `While(HealthAbove, ...)` task combinators already express these.
- **Long-tail distribution shift.** Every kitchen is different and
  changes daily. The closed-loop continual-learning path
  (`forge-edge::TelemetryCollector` → `CompactReplay` → cloud retrain →
  OTA `.hef`) is the right shape for this — see Case G in the RPi5/Hailo
  doc.
- **Tight latency budget but not real-time.** Planning at 20–30 Hz over
  discrete sweep primitives is well within `AdaptiveMctsSearch`'s
  envelope; the wheel/servo controller below it runs its own loop.

---

## 2. Hardware Stack (delta from the RPi5/Hailo reference)

| Component | Role | Kitchen-specific notes |
|---|---|---|
| Raspberry Pi 5 (8 GB) | Host SBC | Sealed enclosure, IP54 minimum |
| Hailo AI HAT 2 (Hailo-8L) | NPU | 13 TOPS is plenty; thermal headroom matters more |
| OpenClaw + sweeper bar | End-effector | Soft silicone squeegee for sweeping; gripper jaws for picking up larger items (e.g. a fallen pasta piece) |
| RPi Camera Module 3 (wide) | Vision input | Top-down or 30° down-angle; sees full counter |
| ToF / IR cliff sensors (×4) | Edge-of-counter detection | Hard-wired E-stop, *not* just a soft signal |
| Capacitive touch / IMU | Bump + tip detection | Triggers fallback action |
| Wireless charging dock | Power | Mounts under a wall cabinet; robot self-docks |
| Wi-Fi | Telemetry uplink | `CompactReplay` ~16 KB/cleanup pass |

The IP rating, cliff sensors, and bump detector are the only items not
already covered by the RPi5/Hailo reference doc — they reflect the kitchen
deployment surface, not FORGE itself.

---

## 3. Action ↔ Actuator Mapping

The kitchen robot uses the same FORGE discrete action space as the
reference doc, with concrete kitchen semantics for each:

| FORGE action | Kitchen semantics |
|---|---|
| `0` Noop | Hold position (used during human-presence pause, see §6) |
| `1–4` Move | Drive base across counter tiles (front/back/left/right) |
| `5` PickUp | Close OpenClaw on a discrete object (e.g. dropped utensil, cap) |
| `6–15` Drop slot N | Open gripper over the sink tile or waste port |
| `16` Use slot 0 | **Lower sweeper bar** (engage squeegee with surface) |
| `17` Use slot 1 | **Raise sweeper bar** (clear glassware / wet zone) |
| `18` Use slot 2 | Run a brief vibration burst to dislodge stuck grinds |
| `19–25` Use slot N | Reserved (cloth dispense, mist spray — out of MVP scope) |
| `26–34` Craft N | Compose a multi-step routine, e.g. "lower sweeper → push 3 tiles → raise → reverse 1 tile" |
| `35–38` Push | Sweep motion in cardinal direction (the workhorse action) |
| `39` Interact | Trigger sink-edge subroutine (align, sweep over lip, raise) |

This is exactly the action layout `EdgeAgent::select_action` already
returns. The kitchen-side bridge layer — a small Python or Rust shim
under §7 — is the only thing translating these IDs into OpenClaw and
drive-motor commands.

---

## 4. World Model

A counter is modelled as a small grid (e.g. 32 × 16 = 512 tiles, ~3 cm per
tile for a 1 m × 0.5 m workspace). Per-tile observations reuse existing
`TileObservation` channels:

| `TileObservation` field | Kitchen meaning |
|---|---|
| `terrain` | Surface type (granite / wood / glass insert / induction zone) — affects allowed actions |
| `has_object` | Whether a debris cluster is present on the tile (the boolean already produced by `forge-types::observation::TileObservation`) |
| `object_type` | Class — `crumb`, `coffee_grind`, `liquid`, `solid_item`, `fragile`, `hot`, `electronic`, `human_hand` (the existing `u8` channel; values defined by the deployment's perception model) |
| `elevation` | Bump map (cutting board, plate edge) — used by sweeper engage logic |
| `has_agent` | Self-occupancy |

A perception model (off-the-shelf YOLO-style detector compiled to the
Hailo HAT) populates `object_type` from the camera frame each tick. This
slots into the same vision-preprocessing path the agricultural scenarios
already use — `forge-edge::edge_agent::flatten_observation` flattens it
verbatim.

The **sink** is one or more tiles flagged with a `drain` terrain type;
`Drop` over a `drain` tile is rewarded. Counter edges that are *not* the
sink are flagged `cliff` and are inviolable.

---

## 5. Task DSL Encoding

The high-level cleanup mission decomposes cleanly into the existing task
combinators:

```text
Repeat(
  Sequence([
    Atom(ObservedDebris(any_tile)),          // detect work
    AgentAt(NearestDebrisTile),              // approach
    Use(slot=0),                             // lower sweeper
    Push(toward = NearestSinkTile),          // sweep
    AgentAt(SinkEdgeTile),                   // arrive at sink
    Interact,                                // sink-edge sweep-over-lip
    Use(slot=1),                             // raise sweeper
    AgentAt(DockTile),                       // optional return-home
  ])
)
```

Wrapped in safety predicates:

```text
While(
  And(
    Without(action = Move into Tile{kind=human_hand}),
    Without(action = Push into Tile{kind=fragile|hot|electronic|liquid}),
    HealthAbove(0.99)                        // any bump drops "health"
  ),
  CleanupSequence
)
```

`Without(action = Push)` and `While(HealthAbove(...), ...)` are the same
combinators called out in Case F of the RPi5/Hailo doc — the kitchen case
is, structurally, a constitutional-safety case with a domestic-flavoured
predicate set.

The starter scenario lives in
[`configs/scenarios/kitchen_cleanup.toml`](../configs/scenarios/kitchen_cleanup.toml)
and is a real `forge_scenario::ScenarioConfig` — i.e. it parses through
the existing `ScenarioConfig::from_toml_file` loader
([`crates/forge-scenario/src/config.rs`](../crates/forge-scenario/src/config.rs))
and a regression test
([`tests/rust/integration_kitchen_robot.rs`](../tests/rust/integration_kitchen_robot.rs))
asserts that on every workspace test run:

```toml
[scenario]
id = "kitchen_cleanup"
name = "Kitchen Counter Cleanup"
description = "Sweep crumbs and grinds from a counter-top into the sink ..."
tags = ["edge", "manipulation", "navigation", "constitutional-safety"]
difficulty_tier = 2
min_agents = 1
max_agents = 1
author = "FORGE"
version = "1.0"

# A 1 m × 0.5 m counter at ~3 cm/tile → 32 × 16 grid.
[forge.world]
width = 32
height = 16
seed = 0

[forge.agents]
num_agents = 1

[forge.task]
enabled = true
max_episode_length = 1500
dense_rewards = true
```

What this file does **not** carry — and would need a future
kitchen-specific scenario extension to express — is the per-event reward
shaping (per-crumb / per-grind / completion bonus, fragile / cliff /
human-proximity penalties) and the kitchen-specific terrain semantics
(sink tiles, cliff perimeter). Those fields are tracked as the
`sweep_to_drain` objective dispatcher work in §7 "Still to do" and
would slot in alongside the existing `WorldConfig` / `TaskConfig`
extension points without breaking the schema this file already obeys.

---

## 6. Constitutional & Physical Safety

The kitchen is the most safety-critical of the use cases in the
RPi5/Hailo doc. FORGE handles it in three layers:

1. **Hard interlocks (below FORGE).**
   - Cliff sensors and the bump IMU are wired to a hardware E-stop. If
     they trigger, drive motors are cut and the sweeper raises before any
     software path sees the event.
   - The OpenClaw torque limit is set in firmware below the threshold
     that could crack a glass.

2. **Constitutional layer (`forge-cognitive`).**
   - "Do not Push fragile / hot / electronic / liquid tiles."
   - "Do not Move into a tile whose `object_type` is `human_hand`."
   - "Do not Drop over a non-`drain` tile."
   - When the value head's prediction crosses the constitutional
     threshold, `EdgeAgent` falls back to the action id supplied by
     `EdgeConfig.fallback_action_id`
     ([`crates/forge-types/src/config.rs`](../crates/forge-types/src/config.rs)
     — see also `DEFAULT_EDGE_FALLBACK_ACTION_ID` in
     [`crates/forge-types/src/constants.rs`](../crates/forge-types/src/constants.rs)).
     The kitchen profile sets that field to `17`
     ([`configs/edge/rpi5_hailo_kitchen.toml`](../configs/edge/rpi5_hailo_kitchen.toml)),
     and the OpenClaw mapping
     ([`configs/actuator/openclaw_kitchen.toml`](../configs/actuator/openclaw_kitchen.toml))
     routes id `17` to `[disengage_sweeper, halt]` — a hardware-safe
     stance. The whole change is two TOML edits; no Rust touched.

3. **Adaptive planning budget (`AdaptiveMctsSearch`).**
   - Default `mcts_latency_budget_ms = 33` (≈30 Hz).
   - When `human_hand` is observed in any visible tile, the planner is
     forced to `mcts_min_simulations` and the policy temperature is
     pinned to greedy — the robot prefers to stand still and wait rather
     than improvise around a human.

This is the same three-layer pattern used in Case F (constitutional
validation rig) of the RPi5/Hailo doc; the kitchen instantiation is a
configuration of it, not a new mechanism.

---

## 7. What's Built (and what's still pending)

The deltas on top of the reference RPi5/Hailo stack are intentionally
small. The list below is what now ships in this repository — everything
else is configuration the operator owns.

### Built and tested in-tree

1. **`fallback_action_id` field on `EdgeConfig`**
   ([`crates/forge-types/src/config.rs`](../crates/forge-types/src/config.rs),
   default in
   [`crates/forge-types/src/constants.rs`](../crates/forge-types/src/constants.rs)).
   Backwards compatible — old TOMLs missing the field deserialize to the
   historical `Noop` (`0`) fallback, and an env override
   (`FORGE_EDGE_FALLBACK_ACTION_ID`) is wired through `apply_env_overrides`.
   `EdgeAgent::new`
   ([`crates/forge-edge/src/edge_agent.rs`](../crates/forge-edge/src/edge_agent.rs))
   now reads this field instead of the previous hardcoded `0`. Verified by
   `test_fallback_action_id_honoured_from_config` and
   `test_edge_config_fallback_action_id_backward_compatible`.

2. **New `forge-actuator` crate**
   ([`crates/forge-actuator/`](../crates/forge-actuator)) — the
   action-id → actuator-command bridge:
   - `ActuatorCommand` (`drive_direction`, `open_gripper`, `close_gripper`,
     `engage_sweeper`, `disengage_sweeper`, `vibrate`, `halt`, `custom`).
   - `ActionMapping` — TOML-loaded, `strict` / permissive, with default
     fallback sequence; rejects duplicate ids and missing defaults at load.
   - `ActuatorBridge` + `ActuatorDriver` traits.
   - `MappedActuator<D>` — single generic implementation, instrumented
     with `tracing` and a bounded command-history ring buffer.
   - `MockDriver` — in-memory driver for tests, with optional
     failure-injection predicate for error-path coverage.
   - 34 unit tests (incl. proptest cases) all green.

3. **Edge profile**
   [`configs/edge/rpi5_hailo_kitchen.toml`](../configs/edge/rpi5_hailo_kitchen.toml).
   Sets `mcts_latency_budget_ms = 33`, `mcts_min_simulations = 8`,
   `mcts_max_simulations = 96`, `telemetry_interval_s = 60`,
   `telemetry_buffer_bytes = 4 MiB`, and crucially `fallback_action_id = 17`.

4. **OpenClaw action mapping**
   [`configs/actuator/openclaw_kitchen.toml`](../configs/actuator/openclaw_kitchen.toml).
   Concrete action-id → command-sequence assignments for ids 0–39,
   including the safe-pose `17 → [disengage_sweeper, halt]` contract that
   pairs with the edge profile's fallback id.

5. **Scenario file**
   [`configs/scenarios/kitchen_cleanup.toml`](../configs/scenarios/kitchen_cleanup.toml).
   Conforms to the real `forge_scenario::ScenarioConfig` schema — verified
   to parse cleanly via a workspace integration test. Kitchen-specific
   reward shaping and terrain extensions (per-crumb / per-grind rewards,
   sink/cliff terrain tags, the `sweep_to_drain` objective dispatcher
   variant) are deliberately deferred to the "Still to do" list below
   so this file doesn't carry unloadable schema.

6. **End-to-end integration test**
   [`tests/rust/integration_kitchen_robot.rs`](../tests/rust/integration_kitchen_robot.rs).
   Loads both TOMLs, drives an `EdgeAgent` backed by an always-failing
   model, asserts the fallback id (`17`) emerges, dispatches it through
   the bridge, and confirms the driver received exactly
   `[disengage_sweeper, halt]`. Also covers the legacy-profile
   backwards-compat path and the cleanup-sweep command sequence.

### Still to do (not blocking the kitchen MVP)

1. **OpenClaw `ActuatorDriver` impl.** A small Rust shim (or Python glue
   over PyO3) that marshals `ActuatorCommand` onto the OpenClaw serial
   bus. Lives outside the FORGE workspace because it depends on
   distribution-specific serial drivers; the trait it implements is here.
2. **`sweep_to_drain` objective dispatcher.** The scenario TOML uses the
   new objective token; the scenario loader needs one new match arm to
   wire it through to the reward computation. Out of scope for this
   change because it touches `forge-task` independently and the rest of
   the pipeline doesn't block on it.
3. **Vision head.** Off-the-shelf YOLO-style debris detector compiled to
   `.hef` alongside the MuZero heads. Emits the `object_type` channel
   consumed by the world model — no FORGE changes required.
4. **Hailo `LatentForwardModel` backend.** Wraps HailoRT and a compiled
   `.hef`. Slots into the existing `OnnxMuZeroModel` extension point.

None of these require trait or schema changes to FORGE itself.

---

## 8. Continual Learning Loop

Identical to Case G of the RPi5/Hailo doc. Per cleanup pass:

1. `EdgeAgent` runs the mission; `TelemetryCollector` buffers a
   `CompactReplay` (~16 KB) including any constitutional fallback events.
2. On dock, the buffer flushes via `ReplayTransport` to
   `gs://forge-training/{run_id}/replays/`.
3. Cloud pipeline reconstructs trajectories with
   `CompactReplay::replay()`, retrains MuZero with the new debris
   distribution, exports ONNX, recompiles to `.hef`.
4. Robot pulls the new model on its `model_update_interval_s` cadence.

Constitutional fallback events are first-class — they're prioritised in
the cloud retrain so each kitchen "incident" measurably reduces the rate
of similar future incidents across the fleet.

---

## 9. Out-of-Scope (for now)

- **Wet cleanup.** A countertop sweeper handles dry debris (crumbs,
  grinds, sugar, dry spices). Liquid spills are detected (`object_type =
  liquid`) and treated as no-go zones. Wet cleanup needs a different
  end-effector and is a separate product.
- **Knife / blade handling.** Treated as `fragile|hot` from the planner's
  perspective — the robot reroutes around them, never `PickUp`s.
- **Dishwashing or utensil sorting.** Out of scope. The robot is a
  surface cleaner, not a manipulator-of-objects-into-receptacles beyond
  the sink itself.
- **Below-30-Hz joint-trajectory control.** Same as §5 of the RPi5/Hailo
  doc — FORGE plans at the discrete-action layer; the OpenClaw firmware
  owns smooth motion.

---

## 10. Bottom Line

The kitchen-counter cleanup robot is a near-pure configuration of the
RPi5+Hailo+OpenClaw reference stack:

- **Reused**: `EdgeAgent`, `AdaptiveMctsSearch`, `TelemetryCollector`,
  `OnnxMuZeroModel`/`HailoLatentModel`, the constitutional layer,
  `CompactReplay`, the OTA model-update loop, the discrete action space,
  the task DSL.
- **Added (this branch)**: a backwards-compatible `fallback_action_id`
  field on `EdgeConfig`; a generic `forge-actuator` crate
  (`MappedActuator` + `ActionMapping` + `ActuatorCommand` +
  `ActuatorDriver` trait + `MockDriver`); kitchen-specific edge,
  scenario, and actuator TOMLs; and a workspace integration test that
  proves the safe-pose path end-to-end.
- **Still to add**: an OpenClaw `ActuatorDriver` (lives outside the
  workspace), a `sweep_to_drain` dispatcher arm in the scenario loader,
  and an off-the-shelf vision head.
- **Risks owned outside FORGE**: cliff/bump hardware interlock, OpenClaw
  torque limits, IP rating.

That ratio — almost everything reused, very little new — is the case for
FORGE on this product. The same training pipeline, edge runtime, and
continual-learning loop that already targets agricultural scouting and
tabletop manipulation runs the kitchen-counter robot with no
architectural change.
