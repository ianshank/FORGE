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
| `has_object` | Vision-detected debris density (binary or quantised) |
| `object_kind` | Class — `crumb`, `coffee_grind`, `liquid`, `solid_item`, `fragile`, `hot`, `electronic`, `human_hand` |
| `height` | Bump map (cutting board, plate edge) — used by sweeper engage logic |
| `agent_id` | Self-occupancy |

A perception model (off-the-shelf YOLO-style detector compiled to the
Hailo HAT) populates `object_kind` from the camera frame each tick. This
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

A starter scenario file (`configs/scenarios/kitchen_cleanup.toml`) would
follow the shape of the existing `patrol.toml` / `escort.toml`:

```toml
[scenario]
name = "kitchen_cleanup"
description = "Sweep crumbs and grinds from a countertop into the sink"
min_agents = 1
max_agents = 1

[scenario.map]
grid_size_x = 32
grid_size_y = 16
terrain_type = "kitchen_counter"
sink_tiles = 4
cliff_tiles = "perimeter_minus_sink"

[scenario.objectives]
type = "sweep_to_drain"
time_limit = 1500
reward_per_grind_drained = 0.05
reward_per_crumb_drained = 0.02
completion_bonus = 5.0
penalty_per_fragile_contact = -10.0
penalty_per_cliff_event = -25.0
penalty_per_human_proximity = -2.0

[scenario.difficulty]
base_tier = 2
fog_of_war = false               # top-down camera sees everything
distractor_objects = true        # stray utensils, mugs, phones
human_hand_events = true         # simulated reach-ins
```

Nothing here requires schema changes — `grid_size_x/y` and the new
terrain enum values map onto existing `forge-types::config` extension
points. The new objective type slots into the existing reward-builder
the same way `escort` and `patrol` do.

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
   - "Do not Move into a tile occupied by a human_hand object_kind."
   - "Do not Drop over a non-`drain` tile."
   - When the value head's prediction crosses the constitutional
     threshold, `EdgeAgent` already falls back to its configured fallback
     action. For the kitchen we override the default `Noop` fallback to
     "raise sweeper + halt" via the existing fallback-action knob in
     `EdgeAgent` (this is one config field, not a code change).

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

## 7. What Has to Be Built

Mirroring §4 of the RPi5/Hailo doc, the kitchen-counter robot adds these
deltas on top of the reference stack:

1. **`configs/scenarios/kitchen_cleanup.toml`** — the scenario sketched
   in §5. Pure config.
2. **`configs/edge/rpi5_hailo_kitchen.toml`** — `EdgeConfig` profile:
   - `mcts_latency_budget_ms = 33`
   - `mcts_min_simulations = 8` (used during human-proximity pauses)
   - `mcts_max_simulations = 96`
   - `telemetry_interval_s = 60` (one flush per typical cleanup pass)
   - Custom fallback action ID for "raise sweeper + halt".
3. **OpenClaw kitchen bridge** — the §7.2 shim from the RPi5/Hailo doc,
   with one extra responsibility: route action `16/17` to the sweeper
   servo and recognise the `drain` terrain tag for `Drop`. ~200 LOC of
   Python or a small Rust crate; lives outside the FORGE workspace or in
   a `forge-actuator-kitchen` crate if we want it in-tree.
4. **Vision head** — a YOLO-style debris detector compiled to `.hef`
   alongside the MuZero heads. Off-the-shelf model, fine-tuned on a few
   thousand kitchen frames; emits the `object_kind` channel consumed by
   the world model. No FORGE changes required — it's just another
   producer for `TileObservation`.
5. **Reward shaping** — the new objective type `sweep_to_drain` plus the
   penalty terms in §5. Slots into the same reward-builder pattern used
   by the existing scenarios.

No trait or schema changes. The Hailo backend (`HailoLatentModel`) and
the OpenClaw bridge described in the RPi5/Hailo doc §4 are reused
verbatim.

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
  grinds, sugar, dry spices). Liquid spills are detected (`object_kind =
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
- **Added**: one scenario TOML, one edge profile TOML, a ~200-LOC
  actuator bridge with sweeper + drain awareness, an off-the-shelf
  vision head, and IP54 packaging.
- **Risks owned outside FORGE**: cliff/bump hardware interlock, OpenClaw
  torque limits, IP rating.

That ratio — almost everything reused, very little new — is the case for
FORGE on this product. The same training pipeline, edge runtime, and
continual-learning loop that already targets agricultural scouting and
tabletop manipulation runs the kitchen-counter robot with no
architectural change.
