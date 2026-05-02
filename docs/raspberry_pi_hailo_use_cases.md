# FORGE on Raspberry Pi 5 + Hailo AI HAT 2 + OpenClaw

Concrete use cases for deploying FORGE-trained policies to a Raspberry Pi 5
fitted with the Hailo AI HAT 2 (Hailo-8/Hailo-8L NPU) and an OpenClaw
end-effector. Each case grounds the usage in existing FORGE primitives —
`forge-edge`, `forge-agent::latent_mcts`, `forge-replay::compact`, and the
scenario library under `configs/scenarios/`.

> Assumption: "OpenClaw" refers to an open-source robotic gripper/manipulator
> controlled over serial/USB or GPIO (servo or stepper). If a different
> platform is intended, swap the actuator-mapping section accordingly; the
> rest of this document is platform-agnostic.

---

## 1. Reference Hardware Stack

| Component | Role | Notes |
|---|---|---|
| Raspberry Pi 5 (8 GB) | Host SBC | 4× Cortex-A76, USB 3, PCIe via M.2 HAT |
| Hailo AI HAT 2 (Hailo-8L 13 TOPS or Hailo-8 26 TOPS) | NPU for ONNX inference | PCIe M.2, HailoRT runtime |
| OpenClaw gripper | End-effector | USB/UART servo bus assumed |
| RPi Camera Module 3 (or AI Camera IMX500) | Vision input | Used for sim-to-real perception |
| Optional: HC-SR04 / ToF sensor | Obstacle detection | Maps to FORGE `TileObservation::has_object` |
| Optional: 4G/Wi-Fi modem | Telemetry uplink | Carries `CompactReplay` (~16 KB/mission) |

### FORGE side
| Crate / Module | Role on this hardware |
|---|---|
| `forge-edge::EdgeAgent` | Top-level on-device agent (already implements `AgentInterface`) |
| `forge-edge::AdaptiveMctsSearch` | Latency-budgeted MCTS, drops sims to fit budget |
| `forge-edge::TelemetryCollector` | Store-and-forward `CompactReplay` buffer |
| `forge-agent::latent_mcts::OnnxMuZeroModel` | Three small ONNX models (~350 KB total) — compiled to Hailo `.hef` |
| `forge-types::config::EdgeConfig` | `mcts_latency_budget_ms`, `onnx_num_threads`, `gcs_model_bucket`, etc. |
| `crates/forge-wasm` | Optional fallback runtime if Hailo path is unavailable |

The default `EdgeConfig` already carries the knobs needed here
(`mcts_latency_budget_ms`, `mcts_min_simulations`, `mcts_max_simulations`,
`telemetry_interval_s`, `gcs_model_bucket`) — no schema changes are required
to bring up Hailo as the inference backend.

---

## 2. Action ↔ OpenClaw Mapping

FORGE's discrete action space already encodes manipulation primitives that
align with a gripper:

| FORGE action | OpenClaw effect |
|---|---|
| `0` Noop | Hold position |
| `1–4` Move | Drive base/chassis (Up/Down/Left/Right or hex equivalents) |
| `5` PickUp | Close gripper on the resource at the current tile |
| `6–15` Drop slot N | Open gripper, release inventory slot N |
| `16–25` Use slot N | Trigger tool-specific actuator (e.g. spray, scan) |
| `26–34` Craft N | Compose multi-step manipulation routine |
| `35–38` Push | Linear push without grip |
| `39` Interact | Call site-specific subroutine (open hatch, dock) |

This is the same action layout `EdgeAgent` already returns from
`select_action` — the bridge layer just translates the discrete ID into a
serial command stream for OpenClaw.

---

## 3. Use Cases

### Case A — Tabletop Sort & Place

**Goal**: pick coloured blocks from a tray and place them in the matching
bin.

**FORGE training**
- 32×32 `Square` grid, `num_agents = 1`.
- `task::Atom(AgentHas)` → `Sequence([PickUp, AgentAt(bin), Drop])`.
- Resource types reused as block colours; bins as `ObjectAt` predicates.
- Train MuZero in cloud via the existing pipeline; export ONNX with
  `OnnxMuZeroModel`.

**Edge**
- Compile the three ONNX heads (representation, dynamics, prediction) to
  `.hef` with the Hailo Dataflow Compiler.
- Wrap the `.hef` behind a `LatentForwardModel` impl (`HailoLatentModel`)
  and hand it to `EdgeAgent::new(...)`.
- `EdgeConfig.mcts_latency_budget_ms = 50` keeps planning under camera
  frame interval.

**Why it's a fit**: pick/drop are first-class actions; latent MCTS at
~50 ms/decision suits a non-time-critical desktop demo.

---

### Case B — Shelf-to-Bench Fetch Robot

**Goal**: small mobile base + OpenClaw retrieves a named tool from a shelf
and brings it back.

**FORGE training**
- Load `configs/scenarios/escort.toml` style mission, replace escortee with
  a tool token.
- Multi-stage task: `Sequence([AgentAt(shelf), PickUp, AgentAt(bench), Drop])`.
- `comm_vocab_size = 8` if a voice front-end issues spoken targets (token =
  tool ID).

**Edge**
- Hailo HAT 2 runs the policy + value heads at 13/26 TOPS — leaves the four
  Cortex-A76 cores free for SLAM and OpenClaw motion control.
- `TelemetryCollector` flushes one `CompactReplay` per fetch when the robot
  re-docks (Wi-Fi available).

**Why it's a fit**: covers all four FORGE pillars (navigation, recognition,
manipulation, telemetry) in one platform that stays under ~$300 BOM.

---

### Case C — Indoor Patrol with Object Retrieval

**Goal**: continuous patrol of a small site (lab, garage); pick up
out-of-place items and return them to a marked tile.

**FORGE training**
- Reuse `configs/scenarios/patrol.toml` with `crafting.enabled = true` so
  the agent's inventory has slots for retrieved items.
- Add `Without(action = Push)` to teach gentle handling — directly
  expressible in the task DSL.

**Edge**
- `AdaptiveMctsSearch` lets the device degrade to `mcts_min_simulations`
  when battery is low (`EdgeConfig` already exposes the floor).
- Compact replays stream to `gs://forge-training/{run_id}/replays/` via the
  existing `forge-edge::telemetry` path; cloud retraining loop
  (`docs/cloud_edge_proposal.md` Part B) is unchanged.

**Why it's a fit**: same architectural template as the MouseDroidAGI case
already documented in `docs/cloud_edge_proposal.md`, but on a cheaper
RPi5+Hailo platform with an end-effector.

---

### Case D — Small-Plot Precision Agriculture

**Goal**: scout a small field, sample soil/leaves with OpenClaw, return for
analysis.

**FORGE training**
- Use the existing `configs/scenarios/crop_scout.toml`,
  `soil_relay.toml`, `field_report.toml` scenarios.
- Drone-style observation fields (`obs.altitude`, `obs.battery`,
  `obs.morphology`) are already flattened by
  `forge-edge::edge_agent::flatten_observation` — no plumbing change needed.

**Edge**
- IMX500 / RPi Camera 3 → preprocessing → Hailo NPU for the
  representation head; latent MCTS runs in 256-D.
- OpenClaw performs the `PickUp` action on the scouted tile (leaf clip,
  soil core).
- Battery-aware budget: `EdgeConfig.mcts_latency_budget_ms` clamped lower
  when `obs.battery < 0.3`.

**Why it's a fit**: FORGE already ships agricultural scenarios and
agri-specific observation channels; only the actuator changes.

---

### Case E — STEM / Education Kit

**Goal**: a $250 classroom kit that runs a real RL agent end-to-end.

**Stack**
- RPi 5 + Hailo HAT 2 (or fall back to `forge-wasm` on bare RPi 5 if no
  HAT is fitted).
- Curriculum from `configs/curriculum/{beginner,intermediate}.toml`.
- Demo UI under `demo_ui/` (FastAPI + SSE) runs locally on the Pi —
  students watch ASCII world stream while OpenClaw mirrors the agent's
  actions on a desktop layout.

**Why it's a fit**: the existing demo UI is already lightweight; FORGE's
deterministic seed model means students get reproducible runs they can
reason about.

---

### Case F — Constitutional-Safety Validation Rig

**Goal**: a benchtop fixture for validating safety constraints before they
ship to a larger robot.

**FORGE training**
- Use `forge-cognitive`'s constitutional layer + `forge-integration-layer`
  to enforce Asimov-style constraints during training.
- Tasks expressed as `Without(action = Push)` or `While(HealthAbove(0.5),
  goal)` to encode "don't damage" / "don't grip too hard".

**Edge**
- The Hailo HAT runs the value head at high speed; if the predicted value
  drops below a constitutional threshold, `EdgeAgent` returns its
  configurable fallback action (currently `0 = Noop` — see
  `edge_agent.rs:104`).
- OpenClaw observes a software E-stop tied to that fallback.

**Why it's a fit**: FORGE already has the constitutional plumbing and the
fallback path; the Pi rig becomes a cheap physical validator before
deployment to a higher-cost robot.

---

### Case G — Continual-Learning Field Unit (closed loop)

**Goal**: deploy a fleet of RPi5+Hailo+OpenClaw units; they collect
real-world experience, upload compact replays, and pull updated models OTA.

**Loop** (mirrors `docs/cloud_edge_proposal.md` Part C)
1. `EdgeAgent` runs missions, `TelemetryCollector` buffers
   `CompactReplay` payloads (~16 KB each).
2. On dock / Wi-Fi, telemetry flushes via `ReplayTransport` to GCS
   (`EdgeConfig.gcs_model_bucket`/`gcs_model_prefix` already wired).
3. Cloud pipeline reconstructs trajectories with `CompactReplay::replay()`,
   retrains, exports new ONNX, recompiles to `.hef`, ships to Artifact
   Registry.
4. Each Pi pulls on its `model_update_interval_s` cadence.

**Why it's a fit**: the entire feedback loop is already designed; this
case just instantiates it on a low-cost edge platform with a manipulator.

---

## 4. What Has to Be Built

The use cases above lean almost entirely on existing FORGE infrastructure.
The deltas are:

1. **`HailoLatentModel`**: a `LatentForwardModel` implementor that wraps
   HailoRT and a compiled `.hef`. Lives next to `OnnxMuZeroModel` under
   `crates/forge-agent/src/latent_mcts/`. Feature-gated (`hailo`) so it
   doesn't regress non-Hailo builds.
2. **OpenClaw bridge**: a small Python or Rust shim that consumes
   `AgentResponse.action_id` and emits the gripper/drive commands in
   §2. No FORGE-internal changes required; it sits outside the workspace
   or in a new `forge-actuator` crate if we want it in-tree.
3. **Hailo compile recipe**: a `scripts/export_hailo.py` companion to
   `scripts/export_edge.py` that takes the exported ONNX and runs the
   Hailo Dataflow Compiler end-to-end. Optional — many users will run
   this off-device anyway.
4. **Documented `EdgeConfig` profile**: a `configs/edge/rpi5_hailo.toml`
   with sensible defaults (e.g. `mcts_latency_budget_ms = 33` for 30 Hz,
   `onnx_num_threads = 2`, `telemetry_buffer_bytes = 4_194_304`).

None of these require schema or trait changes; they slot into the existing
extension points.

---

## 5. Out-of-Scope (for now)

- Real-time motion-planning control loops below ~30 Hz — FORGE plans at
  the discrete-action layer, not the joint-trajectory layer.
- `no_std` / bare-metal FORGE — RPi 5 runs full Linux; the bare-metal path
  is tracked separately in `docs/cloud_edge_proposal.md` Phase 2.
- Multi-arm coordination on a single Pi — single OpenClaw per node is
  assumed; multi-agent coordination uses one Pi per agent.
