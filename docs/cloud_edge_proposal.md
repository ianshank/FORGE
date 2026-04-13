# FORGE Cloud-to-Edge: Distributed Training and Edge Deployment Architecture

## Context

FORGE (Fast Open-source Runtime for Generalist Environments) is a high-performance multi-agent RL simulation platform. Its foundational design decisions -- byte-identical determinism, full-state serialization, zero-allocation hot path, fixed-point arithmetic, and cross-platform compilation (native/WASM/Python) -- were made for correctness and speed. This proposal argues these same properties are exactly what's needed for a distributed cloud training pipeline and an edge deployment system, and that the two halves form a closed feedback loop.

---

## Part A: Cloud Training Pipeline

### The Case

FORGE already achieves 130K+ steps/sec from Python on a single core. The architecture is inherently parallelizable because determinism eliminates simulation state synchronization -- only compact replay packets (seed + config + actions, ~160KB per 10K-tick episode) need to transit the network. This is a 300x compression over full trajectories.

### Architecture: Three Tiers

1. **Rollout Workers** (N stateless GKE pods on spot node pools)
   - Each runs `forge-core` natively with a unique seed range
   - Produces compact replays (~160KB each) published to Cloud Pub/Sub
   - Horizontally scalable via GKE Autopilot -- add pods linearly, nodes auto-provision
   - Existing `rayon` parallelism in `ExpertDemoGenerator` is the single-node template
   - Spot VMs at ~$0.056/hr; preemption-safe because workers are stateless and seed-deterministic

2. **Training Coordinator / Parameter Server** (Vertex AI Training custom job)
   - Subscribes to Pub/Sub, reconstructs full trajectories via `CompactReplay::replay()`
   - Runs gradient computation (PPO / MuZero / offline RL) on A100 GPUs or TPU v4 pods
   - Publishes updated weights (~5MB) to GCS; workers pull on next rollout batch
   - Python training loop via existing `forge-python` Gymnasium API
   - Checkpoints to GCS every N gradient steps for fault tolerance

3. **Orchestration & Monitoring**
   - Vertex AI Pipelines (Kubeflow) orchestrates the full train loop end-to-end
   - `forge-server` Axum WebSocket protocol extended for distributed coordination
   - Cloud Monitoring + custom dashboards for real-time training metrics from all workers
   - Cloud Logging ingests structured JSON logs from `tracing` subscriber
   - Existing `TrainingMetrics` and `DecisionTraceEntry` WebSocket messages feed the dashboard

### Why It Works

| FORGE Property | Cloud Benefit |
|---|---|
| Determinism (PCG RNG + fixed-point) | Workers reproduce any episode from compact replay -- no state sync needed |
| Compact replay format | 160KB per 10K-tick episode = network is never the bottleneck |
| `WorldState::to_bytes()` / `from_bytes()` | Trivial checkpointing to Cloud Storage (GCS) buckets |
| `ForgeConfig` with TOML + env var overrides | GKE ConfigMap injection, per-worker config via Workload Identity |
| `ForgeEnv` Gymnasium compatibility | Direct integration with Vertex AI Training custom containers |
| Zero-allocation `step()` | Predictable memory per worker, precise GCE instance sizing (no overprovisioning) |
| `ForgeJaxEnv` wrapper | Hardware-accelerated vectorized batching on Cloud TPU / A100 GPU instances |

### Cost Efficiency (GCP Pricing)

- A single `c3-standard-4` ($0.1860/hr on GCE) produces ~468M steps/hour via FORGE
- A comparable Python-only Gymnasium env at 5K steps/sec needs 26x more instances
- At scale: ~$1.1K/day vs ~$29K/day for equivalent training throughput
- Compact replays reduce Cloud Storage egress and Pub/Sub message costs by 300x vs full trajectory shipping
- Spot/Preemptible VMs (`c3-standard-4` spot at ~$0.056/hr) reduce rollout worker cost by 70% -- stateless workers tolerate preemption gracefully (just re-seed and restart)
- GCS Standard storage for replay archive: ~$0.02/GB/month; a million 160KB compact replays = ~160GB = ~$3.20/month
- Vertex AI Training auto-scales coordinator GPU instances (A100 at $3.67/hr) only during gradient computation

### Curriculum Scaling on GCP

- Existing `CurriculumConfig` (target_success_rate, window_size, warmup_episodes) drives distributed curriculum
- Coordinator aggregates success rates from all workers via Pub/Sub, broadcasts difficulty adjustments
- Scenario TOML configs (patrol, escort, crop_scout, etc.) stored in GCS, pulled by workers at episode start
- Workers assigned to different scenario tiers based on agent proficiency -- GKE node pool affinity routes harder tiers to more capable instances
- Vertex AI Experiments tracks curriculum progression metrics across training runs

### GCP Integration Points

- **Vertex AI Training**: Custom training containers using existing `docker/Dockerfile`. `ForgeEnv` Gymnasium API works directly with Vertex AI custom training jobs. Checkpointing via bincode to GCS. Vertex AI Experiments for tracking hyperparameters and metrics across runs.
- **Google Kubernetes Engine (GKE)**: Stateless rollout worker pods on GKE Autopilot (auto-scales node pools, pays per pod). `forge.toml` via ConfigMap, `FORGE_*` env vars via Kubernetes Secrets / Workload Identity. GKE spot node pools for 70% cost reduction on rollout workers.
- **Cloud Pub/Sub**: Message queue for compact replay transport (worker -> coordinator). Pub/Sub Lite for high-throughput, lower-cost replay streaming. Dead-letter topics for failed replay processing.
- **Cloud Storage (GCS)**: Replay archive, model checkpoint store, trajectory dataset hosting. Lifecycle policies auto-tier old replays to Nearline/Coldline. GCS FUSE mount for transparent access from training containers.
- **Cloud TPU**: `ForgeJaxEnv` wrapper enables JAX-native vectorized batching on TPU v4/v5e pods. TPU slices for distributed MuZero training. ~3-5x cost-efficiency vs A100 for large batch RL.
- **Artifact Registry**: Container images for rollout workers, training coordinator, dashboard. ONNX model versioning and edge deployment staging.
- **Cloud Monitoring + Cloud Logging**: `forge-server` metrics exported via OpenTelemetry to Cloud Monitoring. Structured JSON logs (already using `tracing` with JSON subscriber) to Cloud Logging. Custom dashboards in Cloud Monitoring for training progress.
- **Vertex AI Pipelines (Kubeflow)**: Orchestrate the full training loop: rollout -> replay collection -> trajectory reconstruction -> gradient computation -> model export -> evaluation. Scheduled retraining pipelines triggered by edge replay ingestion thresholds.
- **Existing Docker stack**: `docker-compose.yml` already has simulation + dashboard + demo services -- maps directly to GKE Deployment manifests

---

## Part B: Edge Device Offline Training & Deployment

### The Case

FORGE's core engine achieves <1us per step, uses no heap allocation on the hot path, requires no FPU (fixed-point arithmetic), and compiles to WASM. These aren't just nice-to-have performance properties -- they're the exact requirements for running on constrained hardware: drones, IoT gateways, agricultural robots, military autonomous systems.

### Why FORGE Runs on Edge

| FORGE Property | Edge Benefit |
|---|---|
| WASM compilation (`forge-wasm`) | Runs on any WASM runtime: Wasmtime, WAMR, WasmEdge (ARM Cortex, RISC-V, x86 embedded) |
| <1us per step | Real-time MCTS planning even on 100MHz microcontrollers |
| Zero-allocation hot path | No heap fragmentation, no GC pauses, no OOM on memory-constrained devices |
| Fixed-point `i32` arithmetic | No FPU required -- full speed on Cortex-M0/M3 without hardware float |
| Pre-sized buffers (`PhysicsScratch`, `AgriScratch`) | Predictable, calculable memory footprint at init time |
| `LatentForwardModel` + ONNX Runtime | Small models (~350KB total) for edge inference |

### Latent MCTS for Edge Inference

The `LatentForwardModel` trait with `OnnxMuZeroModel` backend is designed exactly for this:

- Three small ONNX models: representation (~200KB), dynamics (~100KB), prediction (~50KB)
- Latent MCTS searches in 256-dim vector space rather than cloning full `WorldState`
- Each MCTS simulation is a matrix multiply, not a full environment step
- `OnnxModelConfig` is tunable: `latent_dim`, `action_space_size`, `num_threads`
- ONNX Runtime has embedded-optimized builds (ONNX Runtime Mobile for ARM)

### Compact Telemetry from Edge to GCP

- Edge devices record `CompactReplay` (seed + config + action sequence)
- A 1000-tick single-agent mission = ~16KB
- Over a 2400-baud satellite link: ~53 seconds upload vs ~4.4 hours for full trajectories
- Upload path: edge device -> MQTT -> Cloud IoT Core (or direct HTTPS to GCS signed URL)
- Cloud-side `CompactReplay::replay()` reconstructs full trajectories with complete fidelity
- GCS event notification triggers Vertex AI Pipeline for automatic trajectory reconstruction
- This is the difference between feasible and infeasible in bandwidth-constrained operations

### Offline RL Retraining Loop (GCP-Native)

1. Edge compact replays land in GCS bucket via Cloud IoT Core / signed URL upload
2. GCS object notification triggers Cloud Function or Pub/Sub event
3. Vertex AI Pipeline kicks off: `CompactReplay::replay()` reconstructs full `Trajectory`
4. Feed into `OfflineDataset` -- retrain via Decision Transformer, IQL, or CQL on Vertex AI Training (A100/TPU)
5. Export updated ONNX models via `MuZeroExporter` to Artifact Registry with semantic versioning
6. Edge devices pull new model version on next connectivity window (via Artifact Registry pull or Cloud IoT config push)

This is a **data flywheel**: more edge deployment produces more diverse training data in GCS, which triggers retraining pipelines, which produces better policies in Artifact Registry, which perform better at the edge. The entire loop is event-driven and serverless-triggerable.

---

## Part C: The Cloud-Edge Feedback Loop

### Unified Architecture (GCP)

```
          GOOGLE CLOUD PLATFORM                           EDGE
┌──────────────────────────────────┐      ┌─────────────────────────────┐
│                                  │      │                             │
│  GKE Autopilot (spot pools)     │      │  WASM/Native ARM Runtime    │
│  ┌─ Rollout Worker Pods ──────┐ │      │  (forge-core or forge-wasm) │
│  │  forge-core, N instances   │ │      │         |                   │
│  └────────────|───────────────┘ │      │         v                   │
│               v                 │      │  LatentMctsSearch +         │
│  Cloud Pub/Sub                  │      │  OnnxMuZeroModel            │
│  (compact replay transport)     │      │         |                   │
│               |                 │      │         v                   │
│               v                 │      │  Compact Replay Logger      │
│  Vertex AI Training             │      │  (~16KB per mission)        │
│  (A100 / TPU v4 coordinator)    │      │         |                   │
│  PPO / MuZero / Offline RL      │      │         |                   │
│               |                 │      │         |                   │
│               v                 │      │         |                   │
│  Artifact Registry ─────────────│──>───│  OTA Model Update           │
│  (versioned ONNX models)        │      │  (via Cloud IoT / MQTT)     │
│               ^                 │      │         |                   │
│               |                 │      │         |                   │
│  GCS Bucket <───────────────────│──<───│  Upload Compact Replays     │
│  (replay archive + OfflineDS)   │      │  (store-and-forward)        │
│               |                 │      │                             │
│               v                 │      └─────────────────────────────┘
│  Vertex AI Pipelines            │
│  (Kubeflow: retrain trigger)    │
│               |                 │
│               v                 │
│  Cloud Monitoring + Logging     │
│  (dashboards, alerts, traces)   │
└──────────────────────────────────┘
```

### Why This Loop Is Unique

- **Determinism is the lynchpin**: GCP perfectly reconstructs any edge episode from a 16KB compact replay stored in GCS. Edge devices never need to store/transmit observations, rewards, or state.
- **Serialization is the glue**: Config, state, model, replay, trajectory -- all serde. No format translation between GCP services and edge runtimes.
- **ONNX is the model interchange**: Vertex AI trains, exports ONNX to Artifact Registry. Rust loads ONNX via `ort`. Same model file on GCE x86, ARM edge, WASM runtime.
- **ForgeConfig is the contract**: Single source of truth via TOML/env vars. GKE pods and edge devices use identical configs, guaranteeing identical simulation behavior. GCS-hosted config with versioning.
- **`config_hash` in `CompactReplay`** validates cloud-edge config compatibility.
- **GCP-native observability**: Cloud Monitoring traces the full loop -- rollout throughput, replay ingestion rate, training loss curves, model export events, edge device health -- in a single pane of glass.

---

## Concrete Use Cases

### 1. MouseDroidAGI -- Autonomous Robot Navigation (Flagship Edge Use Case)

The [Mouse-Droid-AGI](https://github.com/ianshank/Mouse-Droid-AGI) project is a physical Star Wars MSE-6 "Mouse Droid" replica that implements autonomous navigation on an NVIDIA Jetson Orin Nano. It is a companion project to FORGE by the same author, and represents the most concrete realization of the cloud-to-edge training loop described in this proposal.

**Hardware**: Jetson Orin Nano (8GB), Waveshare mecanum-wheel chassis, ESP32 motor controller, RPi AI Camera (IMX500), HC-SR04 ultrasonic sensor, wheel encoders, USB microphone. Constrained compute (8GB shared RAM, embedded GPU) with real-time requirements.

**AI Architecture -- "10 Pillars"**: RSSM world model, BDI cognitive architecture, MCTS planning (50-200 simulations, UCB c=1.41), Constitutional RL (PPO + Asimov's Three Laws safety monitor), curiosity-driven exploration (ICM), elastic weight consolidation for continual learning, knowledge distillation, meta-learning (MAML).

**FORGE Integration Already Exists**:
- FORGE contains a dedicated `MouseDroidAgent` class (`python/forge/agents/mousedroid_agent.py`) composing RSSM, BDI, NeuralMCTSPolicy, and ActorCriticNetwork
- Shared weight repository on HuggingFace Hub (`ianshank/mousedroid-weights`)
- FORGE's `WeightLoader` (`python/forge/utils/weight_loader.py`) defaults to this repo
- Agent config at `configs/agents/mousedroid.toml`
- Identical MCTS hyperparameters (UCB c=1.41) across both codebases

**The Cloud-Edge Loop in Practice**:

```
  GCP (Vertex AI)                              Jetson Orin Nano
┌──────────────────────────┐      ┌──────────────────────────────────┐
│                          │      │                                  │
│  FORGE simulation        │      │  MouseDroidAGI runtime           │
│  (130K steps/sec)        │      │  (RSSM + MCTS + BDI)            │
│  ┌─ MouseDroidAgent ───┐│      │                                  │
│  │ RSSM pretrain (2.1) ││      │  Real-world navigation           │
│  │ MCTS warm-start(2.2)││      │  Obstacle avoidance              │
│  │ BDI annotation (2.3)││      │  Voice commands (Whisper STT)    │
│  │ PPO fine-tune  (2.4)││      │  Constitutional safety monitor   │
│  └──────────────────────┘│      │         |                        │
│           |               │      │         v                        │
│           v               │      │  Compact replay logger           │
│  ONNX/TensorRT export ───│──>───│  (~16KB per patrol mission)      │
│  to Artifact Registry     │      │         |                        │
│           ^               │      │         |                        │
│           |               │      │         v                        │
│  OfflineDataset <─────────│──<───│  Upload via WiFi/cellular        │
│  (real-world trajectories)│      │  to GCS signed URL               │
│           |               │      │                                  │
│           v               │      │  TensorRT fp16 inference         │
│  Retrain with EWC         │      │  <50ms per MCTS search           │
│  (no catastrophic forget) │      │  on Jetson embedded GPU          │
└──────────────────────────┘      └──────────────────────────────────┘
```

**Why this is the ideal edge target for FORGE**:
- **Sim-to-real**: FORGE's grid-world models navigation, obstacle avoidance, resource gathering, and multi-agent coordination -- the same tasks the physical droid performs. FORGE serves as the high-throughput pre-training environment before sim-to-real transfer.
- **Shared brain**: The `MouseDroidAgent` in FORGE is architecturally identical to the droid's onboard AI. Weights trained in simulation deploy directly to the Jetson via HuggingFace Hub / Artifact Registry.
- **Compact telemetry fits the platform**: A patrol mission generates ~16KB compact replay. The Jetson uploads this over WiFi at the charging station. GCP reconstructs the full trajectory for offline retraining.
- **Constitutional safety**: Both FORGE (Constitutional RL constraints) and Mouse-Droid-AGI (Asimov's Three Laws safety monitor) share the same safety framework. Safety constraints validated in simulation carry to the physical robot.
- **Continual learning without catastrophic forgetting**: Mouse-Droid-AGI uses Elastic Weight Consolidation (EWC). Cloud retraining on mixed sim + real-world data produces updated models that preserve prior knowledge. This is the data flywheel at its most concrete: droid explores -> uploads replays -> cloud retrains with EWC -> droid gets better without forgetting.
- **Future: multi-droid coordination**: Mouse-Droid-AGI roadmap includes multi-agent coordination. FORGE already supports N-agent scenarios with communication. Cloud trains cooperative policies in FORGE, deploys to a fleet of physical droids.

### 2. Precision Agriculture Drone Fleet
- **Scenarios already built**: `crop_scout.toml`, `spray_mission.toml`, `soil_relay.toml`, `field_report.toml`, `irrigation_mapping.toml`
- **Systems already modeled**: Crop growth, disease spread, NDVI/thermal scanning, soil sensors, spraying, drone altitude/battery, ground vehicle terrain costs
- **Cloud**: Train multi-drone scouting policy via curriculum (tier 1: scout -> tier 2: spray -> tier 3: irrigation -> tier 4: full report)
- **Edge**: ONNX models on drone flight controllers, latent MCTS for real-time path planning, ~16KB telemetry per flight

### 3. Search and Rescue Coordination
- **Scenario**: `search_and_rescue.toml` -- 128x128 grid, 2-8 agents, fog of war, hazard zones
- **Cloud**: Pre-train cooperative search policies with multi-agent communication (configurable `comm_vocab_size`, broadcast radius)
- **Edge**: Robots run latent MCTS for real-time replanning as targets are discovered; compact replays uploaded post-mission

### 4. Perimeter Patrol & Area Denial
- **Scenarios**: `patrol.toml`, `area_denial.toml`, `adversarial_recon.toml`
- **Sensor model**: Visual + Acoustic + Radar with range, noise, jamming
- **SBIR-ready**: `forge-proposal` crate already has DoD, AFWERX, DARPA templates

### 5. Autonomous Warehouse Operations
- Navigation + crafting (assembly) + resource gathering (picking) + multi-agent coordination
- Task DSL (`Sequence`, `And`, `Without`) models complex assembly workflows

---

## GCP Service Mapping

| Pipeline Component | GCP Service | Why This Service |
|---|---|---|
| Rollout worker compute | GKE Autopilot (spot pools) | Auto-scaling, spot pricing ($0.056/hr), stateless pod model |
| Replay message transport | Cloud Pub/Sub (or Pub/Sub Lite) | Serverless, ordered delivery, dead-letter, scales to millions of msgs/sec |
| Replay + checkpoint storage | Cloud Storage (GCS) | Durable, versioned, lifecycle policies, event notifications, GCS FUSE |
| Training compute (GPU) | Vertex AI Training (A100) | Managed training jobs, auto-shutdown, experiment tracking |
| Training compute (TPU) | Cloud TPU v4/v5e | 3-5x cost-efficient for large-batch JAX workloads via `ForgeJaxEnv` |
| Model registry | Artifact Registry | Container + ONNX model versioning, vulnerability scanning, IAM-scoped pulls |
| Pipeline orchestration | Vertex AI Pipelines (Kubeflow) | DAG-based workflows, caching, lineage tracking |
| Edge device management | Cloud IoT Core / MQTT | Device registry, config push, telemetry ingestion |
| Edge replay upload | GCS signed URLs | No service account on device; time-limited, scoped upload |
| Monitoring | Cloud Monitoring + Logging | Custom metrics, structured logs, SLOs, alerting policies |
| Fleet analytics | BigQuery | Trajectory metadata, curriculum progression, fleet-wide query |
| Infrastructure-as-code | Terraform (GCP provider) | Reproducible provisioning, state in GCS backend |
| CI/CD | Cloud Build | Container builds, test execution, deployment triggers |

---

## Key Existing Infrastructure to Leverage

| Component | File | Role in Cloud-Edge |
|---|---|---|
| CompactReplay | `crates/forge-replay/src/compact.rs` | Central data transfer for both cloud (worker->coordinator) and edge (device->cloud) |
| OnnxMuZeroModel | `crates/forge-agent/src/latent_mcts/onnx_model.rs` | Bridge between cloud training and edge inference |
| ExpertDemoGenerator | `crates/forge-data/src/generator.rs` | Template for distributed rollout workers (already uses rayon) |
| WebSocket infra | `crates/forge-server/src/ws_handler.rs` | Monitoring/orchestration backbone to extend |
| ForgeWasmEnv | `crates/forge-wasm/src/lib.rs` | Proves cross-platform compilation; edge deployment foundation |
| ForgeConfig | `crates/forge-types/src/config.rs` | Cloud-edge contract (TOML + env var overrides) |
| OfflineDataset | `crates/forge-data/src/loader.rs` | Aggregator for edge-collected trajectories |
| EvalHarness | `crates/forge-eval/src/harness.rs` | Parallel evaluation, extends to distributed benchmarking |
| Scenario configs | `configs/scenarios/*.toml` | Pre-built curriculum for agriculture, defense, SAR |
| Docker stack | `docker/docker-compose.yml` | Starting point for GKE container images (Artifact Registry) |

---

## Implementation Roadmap (GCP)

### Phase 1: Cloud Foundation on GCP (3 months)
- New `forge-cloud` crate: distributed rollout coordinator with Cloud Pub/Sub integration for replay transport
- GKE Autopilot deployment manifests for rollout worker pods (spot node pools, Workload Identity for GCS/Pub/Sub access)
- Vertex AI Training custom container for coordinator (A100 GPU or TPU v4 pod)
- `forge.training.distributed_trainer` Python module consuming replays from Pub/Sub subscription
- GCS bucket structure: `gs://forge-training/{run_id}/replays/`, `gs://forge-training/{run_id}/checkpoints/`, `gs://forge-training/{run_id}/models/`
- Cloud Monitoring custom metrics + dashboard for training throughput, replay ingestion rate, loss curves
- Terraform / Pulumi IaC for reproducible GCP infrastructure provisioning
- Benchmark: linear scaling verification to 100 GKE worker pods

### Phase 2: Edge Deployment Pipeline (3 months, overlapping)
- Optimize `forge-wasm` binary size (wasm-opt, feature gating, dead code elimination)
- `no_std` support path for `forge-core` and `forge-types` for bare-metal targets
- ONNX Runtime integration testing on ARM (Raspberry Pi 4, Jetson Nano, Coral Edge TPU)
- Edge telemetry daemon: compact replay batching, compression, store-and-forward upload to GCS via signed URLs
- New `forge-edge` crate: edge runtime config, battery-aware scheduling, failsafe policies
- Artifact Registry setup for ONNX model versioning and edge device pull authentication
- Cloud IoT Core / MQTT bridge for edge device management and config push

### Phase 3: Closed Loop on GCP (2 months)
- GCS object notification -> Cloud Function -> Vertex AI Pipeline trigger for automatic replay ingestion
- Vertex AI Pipeline definition: replay reconstruction -> offline RL training -> ONNX export -> Artifact Registry push
- Automated model promotion workflow: staging -> canary (deploy to 10% of edge fleet) -> production
- End-to-end integration test: Vertex AI train -> Artifact Registry -> edge deploy -> edge collect -> GCS -> retrain -> verify improvement
- BigQuery export of trajectory metadata for fleet-wide analytics and curriculum tuning

### Phase 4: Production Hardening (2 months)
- Security: Artifact Registry container/model signing, VPC Service Controls for GCS/Pub/Sub, IAM least-privilege for edge device service accounts
- Monitoring: Cloud Monitoring SLOs and alerting on worker pod failures, edge device heartbeats, replay ingestion lag, training job health
- Cloud Logging structured queries for debugging failed episodes, model export errors, edge upload failures
- Documentation and SBIR proposal templates for specific agency submissions
- Cost optimization: Committed Use Discounts for sustained GKE/Vertex AI usage, lifecycle policies for GCS replay archival
