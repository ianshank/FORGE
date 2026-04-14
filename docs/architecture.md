# FORGE C4 Architecture

This document describes the FORGE architecture using the [C4 model](https://c4model.com/) — four levels of abstraction from system context down to code-level detail.

---

## Level 1: System Context

Shows FORGE and its external actors.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          FORGE Platform                                 │
│    Fast Open-source Runtime for Generalist Environments                 │
│                                                                         │
│  Deterministic grid-world simulation for training AI agents.            │
│  Procedural worlds, crafting, combat, multi-agent, task curriculum.     │
│  130K+ steps/sec from Python, <8 μs/step.                              │
└──────┬──────────────┬──────────────────┬──────────────┬────────────────┘
       │              │                  │              │
┌──────▼──────┐ ┌─────▼────────┐ ┌──────▼──────┐ ┌────▼────────────────┐
│ RL          │ │  Demo User   │ │   Web       │ │  Rust Application   │
│ Researcher  │ │              │ │   Browser   │ │                     │
│             │ │ Watches live │ │             │ │ Embeds simulation   │
│ Trains      │ │ FORGE demo   │ │ Runs FORGE  │ │ engine directly     │
│ agents via  │ │ at           │ │ via WASM in │ │ as a Rust library   │
│ Python API  │ │ localhost:   │ │ browser     │ │ dependency          │
│ (Gymnasium, │ │ 8765 via the │ │ with JS/    │ │                     │
│  PettingZoo,│ │ Demo UI      │ │ JSON API    │ │                     │
│  JAX, SB3)  │ │              │ │             │ │                     │
└─────────────┘ └──────────────┘ └─────────────┘ └─────────────────────┘
```

### External Actors

| Actor | Interface | Description |
|-------|-----------|-------------|
| RL Researcher | Python (PyO3) | Trains agents using Gymnasium/PettingZoo/JAX APIs |
| Demo User | HTTP (localhost:8765) | Interacts with the live demo via the web UI |
| Web Browser | WASM (JSON) | Runs visualization or interactive demos |
| Rust Application | Cargo crate | Embeds simulation as a library dependency |

---

## Level 2: Container Diagram

Shows the major containers (deployable units) within FORGE.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          FORGE Platform                                 │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     Rust Workspace                               │   │
│  │                                                                  │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌───────────┐  │   │
│  │  │forge-types │  │forge-core  │  │forge-      │  │forge-task │  │   │
│  │  │            │  │            │  │worldgen    │  │           │  │   │
│  │  │ Shared     │  │ Simulation │  │            │  │ Task DSL  │  │   │
│  │  │ types,     │  │ engine,    │  │ Perlin     │  │ Curriculum│  │   │
│  │  │ configs,   │  │ step(),    │  │ noise,     │  │ Evaluator │  │   │
│  │  │ errors     │  │ systems    │  │ biomes,    │  │ 6 tiers   │  │   │
│  │  │            │  │            │  │ resources  │  │           │  │   │
│  │  └────────────┘  └────────────┘  └────────────┘  └───────────┘  │   │
│  │                                                                  │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌───────────┐  │   │
│  │  │forge-agent │  │forge-      │  │forge-server│  │forge-bench│  │   │
│  │  │            │  │procgen     │  │            │  │           │  │   │
│  │  │ MCTS       │  │            │  │ HTTP/WS    │  │ Criterion │  │   │
│  │  │ planner,   │  │ Maps,      │  │ API,       │  │ benchmarks│  │   │
│  │  │ baselines, │  │ objectives,│  │ metrics,   │  │ step      │  │   │
│  │  │ policies   │  │ curriculum │  │ live state │  │ throughput│  │   │
│  │  └────────────┘  └────────────┘  └────────────┘  └───────────┘  │   │
│  │                                                                  │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌───────────┐  │   │
│  │  │forge-cloud │  │forge-edge  │  │forge-data  │  │forge-     │  │   │
│  │  │            │  │            │  │            │  │replay     │  │   │
│  │  │ Distributed│  │ Adaptive   │  │ Dataset    │  │           │  │   │
│  │  │ training,  │  │ MCTS,      │  │ loaders,   │  │ Compact   │  │   │
│  │  │ worker     │  │ telemetry, │  │ expert     │  │ replay,   │  │   │
│  │  │ pool,      │  │ EdgeAgent, │  │ demos,     │  │ trajectory│  │   │
│  │  │ storage    │  │ latency    │  │ edge       │  │ export    │  │   │
│  │  │ backends   │  │ estimator  │  │ replay     │  │           │  │   │
│  │  └────────────┘  └────────────┘  └────────────┘  └───────────┘  │   │
│  │                                                                  │   │
│  │  ┌────────────┐  ┌────────────┐                                  │   │
│  │  │forge-python│  │forge-wasm  │                                  │   │
│  │  │            │  │            │                                  │   │
│  │  │ PyO3       │  │ wasm-      │                                  │   │
│  │  │ bindings,  │  │ bindgen,   │                                  │   │
│  │  │ numpy obs  │  │ JSON I/O   │                                  │   │
│  │  │ GIL release│  │            │                                  │   │
│  │  └────────────┘  └────────────┘                                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                   Python Package (forge_env)                     │   │
│  │                                                                  │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌─────────┐ ┌────────────┐   │   │
│  │  │ gymnasium_   │ │ pettingzoo_  │ │ jax_env │ │ wrappers   │   │   │
│  │  │ env.py       │ │ env.py       │ │ .py     │ │ .py        │   │   │
│  │  │              │ │              │ │         │ │            │   │   │
│  │  │ Single-agent │ │ Multi-agent  │ │ Batched │ │ Flatten,   │   │   │
│  │  │ Gymnasium    │ │ PettingZoo   │ │ JAX     │ │ Normalize, │   │   │
│  │  │ wrapper      │ │ Parallel API │ │ vectorize│ │ TimeLimit  │   │   │
│  │  └──────────────┘ └──────────────┘ └─────────┘ └────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │               Python Package (forge — training & agents)        │   │
│  │                                                                  │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌─────────┐ ┌────────────┐   │   │
│  │  │ agents/      │ │ training/    │ │ models/ │ │ traces/    │   │   │
│  │  │              │ │              │ │         │ │            │   │   │
│  │  │ BaseAgent,   │ │ Trainer,     │ │ Policy  │ │ Decision   │   │   │
│  │  │ RandomAgent, │ │ RolloutBuffer│ │ network,│ │ trace      │   │   │
│  │  │ MCTSAgent    │ │ Checkpointing│ │ world   │ │ logging    │   │   │
│  │  │              │ │              │ │ model   │ │            │   │   │
│  │  └──────────────┘ └──────────────┘ └─────────┘ └────────────┘   │   │
│  │  ┌──────────────┐ ┌──────────────┐                               │   │
│  │  │ config.py    │ │ utils/       │                               │   │
│  │  │              │ │              │                               │   │
│  │  │ TOML config  │ │ Device,      │                               │   │
│  │  │ loader       │ │ logging,     │                               │   │
│  │  │              │ │ metrics,seed │                               │   │
│  │  └──────────────┘ └──────────────┘                               │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Container Descriptions

| Container | Technology | Purpose |
|-----------|-----------|---------|
| **forge-types** | Rust crate | Shared types, config structs, error types. Zero heavy dependencies. |
| **forge-core** | Rust crate | Deterministic simulation engine. `WorldState::step()` is the hot path. |
| **forge-worldgen** | Rust crate | Procedural world generation: Perlin noise terrain, biome classification, resource/object placement. |
| **forge-task** | Rust crate | Composable task DSL with 7 operators, 10 predicates, 6 tiers, and adaptive curriculum. |
| **forge-agent** | Rust crate | MCTS planner with PUCT selection, forward model, baseline agents, and a LatentMctsSearch for `.onnx` PyTorch MuZero models via `ort`. |
| **forge-procgen** | Rust crate | Procedural content generation: map generator, objective generator, team composer, curriculum controller with configurable difficulty scaling. |
| **forge-server** | Rust crate | HTTP/WebSocket API server: REST endpoints, metrics collection, live simulation state streaming. |
| **forge-python** | Rust crate (PyO3) | Python bindings exposing `ForgeEnv` with numpy observations, GIL release during step. |
| **forge-wasm** | Rust crate (wasm-bindgen) | WebAssembly bindings with JSON-string I/O for browser environments. |
| **forge-bench** | Rust crate (Criterion) | Performance benchmarks: step throughput, world creation, serialization. |
| **Python wrappers** | Python package (forge_env) | Gymnasium, PettingZoo, JAX wrappers, observation/reward transforms. |
| **Python framework** | Python package (forge) | Training pipeline, agent implementations, policy networks, MangoMAS bridge modules, decision traces, TOML config loader, utility modules. |

The current PR surface also expands the Python control plane with a MangoMAS bridge layer inside `python/forge/mangomas/`. That layer sits above `forge_env` and below experiment code, providing curriculum progression, constitutional safety shaping, curiosity-weight search, batch episode collection, and repeatable MCTS sweep orchestration without changing the deterministic Rust core.

---

### 2.1 Docker Deployment Architecture

The production deployment packages FORGE as three Docker containers orchestrated via Compose.

```
  User Browser                Developer / RL Researcher
       │                              │
       │ http://localhost:3000        │ http://localhost:8765
       ▼                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                       forge-net (bridge)                          │
│                                                                   │
│  ┌───────────────────────┐   ┌──────────────────────────────┐    │
│  │  dashboard            │   │  demo                        │    │
│  │  nginx:1.27-alpine    │   │  python:3.11-slim            │    │
│  │                       │   │                              │    │
│  │  :80 ──► host:3000    │   │  :8765 ──► host:8765        │    │
│  │                       │   │                              │    │
│  │  Serves React SPA     │   │  FastAPI/uvicorn             │    │
│  │  /api/ ──────────────┐│   │  SSE demo streams            │    │
│  │  /ws   ──────────────┤│   │                              │    │
│  └──────────────────────┼┘   └─────────────┬────────────────┘    │
│                         │                  │                      │
│                         │ proxy to         │ HTTP/WS to sim svc   │
│                         ▼                  ▼                      │
│              ┌───────────────────────────────────┐               │
│              │  simulation                        │               │
│              │  rust:1.85-bookworm (build)        │               │
│              │  python:3.11-slim  (runtime)       │               │
│              │                                    │               │
│              │  :8080 ──► host:8080              │               │
│              │                                    │               │
│              │  forge-server (Axum HTTP/WS)       │               │
│              │  forge_env.so (PyO3 native ext.)   │               │
│              │                                    │               │
│              │  GET /health   → {"status":"ok"}   │               │
│              │  GET /api/config, /api/metrics     │               │
│              │  WS  /ws       → live state        │               │
│              └───────────────────────────────────┘               │
└──────────────────────────────────────────────────────────────────┘
```

**Startup sequence** (health-gated):

1. `simulation` starts → waits for `/health` to return 200 (up to 3×15s retries)
2. `dashboard` and `demo` start only after `simulation` is **healthy**

**Ports** (all bound to `127.0.0.1`):

| Container | Internal | Host | Protocol |
|-----------|----------|------|----------|
| simulation | 8080 | 8080 | HTTP/WS |
| dashboard | 80 | 3000 | HTTP |
| demo | 8765 | 8765 | HTTP/SSE |

**Key files:**

| File | Purpose |
|------|---------|
| `docker/Dockerfile` | `rust:1.85` build → `python:3.11-slim` runtime; maturin native ext |
| `docker/Dockerfile.dashboard` | `node:20` build → `nginx:1.27-alpine` serve |
| `docker/Dockerfile.demo` | `python:3.11-slim`; FastAPI/uvicorn |
| `docker/docker-compose.yml` | Three-service orchestration with health gates |
| `docker/nginx.conf` | SPA routing + `/api/` and `/ws` reverse proxy |
| `.dockerignore` | Excludes `target/`, `node_modules/`, `.git/` |

---

## Level 3: Component Diagram

### 3.1 forge-core — Simulation Engine

The core engine executes a deterministic pipeline of systems every tick.

```
                        WorldState::step(actions)
                                │
                ┌───────────────▼───────────────────┐
                │         Action Validation          │
                │   Replaces invalid → Noop           │
                └───────────────┬───────────────────┘
                                │
         ┌──────────────────────▼───────────────────────────┐
         │                 Physics Phase                     │
         │                                                   │
         │  ┌─────────────┐ ┌──────────────┐ ┌───────────┐  │
         │  │  Movement   │ │  Push        │ │  Stamina   │  │
         │  │  System     │ │  System      │ │  Regen     │  │
         │  │             │ │              │ │            │  │
         │  │ Move agents │ │ Push objects │ │ +regen/tick│  │
         │  │ Check walls │ │ Check bounds │ │ Cap at max │  │
         │  │ Drain stam. │ │ Resolve col. │ │            │  │
         │  │ Resolve tie │ │              │ │            │  │
         │  └─────────────┘ └──────────────┘ └───────────┘  │
         └──────────────────────┬───────────────────────────┘
                                │
         ┌──────────────────────▼───────────────────────────┐
         │              Resource & Crafting Phase             │
         │                                                   │
         │  ┌─────────────┐ ┌──────────────┐ ┌───────────┐  │
         │  │  Harvesting │ │  Respawn     │ │  Crafting  │  │
         │  │             │ │              │ │            │  │
         │  │ PickUp →    │ │ Timer tick   │ │ Recipe     │  │
         │  │ check tool, │ │ Replenish    │ │ lookup,    │  │
         │  │ inv space,  │ │ depleted     │ │ input/     │  │
         │  │ quantity    │ │ resources    │ │ output swap│  │
         │  └─────────────┘ └──────────────┘ └───────────┘  │
         └──────────────────────┬───────────────────────────┘
                                │
         ┌──────────────────────▼───────────────────────────┐
         │              Combat & Damage Phase                 │
         │                                                   │
         │  ┌──────────────────┐ ┌─────────────────────────┐ │
         │  │  Melee Combat    │ │  Environmental Damage    │ │
         │  │                  │ │                          │ │
         │  │ Use(Sword) →     │ │ Lava tiles deal          │ │
         │  │ adjacent target, │ │ 1.0 damage/tick          │ │
         │  │ 3.0 damage       │ │                          │ │
         │  └──────────────────┘ └─────────────────────────┘ │
         └──────────────────────┬───────────────────────────┘
                                │
         ┌──────────────────────▼───────────────────────────┐
         │          Communication & World State Phase         │
         │                                                   │
         │  ┌───────────┐ ┌────────────┐ ┌──────────────┐   │
         │  │   Comms   │ │ Visibility │ │  Day/Night   │   │
         │  │           │ │            │ │              │   │
         │  │ Broadcast │ │ Bresenham  │ │ 4-phase      │   │
         │  │ tokens to │ │ raycasting │ │ cycle from   │   │
         │  │ agents in │ │ per agent, │ │ tick count   │   │
         │  │ radius    │ │ fog of war │ │              │   │
         │  └───────────┘ └────────────┘ └──────────────┘   │
         └──────────────────────┬───────────────────────────┘
                                │
                ┌───────────────▼───────────────────┐
                │         Task Evaluation            │
                │   Progress, rewards, termination   │
                └───────────────┬───────────────────┘
                                │
                ┌───────────────▼───────────────────┐
                │      Observation Generation        │
                │  Per-agent ego-centric grid view   │
                │  → StepResult                      │
                └───────────────────────────────────┘
```

### 3.2 forge-worldgen — World Generation Pipeline

```
         WorldGenerator::generate(rng)
                    │
         ┌──────────▼──────────┐
         │  TerrainGenerator   │
         │                     │
         │  PerlinNoise ─────┐ │        ┌──────────────────┐
         │  ┌─────────┐     │ │        │  BiomeClassifier  │
         │  │Elevation│─────┤ │───────▶│                   │
         │  │noise    │     │ │        │  elevation +      │
         │  └─────────┘     │ │        │  moisture →       │
         │  ┌─────────┐     │ │        │  TerrainType      │
         │  │Moisture │─────┘ │        │  (7 biomes)       │
         │  │noise    │       │        └──────────────────┘
         │  └─────────┘       │
         └──────────┬─────────┘
                    │ Grid with terrain
         ┌──────────▼──────────┐
         │  ResourcePlacer     │
         │                     │
         │  6 resource types:  │
         │  Wood (Forest)      │
         │  Stone (Mountain)   │
         │  Ore (Mountain)     │
         │  Fish (Water)       │
         │  Fiber (Ground)     │
         │  Clay (Sand)        │
         └──────────┬──────────┘
                    │ Grid + ResourceNode[]
         ┌──────────▼──────────┐
         │  ObjectPlacer       │
         │                     │
         │  4 object types:    │
         │  Boulder, Station,  │
         │  Container, Torch   │
         └──────────┬──────────┘
                    │ Grid + Object[]
         ┌──────────▼──────────┐
         │  SpawnPlacer        │
         │                     │
         │  Walkable, clear    │
         │  tiles with 3+      │
         │  Manhattan distance  │
         └──────────┬──────────┘
                    │
                    ▼
         (Grid, Vec<ResourceNode>, Vec<Object>, Vec<Position>)
```

### 3.3 forge-task — Task System

```
                    ┌─────────────────┐
                    │  TaskGenerator   │
                    │                  │
                    │ generate_task()  │
                    │ Tier 1-6        │
                    └───────┬─────────┘
                            │ TaskDefinition
                            ▼
         ┌─────────────────────────────────────┐
         │          TaskComposition              │
         │  (Recursive tree of objectives)       │
         │                                       │
         │  ┌──────┐ ┌─────┐ ┌────────────────┐ │
         │  │ Atom │ │ And │ │ Sequence       │ │
         │  │      │ │     │ │                │ │
         │  │Single│ │ All │ │ Ordered steps  │ │
         │  │pred. │ │     │ │                │ │
         │  └──────┘ └─────┘ └────────────────┘ │
         │  ┌──────┐ ┌───────────┐ ┌──────────┐ │
         │  │  Or  │ │  Before   │ │  While   │ │
         │  │      │ │           │ │          │ │
         │  │ Any  │ │ Deadline  │ │ Maintain │ │
         │  │      │ │           │ │ + goal   │ │
         │  └──────┘ └───────────┘ └──────────┘ │
         │  ┌──────────┐                         │
         │  │ Without  │                         │
         │  │          │                         │
         │  │ Forbidden│                         │
         │  │ action   │                         │
         │  └──────────┘                         │
         └─────────────────┬───────────────────┘
                           │
         ┌─────────────────▼───────────────────┐
         │          10 Predicates               │
         │                                      │
         │  AgentAt         AgentHas            │
         │  AgentNear       AgentOnTerrain      │
         │  TimeElapsed     HealthAbove         │
         │  ResourceCount   TeamAlive           │
         │  ObjectAt        ObjectInState       │
         └─────────────────┬───────────────────┘
                           │
         ┌─────────────────▼───────────────────┐
         │       CurriculumController           │
         │                                      │
         │  record_outcome(success/fail)        │
         │        │                             │
         │        ▼                             │
         │  Rolling window of outcomes          │
         │        │                             │
         │        ▼                             │
         │  success_rate vs target_success_rate │
         │        │                             │
         │  ┌─────▼─────┐  ┌───────────────┐   │
         │  │ Too easy:  │  │ Too hard:     │   │
         │  │ Shift to   │  │ Shift to      │   │
         │  │ harder     │  │ easier        │   │
         │  │ tiers      │  │ tiers         │   │
         │  └────────────┘  └───────────────┘   │
         │        │                             │
         │        ▼                             │
         │  sample_tier() → weighted selection  │
         └──────────────────────────────────────┘
```

### 3.4 forge-agent — Planning Architecture

```
         ┌──────────────────────────────────────┐
         │            MctsSearch                  │
         │                                        │
         │  search(state, agent_idx) → Action     │
         │                                        │
         │  ┌──────────────────────────────────┐  │
         │  │ For each simulation:              │  │
         │  │                                   │  │
         │  │  1. SELECT  ──────────────────┐   │  │
         │  │     PUCT = Q(s,a) + c·P(s,a)  │   │  │
         │  │       · √N_parent / (1+N)     │   │  │
         │  │                               │   │  │
         │  │  2. EXPAND  ◀─────────────────┘   │  │
         │  │     Add children with priors      │  │
         │  │     from PolicyValue              │  │
         │  │              │                    │  │
         │  │  3. EVALUATE │                    │  │
         │  │     value = policy.evaluate()     │  │
         │  │              │                    │  │
         │  │  4. BACKPROP │                    │  │
         │  │     Update ancestors with         │  │
         │  │     discounted value              │  │
         │  └──────────────────────────────────┘  │
         │                                        │
         │  best_action() = argmax(visits at root)│
         └──────────────────────────────────────┘
                       │
                       │ Uses
                       ▼
         ┌─────────────────────────┐
         │     ForwardModel        │
         │                         │
         │  simulate(state, acts)  │
         │  → (next_state, result) │
         │                         │
         │   Clones WorldState and │        │ LatentForwardModel(ort) │
         │  calls step() to look  │        │ simulate(latent, action)│
         │  ahead without mutating│        │ → policy, value, reward │
         │  the real simulation   │        │ Uses pre-trained ONNX   │
         └────────────────────────┘        └─────────────────────────┘

         Baseline Agents:
         ┌──────────┐ ┌───────────────┐ ┌────────────────┐ ┌──────┐
         │ Random   │ │ GreedyNav     │ │ Heuristic      │ │ Noop │
         │ Agent    │ │               │ │ Agent          │ │Agent │
         │          │ │ Manhattan     │ │                │ │      │
         │ Uniform  │ │ toward target │ │ PickUp if avail│ │Always│
         │ sampling │ │               │ │ else random    │ │Noop  │
         └──────────┘ └───────────────┘ └────────────────┘ └──────┘
```

### 3.5 forge-procgen — Procedural Content Generation

```
         ┌──────────────────────────────────────┐
         │          forge-procgen                 │
         │                                        │
         │  ┌──────────────┐  ┌───────────────┐  │
         │  │MapGenerator  │  │ObjectiveGen   │  │
         │  │              │  │               │  │
         │  │ generate()   │  │ generate()    │  │
         │  │ → Grid with  │  │ → Task tree   │  │
         │  │   terrain,   │  │   from tier + │  │
         │  │   resources, │  │   seed        │  │
         │  │   spawns     │  │               │  │
         │  └──────────────┘  └───────────────┘  │
         │                                        │
         │  ┌──────────────┐  ┌───────────────┐  │
         │  │TeamComposer  │  │CurriculumCtrl │  │
         │  │              │  │               │  │
         │  │ compose()    │  │ update_params │  │
         │  │ → Agent team │  │ Adaptive      │  │
         │  │   layout,    │  │ difficulty    │  │
         │  │   roles,     │  │ scaling with  │  │
         │  │   loadouts   │  │ configurable  │  │
         │  │              │  │ thresholds    │  │
         │  └──────────────┘  └───────────────┘  │
         │                                        │
         │  ┌──────────────┐  ┌───────────────┐  │
         │  │GrammarSystem │  │SeedManager    │  │
         │  │              │  │               │  │
         │  │ L-system     │  │ Deterministic │  │
         │  │ rules for    │  │ seed chain    │  │
         │  │ structure    │  │ for           │  │
         │  │ generation   │  │ reproducible  │  │
         │  │              │  │ content       │  │
         │  └──────────────┘  └───────────────┘  │
         └────────────────────────────────────────┘
```

### 3.6 forge-server — API Server

```
         ┌──────────────────────────────────────┐
         │          forge-server                  │
         │                                        │
         │  ┌──────────────┐  ┌───────────────┐  │
         │  │  REST API    │  │  WebSocket    │  │
         │  │              │  │  Handler      │  │
         │  │ GET /api/    │  │               │  │
         │  │   state,     │  │ /ws           │  │
         │  │   metrics,   │  │ Live state    │  │
         │  │   config     │  │ streaming     │  │
         │  └──────────────┘  └───────────────┘  │
         │                                        │
         │  ┌──────────────┐  ┌───────────────┐  │
         │  │ MetricsStore │  │ SimState      │  │
         │  │              │  │               │  │
         │  │ Tracks       │  │ Thread-safe   │  │
         │  │ steps/sec,   │  │ simulation    │  │
         │  │ episode      │  │ state with    │  │
         │  │ stats,       │  │ schema        │  │
         │  │ agent perf   │  │ versioning    │  │
         │  └──────────────┘  └───────────────┘  │
         └────────────────────────────────────────┘
```

### 3.7 Python Bindings — Data Flow

```
     Python User Code
            │
            │  from forge_env import ForgeEnv
            │  env = ForgeEnv(config={...})
            │  obs, info = env.reset(seed=42)
            │  obs, rew, term, trunc, info = env.step(action)
            │
            ▼
  ┌─────────────────────────────┐
  │  forge_env Python Package   │
  │                             │
  │  __init__.py                │
  │    ├── ForgeEnv             │──── Native Rust class (PyO3)
  │    ├── ForgeGymnasiumEnv    │──── gymnasium.Env wrapper
  │    ├── ForgeParallelEnv     │──── PettingZoo ParallelEnv
  │    └── ForgeJaxEnv          │──── JAX-vectorized batched
  │                             │
  │  wrappers.py                │
  │    ├── FlattenObservation   │
  │    ├── NormalizeReward      │
  │    ├── TimeLimit            │
  │    └── RecordEpisodeStats   │
  └──────────┬──────────────────┘
             │  PyO3 FFI
             ▼
  ┌─────────────────────────────┐
  │  forge-python (Rust)        │
  │                             │
  │  ForgeEnv #[pyclass]        │
  │    │                        │
  │    ├── new(config_dict)     │──▶ config_from_dict() → ForgeConfig
  │    │                        │    WorldState::new(config)
  │    │                        │
  │    ├── reset(seed)          │──▶ WorldState::reset(seed)
  │    │                        │    obs_to_dict() → PyDict with numpy
  │    │                        │
  │    ├── step(action: u32)    │──▶ Action::from_discrete(action)
  │    │                        │    py.allow_threads(|| state.step())
  │    │                        │    obs_to_dict() → numpy arrays
  │    │                        │
  │    └── render()             │──▶ WorldState::to_debug_grid()
  └──────────┬──────────────────┘
             │
             ▼
  ┌─────────────────────────────┐
  │  forge-core (Rust)          │
  │                             │
  │  WorldState                 │
  │    ├── step(&[Action])      │──▶ 13-phase deterministic pipeline
  │    ├── reset(seed)          │──▶ Regenerate world from seed
  │    └── to_debug_grid()      │──▶ ASCII rendering
  └─────────────────────────────┘
```

### 3.8 MangoMAS Bridge — Training Control Plane

```
     Experiment Script / Notebook
                              │
                              │ load TOML / build dataclass config
                              ▼
     ┌──────────────────────────────┐
     │ python/forge/mangomas        │
     │                              │
     │ config.py                    │
     │   └── Shared defaults for    │
     │       curriculum, sweeps,    │
     │       constitutional rules,  │
     │       curiosity channels     │
     │                              │
     │ curriculum_controller.py     │
     │   └── Tier unlock / demote   │
     │                              │
     │ constitutional_trainer.py    │
     │   └── Constraint penalties   │
     │                              │
     │ curiosity_optimizer.py       │
     │   └── Evolutionary search    │
     │                              │
     │ batch.py / sweep_runner.py   │
     │   └── Episode collection /   │
     │       MCTS grid evaluation   │
     └──────────────┬───────────────┘
                                         │
                                         │ uses
                                         ▼
     ┌──────────────────────────────┐
     │ forge_env wrappers           │
     │                              │
     │ Optional-native Python API   │
     │ for Gymnasium / PettingZoo / │
     │ vectorized rollout flows     │
     └──────────────┬───────────────┘
                                         │
                                         ▼
     ┌──────────────────────────────┐
     │ forge-python / forge-core    │
     │                              │
     │ Deterministic Rust simulator │
     │ and MCTS planning engine     │
     └──────────────────────────────┘
```

This control-plane split is intentional: branch-specific coverage work focuses on keeping config resolution, optional imports, and fallback behavior stable even when native extensions or heavyweight ML packages are unavailable.

---

## Level 4: Code-Level Detail

### 4.1 Key Data Structures

#### WorldState — The Central State Object

```
WorldState
├── tick: u64                          Current simulation step
├── grid: Grid                         2D tile array (row-major)
│   ├── width: u16
│   ├── height: u16
│   └── tiles: Vec<Tile>              Each tile has terrain, elevation,
│       ├── terrain: TerrainType       agent_id, object_id, resource_id,
│       ├── elevation: u8              visibility state
│       ├── agent_id: Option<u32>
│       ├── object_id: Option<u32>
│       ├── resource_id: Option<u32>
│       └── visibility: VisibilityState
├── agents: Vec<Agent>
│   ├── id: u32
│   ├── position: Position {x, y}
│   ├── health: i32                    Fixed-point (65536 = 1.0)
│   ├── stamina: i32                   Fixed-point (65536 = 1.0)
│   ├── inventory: Inventory
│   │   └── slots: Vec<Option<ItemStack>>
│   ├── comm_buffer: SmallVec<[u16; 8]>
│   └── alive: bool
├── objects: Vec<Object>
│   ├── object_type: ObjectType        Boulder/Door/Switch/Container/
│   ├── mass: i32                      CraftingStation/Bridge/Torch
│   ├── durability: i32
│   └── state: ObjectState
├── resources: Vec<ResourceNode>
│   ├── resource_type: ItemType        Wood/Stone/Ore/Fish/Fiber/Clay
│   ├── quantity: u16
│   ├── respawn_timer: u32
│   └── requires_tool: Option<ItemType>
├── tasks: Vec<ActiveTask>
│   ├── definition: TaskDefinition
│   ├── progress: Vec<f32>
│   └── completed/failed: bool
├── recipe_book: RecipeBook            9 default recipes
├── day_phase: u8                      0=Dawn 1=Day 2=Dusk 3=Night
├── rng: ForgeRng                      Pcg64Mcg deterministic PRNG
├── config: Arc<ForgeConfig>           Shared immutable configuration
├── terminated: bool
└── truncated: bool
```

#### ForgeConfig — Configuration Hierarchy

```
ForgeConfig
├── world: WorldConfig
│   ├── width/height: u16              (default: 64x64)
│   ├── seed: u64                      (default: 0)
│   ├── biome_scale: f32               (default: 0.1)
│   ├── resource_density: f32          (default: 0.3)
│   ├── day_night_cycle_length: u32    (default: 1000, 0=disabled)
│   └── max_entities: u16              (default: 256)
├── physics: PhysicsConfig
│   ├── stamina_cost_move: i32         (default: ~0.1 fixed-point)
│   ├── stamina_regen_rate: i32        (default: ~0.05 fixed-point)
│   └── collision_enabled: bool        (default: true)
├── agents: AgentConfig
│   ├── num_agents: u32                (default: 1)
│   ├── default_vision_radius: u8      (default: 5 → 11x11 view)
│   ├── default_carry_capacity: u8     (default: 10)
│   ├── starting_health: i32           (default: 10.0 fixed-point)
│   ├── comm_vocab_size: u16           (default: 16)
│   └── comm_radius: u16              (default: 10, 0=global)
├── crafting: CraftingConfig
│   └── enabled: bool                  (default: true)
├── task: TaskConfig
│   ├── max_episode_length: u64        (default: 10000)
│   ├── max_tier: u8                   (default: 6)
│   └── dense_rewards: bool            (default: true)
└── curriculum: CurriculumConfig
    ├── enabled: bool                  (default: false)
    ├── target_success_rate: f32       (default: 0.5)
    └── window_size: u32              (default: 100)
```

#### Observation — Per-Agent View

```
Observation
├── grid_view: Vec<TileObservation>    (2r+1)x(2r+1) ego-centric grid
│   └── [terrain, has_agent, has_object, has_resource, elevation,
│        object_type, resource_type]    7 channels per tile
├── inventory: InventoryObservation    (capacity, 2) array
│   └── slots: Vec<(item_type, count)> 255 = empty slot
├── health: f32                        Normalized [0.0, 1.0]
├── stamina: f32                       Normalized [0.0, 1.0]
├── position: (u16, u16)              Absolute (x, y)
├── messages: Vec<u16>                 Recent comm tokens
└── day_phase: u8                      0-3
```

### 4.2 Crate Dependency Graph

```
                    forge-types
                   (shared types)
                ╱   │   │    ╲    ╲
               ╱    │   │     ╲    ╲
              ▼     ▼   ▼      ▼    ▼
        forge-  forge- forge- forge- forge-
        worldgen core   task  agent  procgen
             ╲    │    ╱     ╱    ╱
              ╲   │   ╱     ╱    ╱
               ▼  ▼  ▼    ╱    ╱
             forge-python ╱    ╱
             forge-wasm  ╱    ╱
             forge-server    ╱
             forge-bench ───╱
```

| Crate | Dependencies |
|-------|-------------|
| forge-types | serde, thiserror, smallvec, fixed, rand, rand_pcg, tracing |
| forge-worldgen | forge-types, rand_pcg, tracing |
| forge-core | forge-types, forge-worldgen, rand, rand_pcg, fixed, serde, bincode, smallvec, tracing |
| forge-task | forge-types, rand, tracing |
| forge-agent | forge-types, forge-core, rand, rand_pcg, serde, tracing |
| forge-procgen | forge-types, forge-core, rand, rand_pcg, serde, tracing |
| forge-server | forge-types, forge-core, serde, serde_json, tracing |
| forge-python | forge-types, forge-core, forge-worldgen, forge-task, pyo3, numpy, serde_json, tracing |
| forge-wasm | forge-types, forge-core, serde, serde_json, wasm-bindgen, tracing |
| forge-bench | forge-types, forge-core, rand, rand_pcg, criterion |

### 4.3 Determinism Guarantees

FORGE guarantees that `same seed + same actions = byte-identical state`:

```
┌──────────────────────────────────────────────────────┐
│                 Determinism Stack                      │
│                                                       │
│  ┌─────────────────────────────────────────────────┐  │
│  │  RNG: Pcg64Mcg (seeded, deterministic)          │  │
│  │  ForgeRng wraps PCG with generation counting    │  │
│  └─────────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────┐  │
│  │  Arithmetic: Fixed-point i32 (16 frac bits)     │  │
│  │  No floating-point in hot path                  │  │
│  │  65536 = 1.0 (FP_ONE)                          │  │
│  └─────────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────┐  │
│  │  System order: Fixed pipeline (13 phases)       │  │
│  │  Agent priority: Lower ID wins ties             │  │
│  └─────────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────┐  │
│  │  State serialization: bincode for snapshots     │  │
│  │  RNG replay from (seed, generation_count)       │  │
│  └─────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

### 4.4 Python Test Architecture

The Python surface is validated in layers so wrapper logic, pure-Python fallbacks, and training utilities can evolve without depending on a fully built native extension in every test.

```
tests/python/
├── conftest.py                 Shared fixtures built from exported defaults
├── test_forge_env.py           Wrapper contracts, fallback imports, utils branches
├── test_mappo.py               MAPPO config factories, policy behavior, batch actions
├── test_device.py              CPU/CUDA/MPS detection without hardware dependencies
├── test_dashboard_client.py    Dashboard client contract tests
└── ...                         Module-focused tests for wrappers, config, and training
```

Test layering keeps the suite fast and deterministic:

- Pure-Python tests validate wrapper bookkeeping, reward normalization bounds, and helper utilities without requiring the Rust extension
- Native-optional tests call `_skip_if_no_native()` so CI can still execute the Python suite when the extension is unavailable
- Import/device branches are tested with module patching instead of machine-specific hardware assumptions
- Shared constants in fixtures and assertions keep config defaults aligned with production modules instead of duplicating literals

### 4.5 Action Space Encoding

```
Index:  0   1   2   3   4   5   6 ··· 15  16 ··· 25  26 ··· 34  35 36 37 38  39  40 ···
       ─┬─ ─┬───┬───┬───┬─ ─┬─ ─┬─────┬─ ─┬──────┬─ ─┬──────┬─ ─┬──┬──┬──┬─ ─┬─ ─┬─────
        │   │   │   │   │   │   │     │   │      │   │      │   │  │  │  │   │   │
       Noop  Move(4 dirs)  Pick  Drop(10) Use(10)  Craft(9) Push(4) Int Comm(N)
                            Up   slots     slots    recipes  dirs  er  tokens
                                                                   act
```

### 4.6 Biome Classification

```
     Elevation
        1.0 ┬─────────────────────────────────────────┐
            │                Mountain                  │
       0.72 ┤─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┤ mountain_level
            │     Sand     │    Ground    │   Forest   │
            │   (desert)   │             │            │
       0.40 ┤─ ─ ─ ─ ─ ─ ─┤─ ─ ─ ─ ─ ─ ┤─ ─ ─ ─ ─ ┤ sand_level
            │     Sand (beach)                        │
       0.35 ┤─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┤ water_level
            │                 Water                    │
        0.0 ┴─────────────────────────────────────────┘
           0.0          0.25          0.45           1.0
                         Moisture ──────▶
                    desert_moist   forest_moist
```

---

## CI/CD Pipeline

```
  git push / PR
       │
       ▼
  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │   fmt    │  │  clippy  │  │   test   │  │  bench   │  │  python  │
  │          │  │          │  │          │  │          │  │          │
  │ cargo    │  │ cargo    │  │ cargo    │  │ cargo    │  │ ruff     │
  │ fmt --   │  │ clippy   │  │ test     │  │ bench    │  │ mypy     │
  │ check    │  │ -D warn  │  │ --verbose│  │ --no-run │  │ pytest   │
  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘
       All run on: ubuntu-latest, stable Rust, with caching
```

---

## Performance Architecture

```
  Python user code
       │
       │  env.step(action)
       ▼
  ┌──────────────┐
  │  PyO3 FFI    │  ~0.5 μs overhead
  │  GIL release │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │  Rust step() │  ~7 μs total
  │              │
  │  Zero-alloc  │  No heap allocation
  │  Fixed-point │  Integer arithmetic
  │  SmallVec    │  Stack-allocated collections
  │  Row-major   │  Cache-friendly grid layout
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │  numpy array │  Zero-copy where possible
  │  construction│
  └──────────────┘

  Release profile:
    LTO = true
    codegen-units = 1
    opt-level = 3
```
