# FORGE Configuration Catalog

This catalog documents all configuration files across `configs/` and the root workspace, specifying their loader structs, schema versions, validation behaviors, and supported environment variable overrides.

---

## 1. Architectural Configuration Principles

1. **Strict Deserialization (`deny_unknown_fields`)**: All Rust configuration structs in `crates/forge-types/src/config.rs` enforce `#[serde(deny_unknown_fields)]`. Any unrecognized key causes an immediate deserialization error, preventing silent configuration drift.
2. **Environment Variable Hierarchy**: Explicit command-line arguments take precedence over environment variables, which take precedence over TOML configuration files, which take precedence over in-code defaults.
3. **Cross-Language Schema Alignment**:
   - Root `forge.toml` and `configs/training/` use the Python RL trainer schema (`[hardware]`, `[simulation]`, `[training]`).
   - `crates/forge-types` uses the Rust engine schema (`[world]`, `[physics]`, `[agents]`, `[day_night]`, etc.).

---

## 2. Global Environment Overrides

| Environment Variable | Target System | Default Value | Description |
|:---------------------|:--------------|:--------------|:------------|
| `FORGE_CONFIG_PATH` | Python Trainer | `configs/dry_run.toml` | Path to active TOML configuration file |
| `FORGE_LOG_FORMAT` | `forge-observability` | `text` | Log format: `text` or `json` |
| `FORGE_PLOTLY_JS_URL` | `forge-eval` | `https://cdn.plot.ly/plotly-3.7.0.min.js` | Custom Plotly URL for air-gapped evaluation artifact rendering |
| `FORGE_MC_METRICS_URL` | `forge-mc-runner` / Python | `http://127.0.0.1:9090/metrics` | Prometheus metrics scrape target |
| `FORGE_MC_SCHEMA_ID` | `forge-mc-runner` | *(none — required in TOML or env)* | Combined two-input `schema_id` (`sha256(action_map + ":" + rewards)`). Nested reward **contents** fold in. |
| `FORGE_MC_RANDOM_ACTIONS` | `forge-mc-runner` | empty = no override | Env-var ladder over `runner.toml` `random_actions`. `true`/`false`/`1`/`0`; garbage keeps TOML. CLI `--random-actions` is OR-only (can force true, never false). |
| `FORGE_MC_RUNNER_EPISODES` | `forge-mc-runner` | *(TOML default)* | Override `RunnerConfig.episodes` |
| `FORGE_LMSTUDIO_BASE_URL` | `forge-cognitive` / Eval | `http://127.0.0.1:1234/v1` | Base URL for local LLM teacher inference |
| `MLFLOW_TRACKING_URI` | `forge-eval` | *(none)* | Remote MLflow tracking server URI |
| `MLFLOW_TRACKING_TOKEN` | `forge-eval` | *(none)* | Bearer authentication token for MLflow REST API |
| `FORGE_SERVER_BIND` | `forge-server` | `127.0.0.1:8080` | Socket address. Loopback by default; image/compose set `0.0.0.0:8080` inside containers. Wins over `FORGE_SERVER_PORT`. |
| `FORGE_SERVER_PORT` | `forge-server` | `8080` | Port-only override; ignored when `FORGE_SERVER_BIND` is set |
| `FORGE_SERVER_TICK_MS` | `forge-server` | `100` | Tick interval in milliseconds |
| `FORGE_SERVER_AUTH_TOKEN`| `forge-server` | *(none)* | Bearer token required on mutating routes when set |
| `FORGE_SERVER_HISTORY_DIR` | `forge-server` | `forge-history` | JSONL history directory (`/home/forge/forge-history` in the simulation image) |
| `FORGE_CLOUD_STORAGE_BACKEND` | `forge-cloud` | `local` | Storage provider: `local` or `gcs` |
| `FORGE_CLOUD_GCS_BUCKET` | `forge-cloud` | *(none)* | Google Cloud Storage bucket for replays & models |
| `FORGE_CLOUD_GCS_PREFIX` | `forge-cloud` | `forge/` | GCS key prefix |
| `FORGE_SKILLS_CONFIG_PATH` | Python `SkillCatalog` | `configs/agents/skills_default.toml` | Path to the hierarchical skill catalog TOML |
| `FORGE_SKILLS_ENABLED` | Rust `SkillsConfig` | `false` | Opt-in switch for engine-level skill catalog consumption |
| `FORGE_SKILLS_DEFAULT_SKILL` | Rust `SkillsConfig` | `explore` | Default option id when a hierarchical policy has no active skill |
| `FORGE_SKILLS_DEFAULT_HORIZON` | Rust `SkillsConfig` | `32` | Fallback option horizon in ticks |

---

## 3. Configuration Index by Subsystem

### 3.1 Core Simulation & Server

| Path | Loader Struct / Module | Schema / Version | Purpose |
|:-----|:-----------------------|:-----------------|:--------|
| `forge.toml` | Python `ForgeConfig` | Python v1 | Default root configuration for training and environment parameters |
| `configs/dry_run.toml` | Python `ForgeConfig` | Python v1 | Minimal configuration for fast local smoke tests and CI dry-runs |
| `(in-code)` | `forge_types::config::ForgeConfig` | Rust v0.5 | Strict typed engine configuration (`[world]`, `[physics]`, `[agents]`) |
| `(in-code)` | `forge_server::config::ServerConfig` | Rust v0.5 | HTTP/WebSocket server bind options and history buffer limits |

### 3.2 Minecraft Environment & Episode Runner (`forge-env-mc`, `mc-bot`, `forge-mc-runner`)

| Path | Loader Struct / Module | Schema / Version | Purpose |
|:-----|:-----------------------|:-----------------|:--------|
| `configs/minecraft/env.toml` | `MinecraftEnvConfig` / `mc-bot` | v1 | Host connection, bot credentials, and path references for local dev |
| `configs/minecraft/env.docker.toml` | `MinecraftEnvConfig` / `mc-bot` | v1 | Containerized connection parameters for Dockerized runner pipelines |
| `configs/minecraft/runner.toml` | `forge_mc_runner::config::RunnerConfig` | v1 | Episode batch limits, max steps, trajectory export dir, and schema hash |
| `configs/minecraft/action_map.toml` | `forge_env_mc::action_map::ActionMap` | `schema_version = "v1"` | Discrete integer action ID to mineflayer command mappings |
| `configs/minecraft/rewards.toml` | `forge_env_mc::reward_config::RewardConfig` | `schema_version = "v1"` | Dense and sparse reward weights. Nested `config_path` / `crafting_config_path` **file contents** fold into `schema_id` (path-string rename without a content change does not bump). Missing nested files fail closed. |
| `configs/minecraft/milestone_rewards.toml` | nested under rewards | v1 | Milestone bonus rewards; hashed by content into the rewards canonical SHA |
| `configs/minecraft/crafting_rewards.toml` | nested under rewards | v1 | Crafting reward tables; hashed by content into the rewards canonical SHA |
| `configs/minecraft/reset.toml` | `forge_env_mc::protocol::ResetPayload` | v1 | Bot respawn strategy, inventory clearing, and coordinate teleportation |
| `configs/minecraft/block_embeddings.toml` | `forge_env_mc::block_embeddings::BlockEmbeddings` | v1 | Block-id embedding vocab (`max(index)+1`). **Obs-layout pin**, not folded into today's two-input `schema_id`. |

### 3.3 MangoMAS Multi-Agent Benchmark & Scenario Collection

| Path | Loader Struct / Module | Schema / Version | Purpose |
|:-----|:-----------------------|:-----------------|:--------|
| `configs/mangomas/default.toml` | `forge_mangomas::config::MangoMasConfig` | v1 | Baseline multi-agent benchmark parameters and swarm thresholds |
| `configs/mangomas/bdi_mapping.toml` | Python `collector.scenario` | v1 | Belief-Desire-Intention cognitive mapping for agent actions |
| `configs/mangomas/constitutional_mapping.toml` | Python `collector.scenario` | v1 | Safety rules, boundaries, and constitutional constraints |
| `configs/mangomas/car_curriculum.toml` | `forge_task::curriculum::Curriculum` | v1 | Progressive difficulty stages for wheeled ground agent tasks |
| `configs/mangomas/drone_curriculum.toml`| `forge_task::curriculum::Curriculum` | v1 | 3D navigation and surveillance curriculum stages for aerial drones |
| `configs/mangomas/muzero_drone.toml` | Python `muzero` trainer | v1 | MuZero model hyperparameters, latent dimensions, and MCTS budgets |
| `configs/mangomas/rssm_pretrain.toml` | Python world model | v1 | Recurrent State-Space Model pretraining configuration |
| `configs/mangomas/sweep_mcts.toml` | Python hyperparameter sweep | v1 | Grid search parameters for MCTS exploration constants |

### 3.4 Evaluation & Benchmarking (`forge-eval`)

| Path | Loader Struct / Module | Schema / Version | Purpose |
|:-----|:-----------------------|:-----------------|:--------|
| `configs/eval/e2e_long_preset.toml` | `forge_eval::config::EvalConfig` | v1 | End-to-end evaluation preset specifying scenarios, MLflow sink, and LLM teacher |
| `configs/scenarios/*.toml` | `forge_eval::scenario::Scenario` | v1 | 11 benchmark scenarios (e.g. `patrol.toml`, `area_denial.toml`, `escort.toml`, `search_and_rescue.toml`) |

### 3.5 Training & Agent Policies

| Path | Loader Struct / Module | Schema / Version | Purpose |
|:-----|:-----------------------|:-----------------|:--------|
| `configs/training/sac_default.toml` | Python SAC trainer | v1 | Soft Actor-Critic hyperparameters (learning rates, discount, entropy) |
| `configs/training/ppo_default.toml` | Python PPO trainer | v1 | Proximal Policy Optimization hyperparameters (GAE lambda, clip range) |
| `configs/training/distributed.toml` | Python distributed / `forge-cloud` | v1 | Multi-worker distributed training layout, cloud storage, and edge budgets |
| `configs/agents/mcts_default.toml` | `forge_agent::latent_mcts::LatentMctsConfig` | v1 | MCTS simulation count, c_puct exploration constant, and dirichlet noise |
| `configs/agents/hybrid_default.toml` | `forge_agent` | v1 | Combined heuristic and neural policy parameters |
| `configs/agents/mappo_default.toml` | Python MAPPO trainer | v1 | Multi-Agent PPO policy and value network architecture |
| `configs/agents/mousedroid.toml` | `forge_agent` | v1 | Autonomous patrol agent behavior tree configuration |
| `configs/agents/skills_default.toml` | `forge_types::skill::SkillsConfig` / `forge.agents.skills.SkillCatalog` | v1 | Hierarchical skill catalog (options/HRL primitives) shared by Rust `HierarchicalSkillAgent` and Python `HierarchicalSkillPolicy`. Override path via `FORGE_SKILLS_CONFIG_PATH`; `FORGE_SKILLS_ENABLED`, `FORGE_SKILLS_DEFAULT_SKILL`, and `FORGE_SKILLS_DEFAULT_HORIZON` override the in-engine catalog. |

### 3.6 Cognitive, Memory, Social & Integration

| Path | Loader Struct / Module | Schema / Version | Purpose |
|:-----|:-----------------------|:-----------------|:--------|
| `configs/cognitive/default.toml` | `forge_cognitive::config::CognitiveConfig` | v1 | LLM provider selection, reasoning timeout, and prompt templates |
| `configs/cognitive/qwen14b_teacher.toml` | `forge_cognitive::config::TeacherConfig` | v1 | Qwen-14B teacher model parameters, temperature, and context length |
| `configs/cognitive/gemma_e4b_teacher.toml`| `forge_cognitive::config::TeacherConfig` | v1 | Gemma-4B teacher model parameters |
| `configs/memory/default.toml` | `forge_memory::config::MemoryConfig` | v1 | Episodic memory buffer capacity, decay rates, and retrieval top-k |
| `configs/social/default.toml` | `forge_social::config::SocialConfig` | v1 | Trust matrix update factors, alliance thresholds, and gossip weights |
| `configs/integration/default.toml` | `forge_integration_layer::config::IntegrationConfig` | v1 | Cross-layer orchestration frequencies and event routing rules |

---

## 4. Validation & Verification

Run the pinned configuration consistency validator to ensure all cross-file versions and URLs match their canonical sources:

```bash
python3 scripts/check_pinned_config_consistency.py
```
