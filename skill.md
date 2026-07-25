# FORGE MCTS & Training Loop Developer Playbook

This playbook serves as a developer reference for working with the Monte Carlo Tree Search (MCTS) engine, Muzero continuous training pipeline, cross-language schemas, and zero-allocation hot paths within the FORGE workspace.

---

## 🚀 CLI Subcommands & Operations

The `forge.training.muzero_mc.cli` module serves as the primary entry point for managing models, manifests, baselines, and active training loops.

```bash
python -m forge.training.muzero_mc.cli [SUBCOMMAND] [ARGS]
```

### 1. `bootstrap`
Initializes a new, randomly-weighted MuZero model bundle (Representation, Dynamics, Prediction networks) in ONNX format and writes a version 1 model manifest.
* **Usage**:
  ```bash
  python -m forge.training.muzero_mc.cli bootstrap --obs-dim 920 --action-dim 73 --schema-id <sha256_hash> --out models/
  ```
* **Developer Guidelines**:
  - Requires `torch` and `onnx` to generate and serialize the model graphs.
  - Automatically computes the parameter sizes based on input dimensions and applies Xavier initialization.
  - Returns `EXIT_OK` (0) on success, `EXIT_USAGE` (3) on invalid dimensions, and `EXIT_IO` (4) on file system write failures.

### 2. `validate-manifest`
Performs a fast, torch-free verification of a `model_manifest.json` file or its parent directory.
* **Usage**:
  ```bash
  python -m forge.training.muzero_mc.cli validate-manifest models/
  ```
* **Validation Gates**:
  - Parses file as strict JSON.
  - Validates `schema_version` against `MANIFEST_SCHEMA_VERSION` (pinned to `1`).
  - Ensures SHA-256 integrity hashes are present and correctly formatted for all three network roles (representation, dynamics, prediction).
  - Returns `EXIT_VALIDATION` (5) if schema drift is detected, `EXIT_IO` (4) if the file is missing, and `EXIT_OK` (0) on success.

### 3. `train`
Executes single-run or continuous gradient-descent training loops over collected trajectories.
* **Usage (Continuous Mode)**:
  ```bash
  python -m forge.training.muzero_mc.cli train \
    --input trajectories/ \
    --out models/ \
    --schema-id <sha256_hash> \
    --obs-dim 920 \
    --action-dim 73 \
    --continuous \
    --round-iters 10 \
    --round-poll-sleep 5 \
    --max-trajectories 200 \
    --max-bundle-versions 5 \
    --device cpu
  ```
* **Continuous Flow Principles**:
  - **Cold-Start Safety**: The loop is resilient to empty or sparse input directories; it polls the trajectory directory using the configured sleep interval until new episodes are registered.
  - **Atomic Exports**: Each training round writes to an isolated subfolder (e.g., `models/v00000001/`) containing the `.onnx` models, followed by an atomic manifest write (`.tmp-*.json` followed by `os.replace`), which triggers the runner's `HotReloadWatcher` without risking half-written loads.
  - **Graceful SIGINT Handling**: Catches keyboard interrupts to cleanly finalize the current round, write the latest checkpoints, restore the original signal handlers, and exit with `EXIT_OK`.

### 4. `compute-schema-id`
Computes the canonical 64-hex SHA-256 `schema_id` by hashing the combined contents of the action mapping and reward TOML configurations.
* **Usage**:
  ```bash
  python -m forge.training.muzero_mc.cli compute-schema-id --action-map configs/minecraft/action_map.toml --rewards configs/minecraft/rewards.toml --quiet
  ```
* **Implementation Details**:
  - In `--quiet` mode, the command outputs *only* the 64-character hex string to stdout. All log information is redirected to stderr to allow clean bash pipelines:
    `FORGE_MC_SCHEMA_ID=$(python -m forge.training.muzero_mc.cli compute-schema-id ... --quiet)`
  - Returns `EXIT_VALIDATION` (5) on malformed TOML structures and `EXIT_IO` (4) on missing files.

### 5. `capture-baseline`
Drives an episode-capture orchestration against a running Minecraft Stack, recording trajectory logs.
* **Usage**:
  ```bash
  python -m forge.training.muzero_mc.cli capture-baseline --variant random --episodes 100 --out baseline_random.json
  ```
* **Orchestration**:
  - Coordinates episodes via WebSockets.
  - Safe directory routing prevents file evictions mid-capture by leveraging variant-specific trajectory folders.
  - Handles `TimeoutError` and network `OSError` boundaries with friendly diagnostic output, returning `EXIT_IO` (4).

---

## 🧠 Monte Carlo Tree Search (MCTS) Architecture

FORGE leverages a hybrid search architecture executing a latent-space search over simulated paths.

### 1. The Core Loop
The MCTS agent searches ahead in latent space using three model components:
1. **Representation Network**: Encodes the raw observation $o_t$ into an initial latent state $s_0$.
2. **Dynamics Network**: Given $s_k$ and a selected action $a_{k+1}$, predicts the next latent state $s_{k+1}$ and the immediate transition reward $r_{k+1}$.
3. **Prediction Network**: Estimates the policy distribution $p_k$ and value scalar $v_k$ for a given latent state $s_k$.

### 2. Search Invariants & Configuration
Search parameters are loaded via the `LatentMctsSearch` configuration:
* **Simulations**: Number of forward rollout expansions per step (typically `50-200`).
* **Exploration Policy**: Guided by Dirichlet noise injected into the root node priors:
  - $\alpha$: Concentration parameter (defaults to `0.3` for generalist environments).
  - $\epsilon$: Fraction of noise mixed into the policy (defaults to `0.25`).
* **PUCT Formula**: Action selection scales with:
  $$U(s, a) = C_{puct} \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}$$
  Where $N(s)$ is visit counts and $P(s, a)$ is predicted prior probability.

---

## ⚡ Zero-Allocation Hot Path Contract

To achieve high throughput, the core simulation loop enforces a **zero-allocation** constraint. Heap allocations during active stepping degrade cache locality and introduce garbage collection/deallocation overhead.

### 1. Invariant Rules
- **Buffer Reuse**: Instead of returning newly allocated structs, the engine requires caller-owned mutable buffers.
  - **Allocating (Slow)**: `state.step(action) -> StepResult`
  - **Zero-Alloc (Fast)**: `state.step_into(action, &mut step_result_buffer)`
- **Warmup Allocation**: All memory required for tree search and trajectory storage is pre-allocated during initialization. The tree reuse logic resets node structures in place rather than dropping and re-allocating them.
- **Verification**: Zero-allocation compliance is strictly enforced in Rust CI via Criterion allocation audits:
  ```bash
  cargo run -p forge-bench --bin allocation_audit
  ```

### 2. Minecraft/WebSocket Carve-Out
- **Protocol Bounds**: Wire-bound environments (such as `forge-env-mc` integrating with `mc-bot`) implement the same unified `Env::step_into` interface and reuse observation buffers.
- **Exemption**: Internal WebSocket I/O, raw byte decoding, and JSON message parsing *are* allowed to heap-allocate. This carve-out is documented and accepted because network latency dominates wire-bound steps, rendering microsecond memory overhead insignificant.

---

## 🔄 Cross-Language (xlang) Schema Contracts

To ensure perfect synchronization between the Rust simulation runner and the Node `mc-bot` interface, FORGE employs strict cross-language schema verification.

```
+------------------------------------+      +-----------------------------------+
|               Rust                 |      |               Node                |
|      (forge-env-mc client)         |      |             (mc-bot)              |
+-----------------+------------------+      +-----------------+-----------------+
                  |                                           |
                  |             WebSocket Handshake           |
                  | <---------------------------------------> |
                  |   Validates:                              |
                  |   - obs_dim (920)                         |
                  |   - grid_shape {11,11,1,7,73}             |
                  |   - schema_id (SHA-256 of map/rewards)    |
                  v                                           v
```

### 1. The `schema_id` Checksum
Both sides load identical TOML config files:
- `configs/minecraft/action_map.toml`
- `configs/minecraft/rewards.toml`

At handshake, the Node side asserts that its local SHA-256 matches the hex string presented by the Rust runner. If there is a mismatch, the connection is immediately terminated with a handshake failure, preventing poisoned self-play loops.

### 2. Drift Verification
- **Rust Side**: `xlang_schema_version_*` tests verify TOML structural integrity and compute the SHA-256.
- **Node Side**: Pinned unit tests assert the exact action and reward lengths and checksum matches.
- A drift on either side breaks both test suites simultaneously.

---

## 🛠️ Hardening & Verification Guidelines

When contributing code to either the python training scripts or the Rust runner workspace:

### 1. Cross-Platform Directory Paths
Always ensure directory manipulations use platform-agnostic formatting.
- Do not use hardcoded backslashes (`\`) or forward slashes (`/`). Use Python's `pathlib.Path` or Rust's `PathBuf`.
- **WSL Path Parsing**: When driving bash scripts under WSL from Windows tests, convert local windows absolute paths (e.g. `C:\Users\...`) to POSIX mounts (e.g. `/mnt/c/Users/...`) using paths normalized through `get_posix_path`.

### 2. CRLF vs LF Line Endings
Scripts executed under bash inside containers or WSL *must* use strictly Unix line endings (`\n`).
- When writing files dynamically in tests, write them in binary mode (`write_bytes`) or specify `newline='\n'` to prevent Windows Python from translating newlines to CRLF (`\r\n`).
- Git checkout settings on Windows can automatically insert carriage returns. Use the `lf_normalized_script` helper in integration tests to strip `\r` from scripts before execution.

### 3. Ruff & Mypy Checkpoints
Verify all changes before pushing:
```bash
# Style compliance
ruff check .
ruff format --check .

# Type safety
mypy python/
```

## 📊 Observability & Server Infrastructure

### Structured logging (`FORGE_LOG_FORMAT`)
One env var flips text↔JSON logging across all three languages — keep them in lock-step:
- **Rust**: binaries call `forge_observability::init_tracing(TracingOptions::new("<default-filter>"))` — never re-implement `tracing_subscriber` setup. `RUST_LOG` overrides the filter.
- **Python**: `forge.utils.logging_config.setup_logging_from_env()` (honours `FORGE_LOG_FORMAT` + `FORGE_LOG_LEVEL`).
- **Node (mc-bot)**: inject `createLogger()` at the wiring seam (no static side-effects).
- Values: `text` (default) or `json`. Unknown values fall back to `text`.

### `forge-server` REST + persistent history
- Run: `FORGE_SERVER_HISTORY_DIR=/tmp/forge-history cargo run -p forge-server`.
- Writes append-only JSONL under `FORGE_SERVER_HISTORY_DIR` (default `forge-history/`, gitignored). Storage is behind the `HistoryStore` trait (`JsonlHistoryStore` default, `InMemoryHistoryStore` for tests) — swap impls without touching handlers.
- Endpoints: `POST /api/training-metrics`, `POST /api/decision-traces` (persist + broadcast); `GET /api/training-metrics/history`, `GET /api/decision-traces/history`, `GET /api/runs` (`runId`/`limit` query params). Run id: `?runId=` → `X-Forge-Run-Id` header → server-session id.
- Config knobs: `FORGE_SERVER_HISTORY_DIR` / `_RETENTION` (default 10000) / `_QUERY_LIMIT` (default 500). Add any new `FORGE_SERVER_*` key to `ALL_KEYS` in `config.rs` tests or env tests flake.

### Monitoring stack (opt-in)
`docker compose -f docker/compose.minecraft.yml --profile monitoring up -d prometheus grafana` — Prometheus (`:9091`) scrapes the runner's `forge_mc_*` metrics; Grafana (`:3001`) auto-provisions from `docker/monitoring/`. Requires `metrics_bind = "0.0.0.0"` in `runner.toml` (container-internal). Default `up` is unaffected.
