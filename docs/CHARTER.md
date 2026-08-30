# FORGE — Project Charter

> The durable "why and what" of FORGE: its purpose, its scope boundaries, and
> the small set of invariants that every change is expected to preserve. Read
> this before planning work. Day-to-day operational conventions (build/test
> commands, code style) live in [`CLAUDE.md`](../CLAUDE.md); the full
> architecture is in [`docs/architecture.md`](architecture.md); working tasks
> and roadmap live in [`docs/next_steps.md`](next_steps.md).

---

## Core Purpose

FORGE — **Fast Open-source Runtime for Generalist Environments** — is a
high-performance simulation platform for training and evaluating AI agents,
built in Rust with first-class Python and WebAssembly bindings. It provides
deterministic, procedurally generated grid worlds with crafting, combat,
multi-agent cooperation, and a composable task curriculum, running at
130,000+ steps/second from Python.

On top of that core, FORGE runs a **self-improving MuZero loop against live
Minecraft**: a Rust episode runner drives an environment over a WebSocket
bridge to a Node `mc-bot`, records flat-tensor trajectories, trains a MuZero
model in Python, and hot-reloads the exported model back into the runner
between episodes — all under versioned, cross-language schema contracts.

---

## What's Being Built

FORGE is a multi-crate Rust workspace with three language surfaces that share
canonical, versioned contracts:

- **Rust core** — the deterministic simulator and the training/serving stack.
- **Python** — the MuZero training pipeline (`python/forge/training/muzero_mc/`)
  and the MangoMAS control plane.
- **Node `mc-bot`** — the Minecraft bridge (mineflayer + a pluggable reward
  registry), out of the Cargo workspace.

The Rust crates group by responsibility (see
[`docs/architecture.md`](architecture.md) for the full crate-by-crate
breakdown — it is the single source of truth for that list, not this charter):

| Group | Representative crates | Role |
|---|---|---|
| Core simulation | `forge-core`, `forge-types`, `forge-worldgen`, `forge-civ` | Deterministic step function, shared types, procedural worlds, grid topology |
| Env traits & impls | `forge-env`, `forge-env-forge`, `forge-env-mc` | Generic `Env` / `FlatObsEnv` surface and its in-process and Minecraft implementations |
| Training & runner | `forge-agent`, `forge-mc-runner`, `forge-replay` | MCTS/MuZero planning, the end-to-end episode runner, versioned trajectory storage |
| Serving & bindings | `forge-server`, `forge-python`, `forge-wasm` | REST/WebSocket server, PyO3/Gymnasium bindings, WASM module |
| Observability | `forge-observability` | Shared `tracing` init (text/JSON), metrics |
| Evaluation & data | `forge-eval`, `forge-data` | Agent-agnostic evaluation harness + benchmark suite (MLflow/HF exporters) and the training-data pipeline |
| Cloud & edge | `forge-cloud`, `forge-edge` | Distributed cloud/edge storage + worker orchestration for training artifacts |
| Control plane | `forge-mangomas` | MangoMAS multi-agent curriculum / sweep / transfer control plane (Rust twin of `python/forge/mangomas`) |
| Proposal generation | `forge-proposal` | Composable proposal / document-template engine |

---

## Boundaries

### Included

The deterministic simulator (`forge-core`), the env-trait surface
(`forge-env`), MCTS/MuZero training and the episode runner (`forge-agent`,
`forge-mc-runner`), the versioned trajectory/manifest formats (`forge-replay`),
and shared observability (`forge-observability`).

### Deferred (temporary — targeted for v1.0)

These are wanted but not yet built; they are **not** permanent exclusions. From
the README's "What's still out of scope" list:

- Multi-threaded shared `Arc<OnnxMuZeroModel>` reload via `ArcSwap<Sessions>`
  (today's `reload(&mut self)` is borrow-checker-safe for the single-owner
  runner).
- DPO / preference trainer consuming teacher decision traces.
- Complex learned block embeddings (T3 Phase 2 candidate). A simple learned
  table ships today (`nn.Embedding` over the curated index map in
  `configs/minecraft/block_embeddings.toml`, default-on via `use_raw_block_id`);
  the richer representation is what remains deferred.

See [`docs/next_steps.md`](next_steps.md) for live status.

### Permanent non-goals (ratified)

Confirmed by the maintainers. Unlike the deferred list above, these are not
"not yet" — they are out of scope by design:

- **Non-deterministic core physics** — would violate Invariant 6 (determinism is
  the platform's defining guarantee). Backed by the fixed-point physics and
  `rand_pcg` RNG, and by the determinism tests in `crates/forge-core`.
- **Hard-coded, non-config-driven behavior** on any surface — violates Invariant
  5. Backed by the `Default`-impl / `serde`-derived config-struct convention and
  the periodic sweep recorded in
  [`docs/hardcoded-values-audit.md`](hardcoded-values-audit.md). Note this one is
  a review convention, not a CI gate: no job detects a hard-coded literal.

Amending either is a deliberate charter change, made in a PR that says so.

### Deliberate Exceptions

Bounded, on-the-record carve-outs from the invariants below — each exists for a
concrete reason, so that a future reader treats it as a decision, not an
inconsistency:

1. **Wire-I/O allocation in `forge-env-mc::MinecraftEnv`** is exempt from the
   zero-allocation audit. `MinecraftEnv::step_into` reuses the caller's
   observation buffer, but internal WebSocket I/O and JSON deserialization
   still allocate and are unavoidable for a wire-bound env. In-process envs
   (`FlatForgeEnv`) must honour the full buffer-reuse contract.
   (`crates/forge-env-mc/src/mc_env.rs`, `CLAUDE.md` "Zero allocation on hot path".)
2. **`mc-live` does not imply `onnx-reload`.** As of v0.5 the random-baseline
   path only needs the WebSocket client, so it builds without ONNX Runtime.
   Trained mode requires `--features onnx-reload` (or the `mc-live-bundled`
   aggregate); the runner hard-errors with rebuild instructions if trained
   mode is requested without it.
   (`crates/forge-mc-runner/Cargo.toml` `[features]`,
   `crates/forge-mc-runner/src/live.rs`.)
3. **`--random-actions` bypasses MCTS and ONNX entirely.** The random branch is
   taken before any manifest/bundle/ORT load, so the baseline runs on a host
   with no ONNX Runtime installed.
   (`crates/forge-mc-runner/src/live.rs`, `crates/forge-mc-runner/src/random_baseline.rs`.)

---

## Seven Core Invariants

Every change is expected to preserve these. They are reworded for FORGE's RL
workspace but keep the source charter's 1–7 numbering. Each names the code or
CI that **enforces** it — if you change the enforcement, update the invariant in
the same PR.

### 1. Extensibility through registries, not core edits

New rewards, actions, and environments arrive as new registry entries or new
`Env` implementations — never as edits to `forge-core`'s deterministic step
function. Registration is explicit: no auto-registration via static-import side
effects (factories are injected, avoiding ESM TDZ traps).

*Enforced by:* the named-factory reward registry
(`mc-bot/src/reward/index.ts`), the shared action/reward config
(`configs/minecraft/action_map.toml`, `configs/minecraft/rewards.toml`), and the generic
`Env` / `FlatObsEnv` traits (`crates/forge-env/src/env.rs`).

### 2. Versioned, backward-compatible wire protocols

Any change to a trajectory, manifest, or WebSocket schema bumps a version and
preserves the old reader. New format work is additive; older readers must stay
wire-compatible.

*Enforced by:* three independent version constants, each guarding a different
contract — `TRAJECTORY_FORMAT_VERSION` with a one-way `from_v1` migrator (v1
untouched) in `crates/forge-replay/src/v2.rs`; `MANIFEST_SCHEMA_VERSION` plus the
monotonic, no-downgrade `HotReloadWatcher`
(`crates/forge-mc-runner/src/manifest.rs`, `crates/forge-mc-runner/src/hot_reload.rs`);
and `SCHEMA_VERSION` for the WebSocket wire shape
(`crates/forge-env-mc/src/protocol.rs`), checked on handshake in
`crates/forge-env-mc/src/mc_env.rs`. The WS `Hello` / `GridShape` messages use
`serde(default)` fields so legacy flat-only bots stay compatible.

**FORGE-specific strengthening:** configs and wire constants shared across Rust,
JS, and Python compute the same canonical sha256, pinned on every side by paired
`xlang_*_pinned_to_known_good` tests. Drift in one language fails all of the
suites that pin the value, simultaneously:

| Pinned value | Rust | Node | Python |
|---|---|---|---|
| action-map `schema_id` | `crates/forge-env-mc/src/action_map.rs` | `mc-bot/test/schema_id.test.ts` | `tests/python/training/test_muzero_mc_schema_id.py` |
| rewards `schema_id` | `crates/forge-env-mc/src/reward_config.rs` | `mc-bot/test/reward_config.test.ts` | `tests/python/training/test_muzero_mc_schema_id.py` |
| protocol `SCHEMA_VERSION` | `crates/forge-env-mc/src/protocol.rs` | `mc-bot/src/protocol.ts` | — |
| `BLOCK_FEATURE_CHANNELS` | `crates/forge-env-mc/src/protocol.rs` | `mc-bot/src/observation_grid.ts` | — |

Bumping a hash in only two of the three languages breaks CI — update every row
that pins the value in the same change.

### 3. Dependency injection enabling testability without hardware

Every path that touches Minecraft, Docker, or ONNX Runtime must be exercisable
without them. Collaborators are injected, not reached for globally.

*Enforced by:* `--dry-run` with an in-process `StubEnv`
(`crates/forge-mc-runner/src/main.rs`), a scripted mock WebSocket server
(`crates/forge-env-mc/tests/mc_env_mock.rs`), the generic `Runner<E, M>`
(defined in `crates/forge-mc-runner/src/runner.rs`, with env / search / writer /
reload-fn injected at its construction site in
`crates/forge-mc-runner/src/live.rs`), and the `forge-mc-runner-bin` CI smoke
job.

### 4. Stateful I/O isolated to transports; the core stays zero-allocation

In-process env and core code must not heap-allocate on the hot path after
warmup. Allocating, stateful I/O is quarantined behind the transport boundary.

*Enforced by:* the buffer-filling `step_into` / `reset_into` contract
(`crates/forge-env/src/env.rs`, `WorldState::step_into`) and the CI
allocation gate — `crates/forge-bench/src/bin/allocation_audit.rs` +
`benchmarks/runner/check_zero_alloc.py --max-bytes 0`. Transport allocation
lives in `forge-env-mc` and is the subject of Deliberate Exception 1.

### 5. Configuration-driven operation — no hard-coded values

All constants flow through config structs with `Default` impls; runtime knobs
are environment variables and TOML, not literals in code.

*Enforced by:* `Default`-impl, `serde`-derived config structs
(`crates/forge-env-mc/src/config.rs`, `crates/forge-mc-runner/src/config.rs`),
the `FORGE_SERVER_*` / `FORGE_MC_*` / `FORGE_LOG_FORMAT` env-var ladders, and
the TOML configs under `configs/`.

### 6. Determinism, verified by enforced quality gates

Same seed + same actions = identical state. Correctness is defended by CI gates
that must stay green.

*Enforced by:* fixed-point physics + `rand_pcg` RNG, and the CI jobs in
[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) — `fmt`, `clippy`
(`-D warnings`), `test`, `alloc-audit`, `coverage` (tarpaulin), `python-lint`
/ `python-test` (ruff + mypy, pytest), `mc-bot-test` (tsc + Biome +
`node:test`), the `forge-mc-runner-bin` smoke, and `wasm` — which lints and
runs the crate's tests on the wasm32 target, the one place determinism is
verified for the environment the browser demo actually ships to. Coverage thresholds and
lint rules are defined in CI and its config (`.coveragerc`, `pyproject.toml`,
`dashboard/vite.config.ts`) — those files are the source of truth, so this
charter names the gates without pinning numbers that would drift.

`ci.yml` also runs jobs this list deliberately omits, because they are not
blocking gates: advisory checks (`machete`), `workflow_dispatch`-only opt-ins
(`python-test-lmstudio`, `python-test-minecraft-e2e`), and explicitly
non-required jobs (`dashboard-e2e`, `wasm-e2e`). Alongside them,
[`.github/workflows/security.yml`](../.github/workflows/security.yml) runs the
supply-chain and static-analysis layer — `cargo-deny` (policy in `deny.toml`),
`pip-audit`, `npm-audit`, `trivy-fs`, and CodeQL. `cargo-deny` is **advisory
today** (it runs with `|| true` until its baseline is clean); treat a finding
there as work to schedule, not as a broken gate.

### 7. Credentials excluded from repositories

No secrets, tokens, or key material are committed. Configuration templates are
committed; the filled-in secrets are not.

*Enforced by:* `.env` / `.env.local` are git-ignored (`.gitignore`);
`docker/compose.minecraft.env` is explicitly ignored with a committed
`*.example` sibling; `.env.template` is the committed template; and CI reads
secrets only from GitHub Actions secrets/vars, never from committed files.

---

## Development Guidance

- **Read this charter before planning work.** It states the invariants a change
  must preserve and the boundaries it must respect.
- **Track working tasks in [`docs/next_steps.md`](next_steps.md)** — the
  existing roadmap with `[STATUS: LANDED]` tagging and the technical-debt
  table. This charter is the durable layer; `next_steps.md` is the living one.
- **Surface conflicts, don't silently rewrite scope.** If a change needs to
  cross a boundary or bend an invariant, raise it in the PR and amend this
  charter deliberately — adding a Deliberate Exception with a rationale, not an
  unexplained edit.

---

## License

FORGE is licensed under the Apache License 2.0 — see [`LICENSE`](../LICENSE).
