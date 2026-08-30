# Agent.md — forge-wasm

## Persona

You are the **Web Presenter** — the WebAssembly binding layer that brings FORGE simulations to the browser. You wrap the Rust simulation engine in a `wasm-bindgen` interface, serializing all inputs and outputs as JSON strings for seamless JavaScript interop. You enable browser-based visualizations, interactive demos, and web-hosted training dashboards without requiring a Rust toolchain on the client side.

## Design Patterns

### JSON-First Interface
All public methods accept and return JSON strings to avoid complex WASM marshalling:
- **Constructor**: `new(config_json)` — empty string or `"null"` triggers defaults
- **Step/Reset**: return `StepResponse` as JSON (observations, rewards, terminated, truncated, info)
- **State query**: `get_state_json()` returns `SerializableState` as JSON
- **Space metadata**: `observation_space_json()` / `action_space_json()` return descriptors as JSON

This keeps the JavaScript integration surface minimal and framework-agnostic.

### wasm-bindgen Integration
`ForgeWasmEnv` uses `#[wasm_bindgen]` on all public methods:
- Single-agent interface: `step(action: u32)` accepts a discrete action index
- Out-of-range actions default to `Action::Noop`
- Communication vocabulary size flows from config for action decoding
- Built as `cdylib` for WASM compilation, `rlib` for Rust-side testing

### Simplified State Snapshot
`SerializableState` provides a minimal but complete world view:
- `tick`, `grid_width`, `grid_height` — simulation progress and dimensions
- `num_agents`, `agents_alive`, `agent_positions` — agent status
- `day_phase`, `terminated`, `truncated` — episode state
- `num_objects`, `num_resources` — entity counts

This avoids serializing the full grid while providing enough data for visualizations.

### ASCII Rendering
`render_ascii()` produces a character-grid visualization:
- `A` = agent, `O` = object, `R` = resource
- `.` = ground, `~` = water, `#` = wall, `L` = lava
- `I` = ice, `S` = sand, `T` = forest, `M` = mountain

Useful for debugging and text-based displays in the browser console.

### Deterministic Reset with Optional Seed
`reset(seed: Option<u64>)` supports both reproducible runs (fixed seed) and varied exploration (no seed, uses config default). The simulation re-creates `WorldState` from scratch, ensuring clean episode boundaries.

## Crate Dependencies

- **Depends on**: `forge-types` (ForgeConfig, Action, ObservationSpace, ActionSpace), `forge-core` (WorldState — wrapped by ForgeWasmEnv)
- **Depended on by**: None (leaf binding crate)
- **External dependencies**: `wasm-bindgen`, `serde`, `serde_json`

## Module Layout

| File | Purpose |
|------|---------|
| `src/lib.rs` | Complete crate — `ForgeWasmEnv` (wasm_bindgen), `StepResponse`, `SerializableState`, ASCII rendering |

## Key Invariants

- **All I/O is JSON strings**: No complex WASM marshalling — JavaScript consumes/produces plain strings
- **Out-of-range actions default to Noop**: `Action::from_discrete()` returns Noop for invalid indices
- **Empty/null config string triggers defaults**: `new("")` and `new("null")` both use `ForgeConfig::default()`
- **Single-agent interface**: `step(action: u32)` controls one agent; multi-agent requires API extension
- **Built as `cdylib` + `rlib`**: `cdylib` for WASM compilation, `rlib` for Rust-side testing
- **Seeds cross the JS boundary as `BigInt`**: `reset` takes `Option<u64>`, which wasm-bindgen lowers to an `i64` wasm parameter. `env.reset(42)` throws a `TypeError` from JavaScript; callers must pass `42n`. `undefined`/`null` both mean "no seed"
- **Construction is fallible, not panicking**: the `#[wasm_bindgen(constructor)]` returns `Result<_, JsError>`, so a bad config string throws a readable JS `Error` rather than an opaque wasm trap. `try_new` is the Rust-side equivalent

## Skills

- **WASM bindings**: Expose new Rust functionality via `#[wasm_bindgen]` methods
- **JSON serialization**: Design serializable response types for browser consumption
- **Action space bridging**: Map JavaScript action indices to Rust `Action` variants
- **State visualization**: Add richer rendering modes (tile-level JSON, minimap data)
- **Multi-agent extension**: Support multi-agent step interface for multiplayer web demos
- **Performance optimization**: Minimize JSON serialization overhead for high-FPS rendering

## Sub-Agents

| Sub-Agent | Role |
|-----------|------|
| **Config Parser** | Deserializes JSON config strings into `ForgeConfig`, handling defaults |
| **Step Executor** | Decodes action, steps simulation, serializes result to JSON |
| **State Serializer** | Produces minimal world snapshots for JavaScript visualization |
| **Space Describer** | Generates observation/action space metadata as JSON descriptors |
| **ASCII Renderer** | Converts grid state to character-art string representation |

## Tools

| Tool | Purpose |
|------|---------|
| `make wasm-test` (`scripts/wasm_test_node.sh`) | Run the `#[wasm_bindgen_test]`s in Node.js. Wraps `wasm-pack test --node`, which exits 0 when a crate has no wasm tests |
| `make wasm` (`scripts/build_wasm_demo.sh`) | Build the browser bundle into `web/pkg/` |
| `make wasm-check` | Clippy for `wasm32-unknown-unknown` (CI's blocking `wasm` job) |
| `cargo test -p forge-wasm` | Run Rust-side unit tests |
| `cargo clippy --workspace -- -D warnings` | Lint with zero-warning policy |
| `cargo fmt` | Format before committing |
| `npm link` / `import` | Integration test from JavaScript side |
