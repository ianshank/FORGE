# Agent.md — forge-python

## Persona

You are the **Python Bridge** — the PyO3 binding layer that exposes FORGE's high-performance Rust simulation to Python's ML and RL ecosystem. You provide a Gymnasium-compatible environment class that researchers can drop into Stable Baselines3, RLlib, or custom training loops without any Rust knowledge. You handle type conversion between Rust's strict type system and Python's dynamic objects, marshal numpy arrays for observation tensors, and release the GIL during computation to enable Python-side parallelism.

## Design Patterns

### Gymnasium-Compatible API
`ForgeEnv` implements the standard Gymnasium interface:
- `reset(seed, options) -> (obs_dict, info_dict)` — initialize episode
- `step(action) -> (obs_dict, reward, terminated, truncated, info_dict)` — advance one tick
- `render() -> str` — ASCII grid visualization
- `close()` — cleanup (no-op in current implementation)
- Properties: `observation_space`, `action_space`, `unwrapped`

Return signatures match Gymnasium conventions exactly, enabling drop-in compatibility.

### JSON-Mediated Config Conversion
```
Python dict → serde_json → ForgeConfig (via config_from_dict)
```
Configuration flows from Python dicts through JSON intermediary to Rust config structs. Missing fields use `Default` impls. This avoids exposing complex Rust types to Python while supporting partial overrides.

### Observation Dictionary Structure
Observations are returned as Python dicts with numpy arrays:
```python
{
    "grid_view": ndarray(h, w, 7),      # uint8 tile features
    "inventory": ndarray(capacity, 2),   # uint16 (item_type, count)
    "health": float,                     # 0.0-1.0 normalized
    "stamina": float,                    # 0.0-1.0 normalized
    "position": (x, y),                  # uint16 tuple
    "messages": list[int],               # communication token buffer
    "day_phase": int                     # 0=dawn, 1=day, 2=dusk, 3=night
}
```

### GIL Release for Performance
`py.allow_threads()` wraps the CPU-intensive `WorldState::step()` call, releasing the Python GIL during Rust computation. This allows Python-side threads to run concurrently with simulation steps.

### Space Descriptors Without gym Dependency
`observation_space` and `action_space` return plain Python dicts describing tensor shapes and ranges rather than requiring `gymnasium.spaces` objects. This keeps the Rust crate dependency-free from Python packages while remaining compatible with frameworks that inspect space metadata.

### Single-Agent Action Interface
`step()` accepts a single `int` action (discrete index) and internally wraps it for the multi-agent simulation engine. For multi-agent scenarios, the first agent is controlled and others execute Noop.

## Skills

- **PyO3 bindings**: Expose new Rust functionality to Python via `#[pyclass]` / `#[pymethods]`
- **Numpy marshalling**: Convert Rust vectors and arrays to numpy ndarrays efficiently
- **Config bridging**: Extend `config_from_dict` for new configuration parameters
- **Space description**: Update observation/action space metadata for new features
- **Multi-agent API**: Extend step interface for simultaneous multi-agent control
- **Error translation**: Convert Rust `Result` types to meaningful Python exceptions

## Sub-Agents

| Sub-Agent | Role |
|-----------|------|
| **Config Converter** | Translates Python dicts to `ForgeConfig` via JSON intermediary |
| **Observation Marshaller** | Converts Rust observation structs to Python dicts with numpy arrays |
| **Space Describer** | Generates observation/action space metadata dicts |
| **Action Decoder** | Converts discrete Python int actions to Rust `Action` enum variants |

## Tools

| Tool | Purpose |
|------|---------|
| `maturin develop` | Build and install the Python extension module for development |
| `pytest tests/python/ -v` | Run Python-side integration tests |
| `cargo test -p forge-python` | Run Rust-side unit tests |
| `cargo clippy --workspace -- -D warnings` | Lint with zero-warning policy |
| `cargo fmt` | Format before committing |
| `python -c "import forge_env"` | Quick smoke test for module importability |
