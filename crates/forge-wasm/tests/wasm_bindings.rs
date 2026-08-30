//! Runtime tests for the `forge-wasm` bindings, executed inside a real
//! WebAssembly runtime by `wasm-pack test --node` (CI: the `wasm` job, via
//! `scripts/wasm_test_node.sh`).
//!
//! These deliberately duplicate a *subset* of the unit tests in `src/lib.rs`.
//! The point is not extra coverage of `forge-core` -- it is that nothing else
//! in the workspace ever executes this crate's code on the target it actually
//! ships to. `wasm32-unknown-unknown` has a 32-bit `usize` and a different
//! float ABI, and its panics are traps rather than unwinds, so "passes on
//! x86_64" is not evidence about the browser.
//!
//! Being an integration test, this file sees only the public API -- which is
//! exactly the surface a JavaScript consumer sees. The unit tests keep their
//! white-box access to `env.world` / `env.config` and stay where they are.
//!
//! NOT tested here: the `BigInt` marshalling of `reset`'s seed. `Option<u64>`
//! is a plain `u64` on the Rust side; the JS coercion exists only in the
//! wasm-bindgen-generated glue, so it is only reachable from JavaScript. That
//! contract is pinned by the Playwright suite in `tests/web-e2e/`.
//!
//! The whole file compiles away on any non-wasm target, so
//! `cargo test --workspace` on the host is unaffected.
#![cfg(target_arch = "wasm32")]

use forge_wasm::ForgeWasmEnv;
use wasm_bindgen_test::wasm_bindgen_test;

/// Parse a JSON string the bindings returned, failing loudly if it is not JSON.
fn parse(json: &str) -> serde_json::Value {
    serde_json::from_str(json).expect("bindings must return valid JSON")
}

/// Construct the demo's environment: an empty config, i.e. `ForgeConfig::default()`.
/// This is exactly what `web/app.js` does with `DEMO_CONFIG = {}`.
fn demo_env() -> ForgeWasmEnv {
    ForgeWasmEnv::try_new("").expect("default config must construct")
}

#[wasm_bindgen_test]
fn wasm_constructs_with_defaults() {
    // The broadest signal available: reaching this line means the module
    // instantiated, the wasm-bindgen glue linked, and `WorldState::new` ran
    // worldgen to completion under wasm.
    let env = demo_env();
    assert!(!env.render_ascii().is_empty());
}

#[wasm_bindgen_test]
fn wasm_reset_and_step_round_trip_json() {
    let mut env = demo_env();

    let reset = parse(&env.reset(Some(42)));
    assert!(reset.get("observations").is_some());
    assert!(reset.get("rewards").is_some());
    assert_eq!(reset["terminated"].as_bool(), Some(false));

    let step = parse(&env.step(0)); // Noop
    assert!(step.get("observations").is_some());
    assert!(step.get("rewards").is_some());
}

#[wasm_bindgen_test]
fn wasm_determinism_same_seed_same_actions() {
    // CHARTER Invariant 6, verified on the target the browser demo ships.
    // Nothing else in the workspace checks determinism on wasm32, and a
    // 32-bit `usize` plus a different float ABI are exactly the kind of thing
    // that could break it without touching the x86_64 result.
    let config = r#"{"world":{"width":16,"height":16,"seed":42},"agents":{"num_agents":1}}"#;
    let mut a = ForgeWasmEnv::try_new(config).expect("config must construct");
    let mut b = ForgeWasmEnv::try_new(config).expect("config must construct");

    a.reset(Some(42));
    b.reset(Some(42));

    for action in [0, 1, 2, 3, 4, 0, 1, 2] {
        assert_eq!(
            a.step(action),
            b.step(action),
            "determinism broken on wasm32 at action {action}"
        );
    }
}

#[wasm_bindgen_test]
fn wasm_render_ascii_nonempty_contains_agent() {
    // `render_ascii` is the demo's only visual output -- `web/app.js` writes it
    // straight into `<pre id="grid">`.
    let grid = demo_env().render_ascii();
    assert!(!grid.is_empty());
    assert!(grid.contains('A'), "grid should contain an agent: {grid}");
}

#[wasm_bindgen_test]
fn wasm_action_space_n_is_positive() {
    // `web/app.js`'s `readActionCount` parses this and uses `n` as the upper
    // bound for action selection; a zero or missing `n` silently reduces the
    // demo to Noop forever.
    let space = parse(&demo_env().action_space_json());
    let n = space["n"].as_u64().expect("action space must expose `n`");
    assert!(n > 0, "action space must be non-empty, got {n}");
    assert!(space.get("action_names").is_some());
}

#[wasm_bindgen_test]
fn wasm_constructor_returns_error_for_malformed_config() {
    // Pins the Stage-2 change on the target that matters: `new` is the
    // `#[wasm_bindgen(constructor)]`, so this exercises the same path a JS
    // `new ForgeWasmEnv(...)` takes, including `console_error_panic_hook`
    // installation.
    //
    // Deliberately NOT written as a `#[should_panic]` test. Doing so would put
    // a panicking test in the same binary as `set_once()`, which trips
    // "cannot modify the panic hook from a panicking thread"
    // (wasm-bindgen#3779). Returning `Result` means we never have to.
    assert!(
        ForgeWasmEnv::new("{ not json").is_err(),
        "malformed config JSON must be reported as an error, not a trap"
    );
    assert!(
        ForgeWasmEnv::new("").is_ok(),
        "an empty config must still select ForgeConfig::default()"
    );
}
