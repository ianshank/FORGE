//! Allocation audit harness for `WorldState::step()`.
//!
//! Drives every representative [`Action`] variant through a warm `WorldState`
//! under the `dhat` heap profiler to verify (or refute) the "zero allocation
//! on hot path" claim in `README.md` and `docs/architecture.md`.
//!
//! This binary is behind the `dhat-heap` feature because `dhat` installs a
//! custom `#[global_allocator]` and must not be linked into ordinary builds.
//!
//! # Invocation
//!
//! ```sh
//! cargo run -p forge-bench --bin allocation_audit \
//!     --features dhat-heap --release -- \
//!     --warmup 1024 --iters 10000 \
//!     --out /tmp/alloc_audit.json
//! ```
//!
//! # Environment overrides
//!
//! * `FORGE_BENCH_WORLD` -- world side (default 64)
//! * `FORGE_BENCH_SEED`  -- RNG seed (default 42)
//!
//! # Output
//!
//! Emits a JSON report to `--out` (or stdout if omitted) with one row per
//! `Action` variant, capturing `total_blocks` and `total_bytes` allocated
//! during the measured step region. Post-process with
//! `benchmarks/runner/check_zero_alloc.py` to assert the zero-allocation
//! invariant and fail CI on violators.

#![deny(clippy::all)]
#![deny(missing_docs)]

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

use forge_bench::env::{seed_u64, world_side, BENCH_SEED};
use forge_core::WorldState;
use forge_types::config::ForgeConfig;
use forge_types::grid::Direction;
use forge_types::observation::StepResult;
use forge_types::Action;
use serde::Serialize;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

/// Default world side length for the audit run — kept smaller than the
/// scaling bench default because the audit measures a single agent and a
/// tighter world keeps warm-up cheap. Overridable via `FORGE_BENCH_WORLD`.
const DEFAULT_WORLD_SIDE: u16 = 64;
/// Default warm-up iterations before measurement begins.
const DEFAULT_WARMUP: u64 = 1024;
/// Default measured iterations per variant.
const DEFAULT_ITERS: u64 = 10_000;

/// One row of allocation evidence per [`Action`] variant.
#[derive(Debug, Serialize)]
struct VariantReport {
    /// Human-readable Action variant label.
    variant: String,
    /// Iterations executed inside the measured region.
    iters: u64,
    /// Total heap blocks allocated during the measured region.
    total_blocks: u64,
    /// Total heap bytes allocated during the measured region.
    total_bytes: u64,
    /// Peak live heap bytes observed during the measured region.
    /// Named `peak_live_bytes` (not `max_bytes`) to avoid ambiguity with
    /// the `--max-bytes` threshold flag on the post-processor.
    peak_live_bytes: u64,
}

/// Full audit report written to `--out`.
#[derive(Debug, Serialize)]
struct AuditReport {
    /// Path to the running binary for traceability.
    binary: String,
    /// World side length used during the run.
    world_side: u16,
    /// RNG seed used during the run.
    seed: u64,
    /// Warm-up iterations executed before each measured region.
    warmup: u64,
    /// Runtime-detected target description (`std::env::consts::{OS,ARCH}`).
    target: String,
    /// Per-variant allocation rows.
    variants: Vec<VariantReport>,
}

/// Minimal argv parser to avoid pulling `clap` into the audit binary.
///
/// Recognises `--warmup N`, `--iters N`, `--out PATH`. Unknown flags
/// emit a warning and are ignored (so CI can pass `--` and positional
/// arguments safely).
struct Args {
    warmup: u64,
    iters: u64,
    out: Option<PathBuf>,
}

fn parse_args() -> Args {
    let mut warmup = DEFAULT_WARMUP;
    let mut iters = DEFAULT_ITERS;
    let mut out: Option<PathBuf> = None;

    let mut argv = env::args().skip(1);
    while let Some(flag) = argv.next() {
        match flag.as_str() {
            "--warmup" => {
                if let Some(v) = argv.next().and_then(|v| v.parse::<u64>().ok()) {
                    warmup = v;
                }
            }
            "--iters" => {
                if let Some(v) = argv.next().and_then(|v| v.parse::<u64>().ok()) {
                    iters = v;
                }
            }
            "--out" => {
                out = argv.next().map(PathBuf::from);
            }
            other => warn!(flag = other, "unknown argument ignored"),
        }
    }
    Args { warmup, iters, out }
}

/// Representative [`Action`] variants covering the hot-path surface.
///
/// Includes every base variant plus two slotted ones (`Drop`, `Use`) so the
/// audit catches any branch that allocates in slot handling. Drone and
/// agricultural variants are excluded because they require feature flags in
/// `ForgeConfig` and are covered by dedicated suites.
fn audit_variants() -> Vec<(&'static str, Action)> {
    vec![
        ("Noop", Action::Noop),
        ("Move_Up", Action::Move(Direction::Up)),
        ("Move_Down", Action::Move(Direction::Down)),
        ("Move_Left", Action::Move(Direction::Left)),
        ("Move_Right", Action::Move(Direction::Right)),
        ("PickUp", Action::PickUp),
        ("Drop_0", Action::Drop(0)),
        ("Use_0", Action::Use(0)),
        ("Craft_0", Action::Craft(0)),
        ("Push_Right", Action::Push(Direction::Right)),
        ("Interact", Action::Interact),
        ("Communicate_0", Action::Communicate(0)),
    ]
}

/// Runs a warm-up of `warmup` iterations, then a measured `iters` inside
/// a fresh `dhat::Profiler` so only the measured region counts.
///
/// Uses [`WorldState::step_into`] with a single reused [`StepResult`]
/// buffer — this is the **zero-allocation hot-path entry point**. The
/// convenience [`WorldState::step`] always allocates a fresh result and
/// is therefore unsuitable for measuring the contract.
fn measure_variant(
    name: &str,
    action: &Action,
    warmup: u64,
    iters: u64,
    config: ForgeConfig,
) -> VariantReport {
    let mut world = WorldState::new(config).expect("WorldState::new must succeed for audit config");
    let actions = vec![action.clone()];
    let mut result = StepResult::default();

    // Warm caches so allocator churn from first-touch is excluded. The
    // warm-up also fills `world.step_actions`, `world.validated_actions`,
    // `world.physics_scratch`, and the inner `Vec`s of `result` so the
    // measured region exercises the steady-state, capacity-stable path.
    for _ in 0..warmup {
        world.step_into(&actions, &mut result);
    }

    let profiler = dhat::Profiler::new_heap();
    for _ in 0..iters {
        world.step_into(&actions, &mut result);
    }
    let stats = dhat::HeapStats::get();
    drop(profiler); // flush any files dhat writes, though we only read stats.

    VariantReport {
        variant: name.to_string(),
        iters,
        total_blocks: stats.total_blocks,
        total_bytes: stats.total_bytes,
        peak_live_bytes: stats.max_bytes as u64,
    }
}

fn main() -> ExitCode {
    // Honour RUST_LOG if set; default to info on the audit module.
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("allocation_audit=info,forge_core=warn"));
    tracing_subscriber::fmt().with_env_filter(filter).init();

    let args = parse_args();
    let world = world_side(DEFAULT_WORLD_SIDE);
    let audit_seed = seed_u64(BENCH_SEED);

    info!(
        warmup = args.warmup,
        iters = args.iters,
        world_side = world,
        seed = audit_seed,
        out = ?args.out,
        "running allocation audit"
    );

    let mut cfg_proto = ForgeConfig::default();
    cfg_proto.world.width = world;
    cfg_proto.world.height = world;
    cfg_proto.world.seed = audit_seed;
    cfg_proto.agents.num_agents = 1;
    cfg_proto.task.max_episode_length = 0;

    let variants: Vec<VariantReport> = audit_variants()
        .into_iter()
        .map(|(name, action)| {
            info!(variant = name, "measuring");
            measure_variant(name, &action, args.warmup, args.iters, cfg_proto.clone())
        })
        .collect();

    let report = AuditReport {
        binary: env::current_exe()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "<unknown>".into()),
        world_side: world,
        seed: audit_seed,
        warmup: args.warmup,
        target: format!("{}-{}", env::consts::ARCH, env::consts::OS),
        variants,
    };

    let json = match serde_json::to_string_pretty(&report) {
        Ok(s) => s,
        Err(e) => {
            warn!(error = %e, "failed to serialise audit report");
            return ExitCode::from(2);
        }
    };

    match args.out.as_ref() {
        Some(path) => match File::create(path).and_then(|mut f| f.write_all(json.as_bytes())) {
            Ok(()) => info!(path = %path.display(), "wrote audit report"),
            Err(e) => {
                warn!(error = %e, path = %path.display(), "failed to write audit report");
                return ExitCode::from(3);
            }
        },
        None => println!("{json}"),
    }

    ExitCode::SUCCESS
}
