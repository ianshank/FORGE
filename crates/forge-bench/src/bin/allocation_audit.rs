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
//!     --agents 1,8,16,32,64,128 \
//!     --out /tmp/alloc_audit.json
//! ```
//!
//! # Environment overrides
//!
//! * `FORGE_BENCH_WORLD` -- world side (default 64)
//! * `FORGE_BENCH_SEED`  -- RNG seed (default 42)
//! * `FORGE_BENCH_AGENT_COUNTS` -- comma-separated agent count sweep
//!   (default `1,8,16,32,64,128`); overridden by `--agents`.
//!
//! # Output
//!
//! Emits a JSON report to `--out` (or stdout if omitted) with one row per
//! `(Action variant, agent count)` pair, capturing `total_blocks` and
//! `total_bytes` allocated during the measured step region. Variant labels
//! use the `<name>@n=<count>` form (for example `Move_Up@n=8`). Post-process
//! with `benchmarks/runner/check_zero_alloc.py` to assert the zero-allocation
//! invariant and fail CI on violators.

#![deny(clippy::all)]
#![deny(missing_docs)]

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

use forge_bench::env::{agent_counts, seed_u64, world_side, BENCH_SEED, DEFAULT_AGENT_COUNTS};
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
/// scaling bench default because the audit's per-row cost grows with both
/// the variant count and the agent-count sweep, and a tighter world keeps
/// warm-up cheap. Overridable via `FORGE_BENCH_WORLD`.
const DEFAULT_WORLD_SIDE: u16 = 64;
/// Default warm-up iterations before measurement begins.
const DEFAULT_WARMUP: u64 = 1024;
/// Default measured iterations per variant.
const DEFAULT_ITERS: u64 = 10_000;

/// One row of allocation evidence per `(Action variant, agent count)` pair.
#[derive(Debug, Serialize)]
struct VariantReport {
    /// Human-readable label, formatted as `<base>@n=<agents>`
    /// (for example `Move_Up@n=8`).
    variant: String,
    /// Number of agents stepped in lockstep for this measurement. Captured
    /// as a typed field so downstream tooling can filter without parsing
    /// the variant string.
    num_agents: u32,
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
/// Recognises `--warmup N`, `--iters N`, `--out PATH`, and
/// `--agents N[,N,...]`. Unknown flags emit a warning and are ignored
/// (so CI can pass `--` and positional arguments safely).
///
/// `agents` is `None` when the flag is omitted; resolution to a concrete
/// sweep happens in `main` and falls back to `FORGE_BENCH_AGENT_COUNTS`
/// then `DEFAULT_AGENT_COUNTS`.
struct Args {
    warmup: u64,
    iters: u64,
    out: Option<PathBuf>,
    agents: Option<Vec<u32>>,
}

fn parse_args() -> Args {
    parse_args_from(env::args().skip(1))
}

fn parse_args_from<I: IntoIterator<Item = String>>(argv: I) -> Args {
    let mut warmup = DEFAULT_WARMUP;
    let mut iters = DEFAULT_ITERS;
    let mut out: Option<PathBuf> = None;
    let mut agents: Option<Vec<u32>> = None;

    let mut argv = argv.into_iter();
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
            "--agents" => {
                if let Some(raw) = argv.next() {
                    agents = Some(parse_agents_arg(&raw));
                }
            }
            other => warn!(flag = other, "unknown argument ignored"),
        }
    }
    Args {
        warmup,
        iters,
        out,
        agents,
    }
}

/// Parses the `--agents` value (comma-separated u32 list).
///
/// Mirrors the convention in `forge_bench::env::agent_counts`: invalid or
/// zero tokens are dropped with a warning. Returns an empty `Vec` if every
/// token fails — `main` then falls back to the env-var / default chain.
fn parse_agents_arg(raw: &str) -> Vec<u32> {
    raw.split(',')
        .filter_map(|s| {
            let trimmed = s.trim();
            match trimmed.parse::<u32>() {
                Ok(n) if n > 0 => Some(n),
                _ => {
                    warn!(token = trimmed, "ignoring invalid --agents token");
                    None
                }
            }
        })
        .collect()
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
///
/// `actions` must contain exactly `num_agents` entries; every agent steps
/// with the same action so the measurement isolates that variant's hot
/// path under fan-out (versus the throughput bench, which fans actions
/// across `Direction` for diversity).
fn measure_variant(
    label: &str,
    num_agents: u32,
    actions: &[Action],
    warmup: u64,
    iters: u64,
    config: ForgeConfig,
) -> VariantReport {
    debug_assert_eq!(
        actions.len(),
        num_agents as usize,
        "action vector must have one entry per agent"
    );
    let mut world = WorldState::new(config).expect("WorldState::new must succeed for audit config");
    let mut result = StepResult::default();

    // Warm caches so allocator churn from first-touch is excluded. The
    // warm-up also fills `world.step_actions`, `world.validated_actions`,
    // `world.physics_scratch`, and the inner `Vec`s of `result` so the
    // measured region exercises the steady-state, capacity-stable path.
    for _ in 0..warmup {
        world.step_into(actions, &mut result);
    }

    let profiler = dhat::Profiler::new_heap();
    for _ in 0..iters {
        world.step_into(actions, &mut result);
    }
    let stats = dhat::HeapStats::get();
    drop(profiler); // flush any files dhat writes, though we only read stats.

    VariantReport {
        variant: label.to_string(),
        num_agents,
        iters,
        total_blocks: stats.total_blocks,
        total_bytes: stats.total_bytes,
        peak_live_bytes: stats.max_bytes as u64,
    }
}

/// Measures allocation on the generic `forge_env::Env::step_into` hot-path
/// using [`forge_env_forge::WorldEnv`].
fn measure_env_trait_variant(
    label: &str,
    action: Action,
    warmup: u64,
    iters: u64,
    config: ForgeConfig,
) -> VariantReport {
    use forge_env::{Env, StepOutput};
    use forge_env_forge::WorldEnv;

    let mut env = WorldEnv::new(config).expect("WorldEnv::new must succeed for audit config");
    let mut step_out = StepOutput::default();

    for _ in 0..warmup {
        env.step_into(action.clone(), &mut step_out)
            .expect("WorldEnv::step_into must succeed");
    }

    let profiler = dhat::Profiler::new_heap();
    for _ in 0..iters {
        env.step_into(action.clone(), &mut step_out)
            .expect("WorldEnv::step_into must succeed");
    }
    let stats = dhat::HeapStats::get();
    drop(profiler);

    VariantReport {
        variant: label.to_string(),
        num_agents: 1,
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

    // Resolution order: explicit `--agents` flag, then
    // `FORGE_BENCH_AGENT_COUNTS`, then the canonical default sweep that
    // matches `multi_agent_scaling.rs`. An explicit-but-empty `--agents`
    // (e.g. `--agents ,,`) falls through to the env/default chain so a
    // typo never silently produces a zero-row report.
    let agents: Vec<u32> = args
        .agents
        .clone()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| agent_counts(DEFAULT_AGENT_COUNTS));

    info!(
        warmup = args.warmup,
        iters = args.iters,
        world_side = world,
        seed = audit_seed,
        agents = ?agents,
        out = ?args.out,
        "running allocation audit"
    );

    let base_variants = audit_variants();
    let mut variants: Vec<VariantReport> = Vec::with_capacity(base_variants.len() * agents.len());
    for &num_agents in &agents {
        let mut cfg = ForgeConfig::default();
        cfg.world.width = world;
        cfg.world.height = world;
        cfg.world.seed = audit_seed;
        cfg.agents.num_agents = num_agents;
        cfg.task.max_episode_length = 0;

        for (name, action) in &base_variants {
            let label = format!("{name}@n={num_agents}");
            info!(variant = %label, "measuring");
            let actions = vec![action.clone(); num_agents as usize];
            variants.push(measure_variant(
                &label,
                num_agents,
                &actions,
                args.warmup,
                args.iters,
                cfg.clone(),
            ));
        }

        if num_agents == 1 {
            let env_label = "EnvTrait_WorldEnv_Move_Up@n=1";
            info!(variant = %env_label, "measuring generic Env trait allocation");
            variants.push(measure_env_trait_variant(
                env_label,
                Action::Move(Direction::Up),
                args.warmup,
                args.iters,
                cfg.clone(),
            ));
        }
    }

    // Deliberately a stable identifier rather than `env::current_exe()`:
    // committing the latter into `benchmarks/baselines/<profile>/` would
    // bake in the local `target/release/` path of whichever machine
    // produced the snapshot, creating noisy diffs on regeneration.
    let report = AuditReport {
        binary: "allocation_audit".to_string(),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn parse_args_default_when_argv_empty() {
        let args = parse_args_from(argv(&[]));
        assert_eq!(args.warmup, DEFAULT_WARMUP);
        assert_eq!(args.iters, DEFAULT_ITERS);
        assert!(args.out.is_none());
        assert!(
            args.agents.is_none(),
            "no --agents flag => fall back to env-var/default chain in main"
        );
    }

    #[test]
    fn parse_args_explicit_agents_flag_overrides_default() {
        let args = parse_args_from(argv(&["--agents", "1,8,32"]));
        assert_eq!(args.agents.as_deref(), Some(&[1u32, 8, 32][..]));
    }

    #[test]
    fn parse_args_rejects_zero_and_invalid_agent_tokens() {
        // "0" is dropped (zero is invalid for num_agents; matches env.rs convention).
        // "abc" is dropped (parse failure).
        // "16" survives.
        let args = parse_args_from(argv(&["--agents", "0,abc,16"]));
        assert_eq!(args.agents.as_deref(), Some(&[16u32][..]));
    }

    #[test]
    fn parse_args_all_flags_combined() {
        let args = parse_args_from(argv(&[
            "--warmup",
            "256",
            "--iters",
            "1000",
            "--agents",
            "1,8",
            "--out",
            "/tmp/x.json",
        ]));
        assert_eq!(args.warmup, 256);
        assert_eq!(args.iters, 1000);
        assert_eq!(args.agents.as_deref(), Some(&[1u32, 8][..]));
        assert_eq!(
            args.out.as_deref(),
            Some(std::path::Path::new("/tmp/x.json"))
        );
    }

    #[test]
    fn parse_agents_arg_handles_whitespace_and_empty_tokens() {
        // Empty tokens (from `,,`) and whitespace-padded numbers parse correctly.
        let parsed = parse_agents_arg(" 1 , , 8 ");
        assert_eq!(parsed, vec![1, 8]);
    }
}
