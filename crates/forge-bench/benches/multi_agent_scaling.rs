//! Multi-agent scaling benchmarks for the FORGE simulation engine.
//!
//! Extends the single-agent coverage in `step_throughput.rs` to large
//! agent counts (up to 128) so that Phase 0 audit can verify whether
//! the headline "130K+ steps/sec" claim degrades under realistic
//! multi-agent load — this is the bench that exposes scaling behaviour
//! hidden by the existing 1/2/4/8 range.
//!
//! # Configuration
//!
//! All sizing is driven by environment variables with documented defaults,
//! so the same harness runs on both Reference Hardware A and B without
//! recompilation:
//!
//! * `FORGE_BENCH_WORLD` -- world side length in tiles (default 128)
//! * `FORGE_BENCH_AGENT_COUNTS` -- comma-separated agent counts
//!   (default `1,8,16,32,64,128`)
//! * `FORGE_BENCH_SEED` -- deterministic RNG seed (default 42)
//!
//! # Invocation
//!
//! ```sh
//! cargo bench -p forge-bench --bench multi_agent_scaling
//! FORGE_BENCH_WORLD=256 cargo bench -p forge-bench --bench multi_agent_scaling
//! ```

use std::sync::Once;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use forge_bench::env::{agent_counts, seed_u64, world_side, BENCH_SEED};
use forge_core::WorldState;
use forge_types::config::{ForgeConfig, GridType};
use forge_types::grid::{Direction, HexDirection};
use forge_types::Action;
use tracing::{debug, info, warn};
use tracing_subscriber::EnvFilter;

static TRACING_INIT: Once = Once::new();

/// Initialises a `tracing_subscriber` once per process so `info!`/`warn!`
/// inside the bench surface when `RUST_LOG` is set. A no-op if a subscriber
/// is already installed — calling twice is safe.
fn init_tracing() {
    TRACING_INIT.call_once(|| {
        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("multi_agent_scaling=warn"));
        let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
    });
}

/// Default world side length for the scaling sweep — chosen larger than the
/// existing `step_throughput.rs` default so cache-resident effects do not
/// mask multi-agent scaling cost. Overridable via `FORGE_BENCH_WORLD`.
const DEFAULT_WORLD_SIDE: u16 = 128;
/// Default agent-count sweep, designed to expose scaling inflection points
/// beyond the existing 1/2/4/8 range. Overridable via
/// `FORGE_BENCH_AGENT_COUNTS`.
const DEFAULT_AGENT_COUNTS: &[u32] = &[1, 8, 16, 32, 64, 128];

/// Builds a deterministic config for the scaling sweep with the requested
/// topology and agent count.
fn make_config(grid: GridType, num_agents: u32) -> ForgeConfig {
    let mut config = ForgeConfig::default();
    config.world.grid_type = grid;
    config.world.width = world_side(DEFAULT_WORLD_SIDE);
    config.world.height = world_side(DEFAULT_WORLD_SIDE);
    config.world.seed = seed_u64(BENCH_SEED);
    config.agents.num_agents = num_agents;
    // Disable episode truncation so the step is measured under steady state
    // rather than occasionally paying episode-reset cost.
    config.task.max_episode_length = 0;
    config
}

/// Builds a deterministic square-grid action vector for the given agent count.
///
/// Rotates through the four cardinal directions so that not every agent moves
/// into the same tile, which would otherwise serialise through collision
/// resolution and bias the measurement.
fn square_actions(num_agents: u32) -> Vec<Action> {
    (0..num_agents)
        .map(|i| {
            Direction::from_index((i % 4) as u8)
                .map(Action::Move)
                .unwrap_or(Action::Noop)
        })
        .collect()
}

/// Builds a deterministic hex-grid action vector for the given agent count.
fn hex_actions(num_agents: u32) -> Vec<Action> {
    (0..num_agents)
        .map(|i| {
            HexDirection::from_index((i % 6) as u8)
                .map(Action::MoveHex)
                .unwrap_or(Action::Noop)
        })
        .collect()
}

/// Runs the scaling sweep for a single grid topology.
///
/// Uses `Throughput::Elements(num_agents)` so Criterion reports
/// steps-per-second normalised by agent count, which is the most
/// interpretable metric for "how much does adding agents cost?".
fn bench_scaling(c: &mut Criterion, group_name: &str, grid: GridType) {
    init_tracing();
    let side = world_side(DEFAULT_WORLD_SIDE);
    let counts = agent_counts(DEFAULT_AGENT_COUNTS);
    info!(
        group = group_name,
        world_side = side,
        counts = ?counts,
        "starting multi-agent scaling bench"
    );

    let mut group = c.benchmark_group(group_name);
    for &num_agents in &counts {
        let config = make_config(grid, num_agents);
        let mut world = match WorldState::new(config) {
            Ok(w) => w,
            Err(e) => {
                warn!(
                    group = group_name,
                    num_agents,
                    error = %e,
                    "skipping: WorldState::new failed"
                );
                continue;
            }
        };
        let actions = match grid {
            GridType::Square => square_actions(num_agents),
            GridType::Hex => hex_actions(num_agents),
        };
        debug!(num_agents, actions = actions.len(), "configured bench");
        group.throughput(Throughput::Elements(u64::from(num_agents)));
        group.bench_with_input(
            BenchmarkId::new("num_agents", num_agents),
            &num_agents,
            |b, _| {
                b.iter(|| {
                    black_box(world.step(&actions));
                });
            },
        );
    }
    group.finish();
}

fn bench_square(c: &mut Criterion) {
    bench_scaling(c, "multi_agent_scaling_square", GridType::Square);
}

fn bench_hex(c: &mut Criterion) {
    bench_scaling(c, "multi_agent_scaling_hex", GridType::Hex);
}

criterion_group!(benches, bench_square, bench_hex);
criterion_main!(benches);
