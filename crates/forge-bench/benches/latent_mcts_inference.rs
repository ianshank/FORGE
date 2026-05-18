//! Latent MCTS planning-latency benchmark.
//!
//! Measures per-decision latency of [`LatentMctsSearch::search`] at
//! varying simulation budgets using [`StubLatentModel`] so the bench
//! has no ONNX dependency. The stub returns a uniform policy and zero
//! values, isolating tree-construction overhead from real network
//! inference cost.
//!
//! This is the missing bench called out in the Minecraft RL integration
//! plan v2 §3.4 / `docs/next_steps.md` "Phase 4 — `forge-mc-runner` + ...
//! Criterion bench (NEW — the gap identified in the original audit)".
//!
//! ## Configuration
//!
//! Sweep is driven by env vars so the same harness runs on both
//! reference hardware profiles without recompilation:
//!
//! * `FORGE_BENCH_MCTS_SIMS` — comma-separated sim budgets
//!   (default `1,8,25,50,100,200`; matches the plan).
//! * `FORGE_BENCH_MCTS_OBS_DIM` — flat obs length passed to
//!   `search` (default `960`, matches the FORGE drone obs).
//! * `FORGE_BENCH_MCTS_ACTIONS` — action-space size (default `64`).
//! * `FORGE_BENCH_MCTS_LATENT_DIM` — latent vector size (default `256`).
//!
//! ## Invocation
//!
//! ```sh
//! cargo bench -p forge-bench --bench latent_mcts_inference
//! FORGE_BENCH_MCTS_SIMS=1,50,200 \
//!   cargo bench -p forge-bench --bench latent_mcts_inference
//! ```

use std::sync::Once;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use forge_agent::latent_mcts::model::StubLatentModel;
use forge_agent::latent_mcts::search::{LatentMctsConfig, LatentMctsSearch};

const DEFAULT_SIM_BUDGETS: &[u32] = &[1, 8, 25, 50, 100, 200];
const DEFAULT_OBS_DIM: usize = 960;
const DEFAULT_ACTION_COUNT: u32 = 64;
const DEFAULT_LATENT_DIM: usize = 256;

static INIT_LOGS: Once = Once::new();

fn init_logging() {
    INIT_LOGS.call_once(|| {
        // tracing logs are noise during benches; only enable when the
        // user explicitly sets RUST_LOG.
        if std::env::var_os("RUST_LOG").is_some() {
            let _ = tracing_subscriber::fmt::try_init();
        }
    });
}

/// Parse a comma-separated u32 env var. Returns the parsed values
/// (filtering out non-positive entries silently) or the compile-time
/// default when the variable is unset.
fn sim_budgets() -> Vec<u32> {
    match std::env::var("FORGE_BENCH_MCTS_SIMS") {
        Ok(raw) => raw
            .split(',')
            .filter_map(|s| s.trim().parse::<u32>().ok())
            .filter(|n| *n > 0)
            .collect(),
        Err(_) => DEFAULT_SIM_BUDGETS.to_vec(),
    }
}

fn obs_dim() -> usize {
    std::env::var("FORGE_BENCH_MCTS_OBS_DIM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_OBS_DIM)
}

fn action_count() -> u32 {
    std::env::var("FORGE_BENCH_MCTS_ACTIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_ACTION_COUNT)
}

fn latent_dim() -> usize {
    std::env::var("FORGE_BENCH_MCTS_LATENT_DIM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_LATENT_DIM)
}

/// Build a [`LatentMctsConfig`] with the given simulation budget,
/// noise disabled so the bench is deterministic across runs.
fn config_for_sims(sims: u32) -> LatentMctsConfig {
    let mut cfg = LatentMctsConfig::default();
    cfg.base.num_simulations = sims;
    // Disable exploration noise: the bench should measure raw search
    // cost, not noise sampling, and noise-free runs are reproducible.
    cfg.add_exploration_noise = false;
    cfg
}

fn bench_latent_mcts_search(c: &mut Criterion) {
    init_logging();
    let mut group = c.benchmark_group("latent_mcts_search");
    // PUCT cost grows roughly linearly with sims; declare so Criterion
    // can report per-simulation throughput in addition to wall time.
    let obs_dim_v = obs_dim();
    let actions = action_count();
    let latent = latent_dim();
    let obs = vec![0.5f32; obs_dim_v];
    let budgets = sim_budgets();
    if budgets.is_empty() {
        eprintln!(
            "FORGE_BENCH_MCTS_SIMS was set but produced no usable values; \
             using compile-time defaults"
        );
    }
    let effective = if budgets.is_empty() {
        DEFAULT_SIM_BUDGETS.to_vec()
    } else {
        budgets
    };
    for &sims in &effective {
        let search =
            LatentMctsSearch::new(StubLatentModel::new(actions, latent), config_for_sims(sims));
        group.bench_with_input(BenchmarkId::new("stub_model", sims), &sims, |b, &_sims| {
            b.iter(|| {
                let res = search
                    .search(black_box(&obs))
                    .expect("stub search must succeed");
                black_box(res);
            });
        });
    }
    group.finish();
}

criterion_group!(latent_mcts_inference, bench_latent_mcts_search);
criterion_main!(latent_mcts_inference);
