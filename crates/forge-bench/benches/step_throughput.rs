//! Step throughput benchmarks for the FORGE simulation engine.
//!
//! Measures steps/second for single and multi-agent environments
//! across various world sizes.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use forge_core::WorldState;
use forge_types::config::ForgeConfig;
use forge_types::grid::Direction;
use forge_types::Action;

fn make_config(width: u16, height: u16, num_agents: u32) -> ForgeConfig {
    let mut config = ForgeConfig::default();
    config.world.width = width;
    config.world.height = height;
    config.world.seed = 42;
    config.agents.num_agents = num_agents;
    config.task.max_episode_length = 0; // no truncation during bench
    config
}

fn bench_step_single_agent(c: &mut Criterion) {
    let mut group = c.benchmark_group("step_single_agent");

    for size in [16u16, 32, 64, 128] {
        let config = make_config(size, size, 1);
        let mut world = WorldState::new(config);
        let actions = vec![Action::Move(Direction::Right)];

        group.bench_with_input(
            BenchmarkId::new("grid_size", format!("{}x{}", size, size)),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(world.step(&actions));
                });
            },
        );
    }

    group.finish();
}

fn bench_step_multi_agent(c: &mut Criterion) {
    let mut group = c.benchmark_group("step_multi_agent");

    for num_agents in [1u32, 2, 4, 8] {
        let config = make_config(64, 64, num_agents);
        let mut world = WorldState::new(config);
        let actions: Vec<Action> = (0..num_agents)
            .map(|i| Action::Move(Direction::from_index((i % 4) as u8).unwrap()))
            .collect();

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

fn bench_step_noop(c: &mut Criterion) {
    let config = make_config(64, 64, 1);
    let mut world = WorldState::new(config);
    let actions = vec![Action::Noop];

    c.bench_function("step_noop_64x64", |b| {
        b.iter(|| {
            black_box(world.step(&actions));
        });
    });
}

fn bench_world_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("world_creation");

    for size in [16u16, 32, 64, 128] {
        let config = make_config(size, size, 1);

        group.bench_with_input(
            BenchmarkId::new("grid_size", format!("{}x{}", size, size)),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(WorldState::new(config.clone()));
                });
            },
        );
    }

    group.finish();
}

fn bench_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialization");

    for size in [16u16, 32, 64] {
        let config = make_config(size, size, 1);
        let world = WorldState::new(config);

        group.bench_with_input(
            BenchmarkId::new("to_bytes", format!("{}x{}", size, size)),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(world.to_bytes());
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_step_single_agent,
    bench_step_multi_agent,
    bench_step_noop,
    bench_world_creation,
    bench_serialization
);
criterion_main!(benches);
