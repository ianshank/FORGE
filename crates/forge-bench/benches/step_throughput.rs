//! Step throughput benchmarks for the FORGE simulation engine.
//!
//! Measures steps/second for single and multi-agent environments
//! across various world sizes.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use forge_core::WorldState;
use forge_types::config::ForgeConfig;
use forge_types::grid::{Direction, HexDirection};
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
        let mut world = WorldState::new(config).unwrap();
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
        let mut world = WorldState::new(config).unwrap();
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
    let mut world = WorldState::new(config).unwrap();
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
                    black_box(WorldState::new(config.clone()).unwrap());
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
        let world = WorldState::new(config).unwrap();

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

fn make_hex_config(width: u16, height: u16, num_agents: u32) -> ForgeConfig {
    let mut config = make_config(width, height, num_agents);
    config.world.grid_type = forge_types::config::GridType::Hex;
    config
}

fn bench_step_hex_single_agent(c: &mut Criterion) {
    let mut group = c.benchmark_group("step_hex_single_agent");

    for size in [16u16, 32, 64, 128] {
        let config = make_hex_config(size, size, 1);
        let mut world = WorldState::new(config).unwrap();
        let actions = vec![Action::MoveHex(HexDirection::E)];

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

fn bench_step_hex_multi_agent(c: &mut Criterion) {
    let mut group = c.benchmark_group("step_hex_multi_agent");

    for num_agents in [1u32, 2, 4, 8] {
        let config = make_hex_config(64, 64, num_agents);
        let mut world = WorldState::new(config).unwrap();
        let actions: Vec<Action> = (0..num_agents)
            .map(|i| Action::MoveHex(HexDirection::from_index((i % 6) as u8).unwrap()))
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

fn bench_env_trait_throughput(c: &mut Criterion) {
    use forge_env::Env;
    use forge_env_forge::WorldEnv;

    let mut group = c.benchmark_group("env_trait_single_agent");

    for size in [16u16, 32, 64, 128] {
        let config = make_config(size, size, 1);
        let mut env = WorldEnv::new(config).unwrap();
        let action = Action::Move(Direction::Right);
        let mut out = forge_env::StepOutput::default();

        group.bench_with_input(
            BenchmarkId::new("grid_size", format!("{}x{}", size, size)),
            &size,
            |b, _| {
                b.iter(|| {
                    env.step_into(action.clone(), &mut out).unwrap();
                    black_box(&out);
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
    bench_serialization,
    bench_step_hex_single_agent,
    bench_step_hex_multi_agent,
    bench_env_trait_throughput
);
criterion_main!(benches);
