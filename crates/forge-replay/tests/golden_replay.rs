//! Bit-identity gate for CompactReplay format version 2.
//!
//! PR CI runs this via `cargo test --workspace`. A mismatch is a hard fail.
//! Intentional corpus updates: set `UPDATE_GOLDEN_REPLAYS=1`, re-run this
//! test, and append a row to `docs/results/replay_flip_log.md`.

use std::path::PathBuf;

use forge_replay::compact::{hash_config, CompactReplay, FORMAT_VERSION, GOLDEN_TIMESTAMP};
use forge_types::config::ForgeConfig;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/replays")
}

fn canonical_replay() -> CompactReplay {
    let mut config = ForgeConfig::default();
    config.world.width = 16;
    config.world.height = 16;
    config.agents.num_agents = 1;
    config.agents.comm_vocab_size = 0;
    config.task.max_episode_length = 32;
    let mut builder = CompactReplay::builder(config, 42);
    builder.record_tick(vec![0]);
    builder.record_tick(vec![4]);
    builder.record_tick(vec![0]);
    builder
        .agent_names(vec!["golden".into()])
        .final_rewards(vec![0.0])
        .timestamp(GOLDEN_TIMESTAMP)
        .build()
}

fn golden_json_path() -> PathBuf {
    golden_dir().join("v2_seed42.json")
}

#[test]
fn golden_v2_seed42_bit_identity() {
    let replay = canonical_replay();
    assert_eq!(replay.format_version, FORMAT_VERSION);
    assert_eq!(replay.config_hash, hash_config(&replay.config));
    assert_eq!(replay.metadata.timestamp, GOLDEN_TIMESTAMP);

    let mut actual = replay.to_json().expect("serialize golden replay");
    if !actual.ends_with('\n') {
        actual.push('\n');
    }

    let path = golden_json_path();
    if std::env::var("UPDATE_GOLDEN_REPLAYS").as_deref() == Ok("1") {
        std::fs::create_dir_all(golden_dir()).expect("create golden dir");
        std::fs::write(&path, &actual).expect("write golden");
        return;
    }

    let expected = std::fs::read_to_string(&path).unwrap_or_else(|err| {
        panic!(
            "missing golden {} ({err}). Generate with UPDATE_GOLDEN_REPLAYS=1 \
             cargo test -p forge-replay --test golden_replay, then append \
             docs/results/replay_flip_log.md",
            path.display()
        )
    });
    assert_eq!(
        expected,
        actual,
        "CompactReplay golden mismatch at {}. If the format change is \
         intentional, regenerate with UPDATE_GOLDEN_REPLAYS=1 and log the \
         flip in docs/results/replay_flip_log.md",
        path.display()
    );
}

#[test]
fn golden_v2_seed42_replays_with_full_agent_snapshot() {
    let path = golden_json_path();
    let json =
        std::fs::read_to_string(&path).unwrap_or_else(|_| canonical_replay().to_json().unwrap());
    let replay = CompactReplay::from_json(&json).expect("parse golden JSON");
    let mut iter = replay.replay().expect("start golden replay");
    let mut steps = 0_usize;
    while let Some(step) = iter.next() {
        let result = step.expect("golden step");
        let agent = &iter.world().agents[0];
        assert_eq!(
            result.observations[0].position,
            (agent.position.x, agent.position.y)
        );
        assert_eq!(result.observations[0].altitude, agent.altitude);
        assert!(agent.alive);
        assert!(agent.battery >= 0);
        steps += 1;
    }
    assert_eq!(steps, 3);
}
