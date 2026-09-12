use std::sync::Arc;

use super::*;
use forge_types::config::{ForgeConfig, GridType};
use forge_types::grid::{Direction, Position};
use forge_types::task::ActiveTask;
use forge_types::Action;

fn make_test_world() -> WorldState {
    let mut config = ForgeConfig::default();
    config.world.width = 16;
    config.world.height = 16;
    config.world.seed = 42;
    config.agents.num_agents = 1;
    config.task.max_episode_length = 1000;
    WorldState::new(config).unwrap()
}

#[test]
fn test_world_creation() {
    let world = make_test_world();
    assert_eq!(world.tick, 0);
    assert_eq!(world.agents.len(), 1);
    assert_eq!(world.grid.width, 16);
    assert_eq!(world.grid.height, 16);
    assert!(!world.terminated);
    assert!(!world.truncated);
}

#[test]
fn test_world_step() {
    let mut world = make_test_world();
    let start_pos = world.agents[0].position;

    let result = world.step(&[Action::Move(Direction::Right)]);

    assert_eq!(world.tick, 1);
    assert_eq!(result.observations.len(), 1);
    assert_eq!(result.rewards.len(), 1);
    assert!(!result.terminated);
    assert!(!result.truncated);

    // Agent should have moved (unless at boundary)
    if start_pos.x + 1 < world.grid.width {
        assert_eq!(
            world.agents[0].position,
            Position::new(start_pos.x + 1, start_pos.y)
        );
    }
}

#[test]
fn test_world_reset() {
    let mut world = make_test_world();
    world.step(&[Action::Move(Direction::Right)]);
    world.step(&[Action::Move(Direction::Right)]);
    assert_eq!(world.tick, 2);

    let result = world.reset(Some(99));
    assert_eq!(world.tick, 0);
    assert!(!world.terminated);
    assert!(!world.truncated);
    assert_eq!(result.observations.len(), 1);
}

#[test]
fn test_world_truncation() {
    let mut config = ForgeConfig::default();
    config.world.width = 8;
    config.world.height = 8;
    config.agents.default_vision_radius = 3;
    config.agents.num_agents = 1;
    config.task.max_episode_length = 5;
    let mut world = WorldState::new(config).unwrap();

    for _ in 0..5 {
        let result = world.step(&[Action::Noop]);
        if result.truncated {
            break;
        }
    }

    assert!(world.truncated);
}

#[test]
fn test_determinism() {
    let config1 = {
        let mut c = ForgeConfig::default();
        c.world.width = 16;
        c.world.height = 16;
        c.world.seed = 42;
        c.agents.num_agents = 2;
        c
    };
    let config2 = config1.clone();

    let mut world1 = WorldState::new(config1).unwrap();
    let mut world2 = WorldState::new(config2).unwrap();

    // Same actions should produce identical states
    let action_sequence = vec![
        vec![
            Action::Move(Direction::Right),
            Action::Move(Direction::Down),
        ],
        vec![Action::Move(Direction::Up), Action::Move(Direction::Left)],
        vec![Action::Noop, Action::Move(Direction::Right)],
        vec![Action::Move(Direction::Down), Action::Move(Direction::Down)],
    ];

    for actions in &action_sequence {
        let r1 = world1.step(actions);
        let r2 = world2.step(actions);

        assert_eq!(world1.tick, world2.tick);
        // Length guard first: `zip` silently truncates to the shorter
        // iterator, so a divergence that changed the agent count would
        // otherwise slip through the per-agent comparison below.
        assert_eq!(
            world1.agents.len(),
            world2.agents.len(),
            "agent count diverged"
        );
        assert_eq!(
            world1.objects.len(),
            world2.objects.len(),
            "object count diverged"
        );
        assert_eq!(
            world1.resources.len(),
            world2.resources.len(),
            "resource count diverged"
        );
        for (a1, a2) in world1.agents.iter().zip(world2.agents.iter()) {
            assert_eq!(a1.position, a2.position);
            assert_eq!(a1.health, a2.health);
            assert_eq!(a1.stamina, a2.stamina);
        }
        assert_eq!(r1.terminated, r2.terminated);
        assert_eq!(r1.truncated, r2.truncated);
    }
}

#[test]
fn test_observation_shape() {
    let world = make_test_world();
    let obs = world.generate_observation(&world.agents[0]);

    let vr = world.agents[0].vision_radius as u16;
    let expected_side = 2 * vr + 1;
    assert_eq!(obs.view_width, expected_side);
    assert_eq!(obs.view_height, expected_side);
    assert_eq!(
        obs.grid_view.len(),
        (expected_side as usize) * (expected_side as usize)
    );
}

#[test]
fn test_debug_grid() {
    let world = make_test_world();
    let debug = world.to_debug_grid();
    assert!(!debug.is_empty());
    assert!(debug.contains('A')); // At least one agent
    assert!(debug.contains('.')); // At least some ground
}

#[test]
fn test_serialization() {
    let world = make_test_world();
    let bytes = world.to_bytes();
    assert!(!bytes.is_empty());
    // Verify size is reasonable (should be well under 64KB for a 16x16 world)
    assert!(
        bytes.len() < 65536,
        "state too large: {} bytes",
        bytes.len()
    );
}

#[test]
fn test_step_with_wrong_action_count() {
    let mut world = make_test_world();
    // Too few actions — should be padded with Noop
    let result = world.step(&[]);
    assert!(!result.terminated);

    // Too many actions — extra should be ignored
    let result = world.step(&[Action::Noop, Action::Noop, Action::Noop]);
    assert!(!result.terminated);
}

#[test]
fn test_multi_agent_world() {
    let mut config = ForgeConfig::default();
    config.world.width = 32;
    config.world.height = 32;
    config.world.seed = 42;
    config.agents.num_agents = 4;
    let world = WorldState::new(config).unwrap();

    assert_eq!(world.agents.len(), 4);
    // All agents should be alive
    assert!(world.agents.iter().all(|a| a.alive));
}

// ---- Edge case tests ----

#[test]
fn test_grid_with_min_dimension() {
    let mut config = ForgeConfig::default();
    config.world.width = 8;
    config.world.height = 8;
    config.agents.default_vision_radius = 3;
    config.world.seed = 42;
    config.agents.num_agents = 1;
    let world = WorldState::new(config).unwrap();

    assert_eq!(world.grid.width, 8);
    assert_eq!(world.grid.height, 8);
    assert_eq!(world.agents.len(), 1);
    assert!(world.agents[0].alive);
    // Agent should be within bounds
    assert!(world.agents[0].position.x < 8);
    assert!(world.agents[0].position.y < 8);
}

#[test]
fn test_observation_agent_at_corner_origin() {
    let mut config = ForgeConfig::default();
    config.world.width = 16;
    config.world.height = 16;
    config.world.seed = 42;
    config.agents.num_agents = 1;
    let mut world = WorldState::new(config).unwrap();

    // Move agent to (0,0)
    let old_pos = world.agents[0].position;
    if let Some(tile) = world.grid.get_mut(old_pos.x, old_pos.y) {
        tile.agent_id = None;
    }
    world.agents[0].position = Position::new(0, 0);
    world.grid.get_mut(0, 0).unwrap().agent_id = Some(0);

    let obs = world.generate_observation(&world.agents[0]);

    // Observation should still have the correct shape
    let vr = world.agents[0].vision_radius as u16;
    let expected_side = 2 * vr + 1;
    assert_eq!(obs.view_width, expected_side);
    assert_eq!(obs.view_height, expected_side);
    assert_eq!(
        obs.grid_view.len(),
        (expected_side as usize) * (expected_side as usize)
    );

    // Tiles beyond the boundary should show as walls
    // The top-left corner of the view (at offset -vr, -vr from agent at (0,0))
    // should be out of bounds (wall)
    let first_tile = &obs.grid_view[0];
    assert_eq!(first_tile.terrain, forge_types::TerrainType::Wall as u8);
}

#[test]
fn test_hex_observation_filters_positions_outside_hex_radius() {
    let mut config = ForgeConfig::default();
    config.world.width = 16;
    config.world.height = 16;
    config.world.grid_type = GridType::Hex;
    config.world.seed = 42;
    config.agents.num_agents = 1;
    config.agents.default_vision_radius = 1;
    let mut world = WorldState::new(config).unwrap();

    let old_pos = world.agents[0].position;
    if let Some(tile) = world.grid.get_mut(old_pos.x, old_pos.y) {
        tile.agent_id = None;
    }
    world.agents[0].position = Position::new(5, 5);
    world.grid.get_mut(5, 5).unwrap().agent_id = Some(0);

    let obs = world.generate_observation(&world.agents[0]);

    assert_eq!(obs.view_width, 3);
    assert_eq!(obs.view_height, 3);
    assert_eq!(obs.grid_view.len(), 9);
    assert_eq!(
        obs.grid_view[0].terrain,
        forge_types::TerrainType::Wall as u8,
        "hex view should treat cells outside the hex radius as walls"
    );
}

#[test]
fn test_reset_with_none_seed() {
    let mut world = make_test_world();
    let initial_seed = world.config.world.seed;

    // Step a few times to advance the RNG
    world.step(&[Action::Noop]);
    world.step(&[Action::Noop]);

    // Reset with None — should derive a new seed from the RNG
    let result = world.reset(None);
    assert_eq!(world.tick, 0);
    assert!(!world.terminated);
    assert!(!world.truncated);
    assert_eq!(result.observations.len(), 1);

    // The seed should have changed (overwhelmingly likely)
    assert_ne!(
        world.config.world.seed, initial_seed,
        "reset(None) should use a derived seed"
    );
}

#[test]
fn test_multiple_resets_produce_different_states() {
    let mut world = make_test_world();

    // First reset with a specific seed
    world.reset(Some(100));
    let pos_after_first = world.agents[0].position;

    // Second reset with a different seed
    world.reset(Some(200));
    let pos_after_second = world.agents[0].position;

    // Third reset with another seed
    world.reset(Some(300));
    let pos_after_third = world.agents[0].position;

    // At least two of the three positions should differ
    // (technically all could coincide, but with a 16x16 grid that is very unlikely)
    let all_same = pos_after_first == pos_after_second && pos_after_second == pos_after_third;
    assert!(
        !all_same,
        "multiple resets with different seeds should produce different states"
    );
}

#[test]
fn test_to_debug_grid_with_empty_world() {
    let mut config = ForgeConfig::default();
    config.world.width = 8;
    config.world.height = 8;
    config.agents.default_vision_radius = 3;
    config.world.seed = 42;
    config.agents.num_agents = 0;
    config.task.enabled = false;
    let world = WorldState::new(config).unwrap();

    let debug = world.to_debug_grid();
    assert!(!debug.is_empty());
    // With no agents, should be all ground tiles
    assert!(!debug.contains('A'));
    // Should have 8 rows (each 8 chars + newline)
    let lines: Vec<&str> = debug.lines().collect();
    assert_eq!(lines.len(), 8);
    for line in &lines {
        assert_eq!(line.len(), 8);
    }
}

#[test]
fn test_to_bytes_round_trip_consistency() {
    let world = make_test_world();
    let bytes1 = world.to_bytes();
    let bytes2 = world.to_bytes();

    assert!(!bytes1.is_empty());
    assert_eq!(bytes1, bytes2, "serialization should be deterministic");
}

#[test]
fn test_to_bytes_changes_after_step() {
    let mut world = make_test_world();
    let bytes_before = world.to_bytes();

    world.step(&[Action::Move(Direction::Right)]);
    let bytes_after = world.to_bytes();

    assert_ne!(
        bytes_before, bytes_after,
        "state bytes should change after stepping"
    );
}

#[test]
fn test_step_after_termination_returns_terminal_result() {
    let mut config = ForgeConfig::default();
    config.world.width = 16;
    config.world.height = 16;
    config.world.seed = 42;
    config.agents.num_agents = 1;
    let mut world = WorldState::new(config).unwrap();

    // Kill the agent to trigger termination
    world.agents[0].alive = false;
    let result = world.step(&[Action::Noop]);
    assert!(result.terminated);

    // Subsequent step should still return terminal result
    let result2 = world.step(&[Action::Noop]);
    assert!(result2.terminated);
}

#[test]
fn test_world_min_dimension_width() {
    let mut config = ForgeConfig::default();
    config.world.width = 8;
    config.world.height = 16;
    config.world.seed = 42;
    config.agents.num_agents = 1;
    config.agents.default_vision_radius = 3;
    let mut world = WorldState::new(config).unwrap();

    // Should still work with min width
    let result = world.step(&[Action::Noop]);
    assert!(!result.terminated);
    assert_eq!(world.grid.width, 8);
}

#[test]
fn test_world_min_dimension_height() {
    let mut config = ForgeConfig::default();
    config.world.width = 16;
    config.world.height = 8;
    config.agents.default_vision_radius = 3;
    config.world.seed = 42;
    config.agents.num_agents = 1;
    let mut world = WorldState::new(config).unwrap();

    let result = world.step(&[Action::Noop]);
    assert!(!result.terminated);
    assert_eq!(world.grid.height, 8);
}

#[test]
fn test_observation_position_field() {
    let world = make_test_world();
    let obs = world.generate_observation(&world.agents[0]);

    assert_eq!(obs.position.0, world.agents[0].position.x);
    assert_eq!(obs.position.1, world.agents[0].position.y);
}

#[test]
fn test_observation_health_and_stamina_normalized() {
    let world = make_test_world();
    let obs = world.generate_observation(&world.agents[0]);

    // Health and stamina should be between 0.0 and 1.0
    assert!(obs.health >= 0.0 && obs.health <= 1.0);
    assert!(obs.stamina >= 0.0 && obs.stamina <= 1.0);
}

// ---- from_bytes / from_json roundtrip tests ----

#[test]
fn test_from_bytes_roundtrip() {
    let mut world = make_test_world();
    world.step(&[Action::Noop]);
    world.step(&[Action::Move(Direction::Right)]);

    let bytes = world.to_bytes();
    let restored = WorldState::from_bytes(&bytes, world.config.clone()).unwrap();

    assert_eq!(restored.tick, world.tick);
    assert_eq!(restored.day_phase, world.day_phase);
    assert_eq!(restored.agents.len(), world.agents.len());
    assert_eq!(restored.agents[0].position, world.agents[0].position);
    assert_eq!(restored.agents[0].health, world.agents[0].health);
    assert_eq!(restored.terminated, world.terminated);
    assert_eq!(restored.truncated, world.truncated);
}

#[test]
fn test_from_bytes_rng_preserves_sequence() {
    let mut world = make_test_world();
    for _ in 0..5 {
        world.step(&[Action::Noop]);
    }

    let bytes = world.to_bytes();
    let mut restored = WorldState::from_bytes(&bytes, world.config.clone()).unwrap();

    // After restoration, stepping both should produce the same results
    let result_original = world.step(&[Action::Noop]);
    let result_restored = restored.step(&[Action::Noop]);
    assert_eq!(
        result_original.observations[0].position,
        result_restored.observations[0].position,
    );
}

#[test]
fn test_from_bytes_invalid_data() {
    let config = Arc::new(ForgeConfig::default());
    let result = WorldState::from_bytes(&[0, 1, 2, 3], config);
    assert!(result.is_err());
}

#[test]
fn test_to_json_roundtrip() {
    let mut world = make_test_world();
    world.step(&[Action::Noop]);

    let json = world.to_json().unwrap();
    assert!(!json.is_empty());

    let restored = WorldState::from_json(&json, world.config.clone()).unwrap();
    assert_eq!(restored.tick, world.tick);
    assert_eq!(restored.agents[0].position, world.agents[0].position);
    assert_eq!(restored.day_phase, world.day_phase);
}

#[test]
fn test_from_json_invalid_data() {
    let config = Arc::new(ForgeConfig::default());
    let result = WorldState::from_json("not valid json", config);
    assert!(result.is_err());
}

// ---- Fallible serialization twin ----

#[test]
fn test_try_to_bytes_succeeds_and_is_non_empty() {
    let mut world = make_test_world();
    world.step(&[Action::Move(Direction::Right)]);

    let bytes = world
        .try_to_bytes()
        .expect("bincode must encode a well-formed world");
    assert!(
        !bytes.is_empty(),
        "a successful encoding must never be empty — an empty Vec is the \
         failure sentinel returned by the infallible `to_bytes` wrapper"
    );
}

#[test]
fn test_try_to_bytes_agrees_with_to_bytes() {
    let mut world = make_test_world();
    world.step(&[Action::Noop]);

    let fallible = world.try_to_bytes().expect("encoding must succeed");
    assert_eq!(
        fallible,
        world.to_bytes(),
        "`to_bytes` must delegate to `try_to_bytes` byte-for-byte"
    );
}

// ---- Golden state-hash regression ----
//
// A same-process A/B comparison (see `test_determinism` / the
// `step_determinism` property) proves two worlds in *this* binary agree.
// It cannot detect drift between binaries: a change to world generation,
// system ordering, RNG behaviour, `ForgeConfig::default()`, or the bincode
// layout silently changes the state a given seed produces while every A/B
// test stays green. The digests below pin that mapping.

/// FNV-1a 64-bit offset basis, from the FNV reference specification.
///
/// FNV is used rather than `std::hash::DefaultHasher` because the latter's
/// output is explicitly *not* guaranteed stable across Rust releases,
/// which would make a committed golden value meaningless. Hard-coding the
/// two FNV constants keeps `forge-core` free of a hashing dependency.
const FNV1A64_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

/// FNV-1a 64-bit prime, from the FNV reference specification.
const FNV1A64_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Seeds pinned by [`test_golden_state_hash`]. Three distinct seeds so a
/// drift affecting only some world layouts is still caught.
const GOLDEN_SEEDS: [u64; 3] = [1, 42, 7_777];

/// Number of simulation steps taken before the digest is computed.
///
/// Chosen to cross the first day/night phase boundary: with the default
/// 1000-tick cycle each phase lasts 250 ticks, so the run must exceed 250
/// steps for the visibility system's phase-dependent branch to be part of
/// what the digest pins. Still well under
/// [`GOLDEN_MAX_EPISODE_LEN`], and cheap enough to stay a sub-millisecond
/// test across all three seeds.
const GOLDEN_STEP_COUNT: usize = 260;

/// Grid edge length of the golden worlds.
const GOLDEN_WORLD_SIZE: u16 = 16;

/// Agent count of the golden worlds.
const GOLDEN_AGENT_COUNT: u32 = 2;

/// Episode-length cap for the golden worlds. Comfortably above
/// [`GOLDEN_STEP_COUNT`] so truncation never cuts a run short.
const GOLDEN_MAX_EPISODE_LEN: u64 = 1000;

/// Expected FNV-1a-64 digests of `try_to_bytes()` after
/// [`GOLDEN_STEP_COUNT`] steps, one per entry of [`GOLDEN_SEEDS`].
///
/// **Regenerating:** these encode real simulation behaviour, so a change
/// here means the state produced by a fixed seed changed. If that change
/// was intentional, run `cargo test -p forge-core golden_state_hash` and
/// copy the actual digests out of the assertion failure message. If it
/// was not, a determinism regression has been caught — investigate before
/// updating the constants.
const GOLDEN_DIGESTS: [u64; 3] = [
    0x2622_5f16_87e9_0062,
    0xf78d_bfe8_14b5_5f58,
    0xe75a_cfb5_49aa_8dec,
];

/// Version-stable 64-bit digest of a byte slice (FNV-1a).
///
/// `wrapping_mul` is deliberate: FNV specifies modular arithmetic, and the
/// workspace builds `--release` with `overflow-checks = true`, so a plain
/// `*` here would panic.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = FNV1A64_OFFSET_BASIS;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV1A64_PRIME);
    }
    hash
}

/// The fixed per-step action script the golden digests are computed
/// against, cycled until [`GOLDEN_STEP_COUNT`] steps have been taken.
/// One action per agent, so its inner length matches
/// [`GOLDEN_AGENT_COUNT`].
fn golden_action_script() -> Vec<Vec<Action>> {
    vec![
        vec![
            Action::Move(Direction::Right),
            Action::Move(Direction::Down),
        ],
        vec![Action::Noop, Action::Move(Direction::Left)],
        vec![Action::PickUp, Action::Interact],
        vec![Action::Move(Direction::Up), Action::Push(Direction::Right)],
    ]
}

/// Builds the world the golden digests are computed against.
fn make_golden_world(seed: u64) -> WorldState {
    let mut config = ForgeConfig::default();
    config.world.width = GOLDEN_WORLD_SIZE;
    config.world.height = GOLDEN_WORLD_SIZE;
    config.world.seed = seed;
    config.agents.num_agents = GOLDEN_AGENT_COUNT;
    config.task.max_episode_length = GOLDEN_MAX_EPISODE_LEN;
    WorldState::new(config).expect("golden config must be valid")
}

/// Runs the golden script for one seed and returns the state digest.
fn golden_digest_for_seed(seed: u64) -> u64 {
    let script = golden_action_script();
    let mut world = make_golden_world(seed);
    for step in 0..GOLDEN_STEP_COUNT {
        world.step(&script[step % script.len()]);
    }

    let bytes = world
        .try_to_bytes()
        .expect("golden world must serialize successfully");
    assert!(
        !bytes.is_empty(),
        "golden world serialized to an empty payload for seed {seed}"
    );
    assert!(
        !world.truncated,
        "golden run must not truncate: the digest is only meaningful for a \
         full GOLDEN_STEP_COUNT-step run. Most likely GOLDEN_STEP_COUNT now \
         exceeds GOLDEN_MAX_EPISODE_LEN, but any other truncation cause \
         invalidates the golden value the same way"
    );
    assert_ne!(
        world.day_phase, 0,
        "golden run must cross a day/night phase boundary — see \
         GOLDEN_STEP_COUNT"
    );
    fnv1a64(&bytes)
}

/// Same as [`make_golden_world`] but with aerial morphology enabled so the
/// integer-grid drone path is covered without rewriting the ground digests.
fn make_golden_aerial_world(seed: u64) -> WorldState {
    let mut config = ForgeConfig::default();
    config.world.width = GOLDEN_WORLD_SIZE;
    config.world.height = GOLDEN_WORLD_SIZE;
    config.world.seed = seed;
    config.agents.num_agents = GOLDEN_AGENT_COUNT;
    config.task.max_episode_length = GOLDEN_MAX_EPISODE_LEN;
    config.drone.enabled = true;
    config.drone.num_aerial = GOLDEN_AGENT_COUNT;
    WorldState::new(config).expect("golden aerial config must be valid")
}

fn golden_aerial_digest_for_seed(seed: u64) -> u64 {
    let script = golden_action_script();
    let mut world = make_golden_aerial_world(seed);
    for step in 0..GOLDEN_STEP_COUNT {
        world.step(&script[step % script.len()]);
    }
    let bytes = world
        .try_to_bytes()
        .expect("golden aerial world must serialize successfully");
    assert!(!bytes.is_empty(), "golden aerial world serialized empty");
    fnv1a64(&bytes)
}

#[test]
fn test_golden_state_hash() {
    let actual: Vec<u64> = GOLDEN_SEEDS
        .iter()
        .copied()
        .map(golden_digest_for_seed)
        .collect();

    // Compared as a whole array so a failure prints every digest at once,
    // which is what the regeneration instructions on `GOLDEN_DIGESTS`
    // rely on.
    assert_eq!(
        actual,
        GOLDEN_DIGESTS.to_vec(),
        "golden state digests drifted for seeds {GOLDEN_SEEDS:?}; actual \
         values are {actual:#018x?} — update GOLDEN_DIGESTS only if the \
         change to simulation behaviour was intentional"
    );
}

const GOLDEN_AERIAL_DIGESTS: [u64; 3] = [
    0x08d9_9c10_a75f_54dc,
    0x1fb0_25ff_5c93_080e,
    0x6383_f129_8c9b_9e12,
];

#[test]
fn test_golden_aerial_state_hash() {
    let actual: Vec<u64> = GOLDEN_SEEDS
        .iter()
        .copied()
        .map(golden_aerial_digest_for_seed)
        .collect();
    assert_eq!(
        actual,
        GOLDEN_AERIAL_DIGESTS.to_vec(),
        "golden aerial digests drifted for seeds {GOLDEN_SEEDS:?}; actual \
         values are {actual:#018x?} — update GOLDEN_AERIAL_DIGESTS only if \
         the aerial path change was intentional"
    );
}

#[test]
fn test_golden_state_hash_is_sensitive_to_state() {
    // Guards the guard: if `fnv1a64` collapsed to a constant (or
    // `try_to_bytes` returned a fixed payload) the golden test above would
    // pass vacuously. Distinct seeds must yield distinct digests.
    let mut digests: Vec<u64> = GOLDEN_SEEDS
        .iter()
        .copied()
        .map(golden_digest_for_seed)
        .collect();

    digests.sort_unstable();
    let before = digests.len();
    digests.dedup();
    assert_eq!(
        digests.len(),
        before,
        "distinct seeds produced identical state digests — the digest is \
         not discriminating between world states"
    );
}

// ---- Reward value assertions ----

/// Tolerance for reward comparisons. The expected values below are small
/// sums of near-exactly-representable `f32`s, so this only absorbs the
/// rounding in `1.0 - distance / 100.0`. Anything looser would let a
/// scaling bug through.
const REWARD_EPSILON: f32 = 1e-6;

/// Reward scale used by the reward-value tests. Deliberately *not* 1.0 so
/// that a dropped or doubled `reward_scale` multiplication is visible.
const REWARD_TEST_SCALE: f32 = 2.0;

/// Completion reward assigned to the test task, before scaling.
const REWARD_TEST_TASK_REWARD: f32 = 5.0;

/// Manhattan distance between the agent and an unreachable-this-tick task
/// target, used to derive the partial-progress reward. Small enough to
/// stay inside a 16x16 grid from any spawn tile.
const REWARD_TEST_TARGET_DISTANCE: u16 = 4;

/// Denominator `eval_agent_at` uses to turn Manhattan distance into
/// progress (`progress = 1 - min(distance / MAX_DIST, 1)`).
const REWARD_TEST_PROGRESS_MAX_DISTANCE: f32 = 100.0;

/// Grid edge length of the reward-test worlds. Must exceed
/// `2 * REWARD_TEST_TARGET_DISTANCE` so a target that distance from the
/// spawn tile is in bounds whichever side of the grid the agent spawns
/// on.
const REWARD_TEST_WORLD_SIZE: u16 = 16;

/// Seed of the reward-test worlds. Any fixed value works; the derivations
/// below are independent of where the agent happens to spawn.
const REWARD_TEST_SEED: u64 = 42;

/// Episode-length cap for the reward-test worlds. Far above the handful
/// of steps these tests take, so truncation never interferes.
const REWARD_TEST_MAX_EPISODE_LEN: u64 = 1000;

/// Builds a world with `num_agents` agents whose task rewards are scaled
/// by [`REWARD_TEST_SCALE`].
fn make_reward_world(num_agents: u32) -> WorldState {
    let mut config = ForgeConfig::default();
    config.world.width = REWARD_TEST_WORLD_SIZE;
    config.world.height = REWARD_TEST_WORLD_SIZE;
    config.world.seed = REWARD_TEST_SEED;
    config.agents.num_agents = num_agents;
    config.task.max_episode_length = REWARD_TEST_MAX_EPISODE_LEN;
    config.task.reward_scale = REWARD_TEST_SCALE;
    WorldState::new(config).expect("reward-test config must be valid")
}

/// Wraps an `AgentAt` goal in an `ActiveTask` with dense shaping enabled.
fn make_agent_at_task(agent_id: u32, target: Position) -> ActiveTask {
    ActiveTask {
        definition: forge_types::task::TaskDefinition {
            id: 1,
            description: "reach target".to_string(),
            goal: forge_types::task::TaskComposition::Atom(forge_types::task::Predicate::AgentAt(
                agent_id, target,
            )),
            tier: forge_types::task::TaskTier::new(1),
            estimated_steps: 1,
            reward: REWARD_TEST_TASK_REWARD,
            // Non-empty enables dense shaping; the evaluator gates on
            // emptiness only and never reads the weight values.
            dense_reward_weights: vec![1.0],
        },
        progress: vec![0.0],
        sequence_index: 0,
        completed: false,
        failed: false,
    }
}

#[test]
fn test_step_rewards_are_exactly_zero_without_tasks() {
    // With no active tasks nothing writes `last_task_rewards`, so
    // `fill_step_result` must zero-fill. Pinning the exact value (rather
    // than only the length) catches a stray reward leaking in from
    // another system or a stale buffer surviving between steps.
    let mut world = make_reward_world(3);
    assert!(world.tasks.is_empty(), "precondition: no tasks configured");

    for _ in 0..3 {
        let result = world.step(&[Action::Noop, Action::Noop, Action::Noop]);
        assert_eq!(result.rewards.len(), 3);
        for (i, reward) in result.rewards.iter().enumerate() {
            assert_eq!(
                *reward, 0.0,
                "agent {i} received a non-zero reward with no active tasks"
            );
        }
    }
}

#[test]
fn test_step_reward_value_for_completed_task() {
    // Derivation (forge_task::evaluator::evaluate_tasks):
    //   The goal is `AgentAt(0, <spawn position>)` and the agent takes a
    //   Noop, so the predicate is satisfied on the first evaluation:
    //     progress: 0.0 -> 1.0
    //   dense reward      = (new_progress - old_progress) * reward_scale
    //                     = (1.0 - 0.0) * 2.0 = 2.0
    //   completion reward = task.reward * reward_scale
    //                     = 5.0 * 2.0 = 10.0
    //   total             = 12.0
    let mut world = make_reward_world(1);
    let spawn = world.agents[0].position;
    world.tasks.push(make_agent_at_task(0, spawn));

    let result = world.step(&[Action::Noop]);

    assert_eq!(
        world.agents[0].position, spawn,
        "precondition: a Noop must not move the agent"
    );
    assert_eq!(result.rewards.len(), 1);

    let expected = REWARD_TEST_SCALE + REWARD_TEST_TASK_REWARD * REWARD_TEST_SCALE;
    assert!(
        (result.rewards[0] - expected).abs() < REWARD_EPSILON,
        "expected reward {expected}, got {}",
        result.rewards[0]
    );
    assert!(world.tasks[0].completed, "task should be marked completed");
    assert!(
        result.terminated,
        "the episode terminates once every task is done"
    );
}

#[test]
fn test_step_reward_value_for_partial_progress() {
    // Derivation: the goal is `AgentAt(0, target)` with `target` exactly
    // REWARD_TEST_TARGET_DISTANCE tiles away in Manhattan distance, and
    // the agent takes a Noop, so it never arrives.
    //   progress          = 1 - (4 / 100) = 0.96
    //   dense reward      = (0.96 - 0.0) * 2.0 = 1.92
    //   completion reward = 0 (predicate unsatisfied)
    let mut world = make_reward_world(1);
    let spawn = world.agents[0].position;
    // Offset along x, away from the nearer edge so the target stays in
    // bounds from any spawn column (see REWARD_TEST_WORLD_SIZE).
    let target_x = if spawn.x >= REWARD_TEST_TARGET_DISTANCE {
        spawn.x - REWARD_TEST_TARGET_DISTANCE
    } else {
        spawn.x + REWARD_TEST_TARGET_DISTANCE
    };
    let target = Position::new(target_x, spawn.y);
    assert_eq!(
        spawn.manhattan_distance(&target),
        u32::from(REWARD_TEST_TARGET_DISTANCE),
        "precondition: target must sit at the derived distance"
    );
    world.tasks.push(make_agent_at_task(0, target));

    let result = world.step(&[Action::Noop]);

    let progress = 1.0 - f32::from(REWARD_TEST_TARGET_DISTANCE) / REWARD_TEST_PROGRESS_MAX_DISTANCE;
    let expected = progress * REWARD_TEST_SCALE;
    assert!(
        (result.rewards[0] - expected).abs() < REWARD_EPSILON,
        "expected dense-only reward {expected}, got {}",
        result.rewards[0]
    );
    assert!(
        !world.tasks[0].completed,
        "task must not complete from an unreached target"
    );
}

#[test]
fn test_step_reward_is_not_credited_to_dead_agents() {
    // The evaluator credits alive agents only. Pinning both values at
    // once catches a reward that is broadcast indiscriminately as well as
    // a sign inversion on the surviving agent's share.
    let mut world = make_reward_world(2);
    let spawn = world.agents[0].position;
    world.tasks.push(make_agent_at_task(0, spawn));
    world.agents[1].alive = false;

    let result = world.step(&[Action::Noop, Action::Noop]);

    let expected = REWARD_TEST_SCALE + REWARD_TEST_TASK_REWARD * REWARD_TEST_SCALE;
    assert_eq!(result.rewards.len(), 2);
    assert!(
        (result.rewards[0] - expected).abs() < REWARD_EPSILON,
        "alive agent: expected {expected}, got {}",
        result.rewards[0]
    );
    assert_eq!(
        result.rewards[1], 0.0,
        "dead agent must receive no task reward"
    );
}

// ---- Proptest: world invariants ----

mod proptests {
    use super::*;
    use forge_types::grid::HexDirection;
    use proptest::prelude::*;
    use proptest::test_runner::TestCaseError;

    /// Edge length of the square worlds built for the determinism
    /// properties. 16x16 is large enough for agents to move freely for
    /// the whole generated action sequence without every case pinning
    /// them against a wall, and small enough that world generation stays
    /// cheap across proptest's default 256 cases.
    const DETERMINISM_WORLD_SIZE: u16 = 16;

    /// Number of agents in each determinism world. Two agents exercise
    /// the multi-agent ordering paths (collisions, pushes, communication)
    /// that a single-agent world cannot reach.
    const DETERMINISM_AGENT_COUNT: u32 = 2;

    /// Episode length cap for determinism worlds. Must exceed
    /// `MAX_ACTION_SEQUENCE_LEN / DETERMINISM_AGENT_COUNT` so a generated
    /// sequence is never cut short by truncation, which would mask a
    /// divergence occurring in the final steps.
    const DETERMINISM_MAX_EPISODE_LEN: u64 = 100;

    /// Upper bound on the length of a generated action sequence. The
    /// sequence is consumed `DETERMINISM_AGENT_COUNT` actions per step,
    /// so this caps each case at 12 simulation steps for two worlds —
    /// enough for divergence to compound, cheap enough for 256 cases.
    const MAX_ACTION_SEQUENCE_LEN: usize = 24;

    /// Highest inventory slot index used by generated slot-indexed
    /// actions (`Drop`, `Use`, `DropPayload`, `Spray`). Two past the
    /// default 10-slot inventory so the out-of-range rejection path in
    /// `validate_actions_into` is exercised as well.
    const MAX_GENERATED_SLOT: u8 = 11;

    /// Highest recipe index used by generated `Craft` actions. Matches
    /// the 0-8 recipe range in the flat action encoding documented on
    /// `Action::from_index`.
    const MAX_GENERATED_RECIPE: u16 = 8;

    /// Highest communication token used by generated `Communicate`
    /// actions. One past the default 16-token vocabulary so the
    /// out-of-vocabulary rejection path is exercised too.
    const MAX_GENERATED_COMM_TOKEN: u16 = 16;

    fn make_world_with_seed(seed: u64) -> WorldState {
        let mut config = ForgeConfig::default();
        config.world.width = DETERMINISM_WORLD_SIZE;
        config.world.height = DETERMINISM_WORLD_SIZE;
        config.world.seed = seed;
        config.agents.num_agents = DETERMINISM_AGENT_COUNT;
        config.task.max_episode_length = DETERMINISM_MAX_EPISODE_LEN;
        WorldState::new(config).unwrap()
    }

    /// Strategy over the four cardinal directions.
    fn direction_strategy() -> impl Strategy<Value = Direction> {
        proptest::sample::select(Direction::all().to_vec())
    }

    /// Strategy over the six hex directions.
    fn hex_direction_strategy() -> impl Strategy<Value = HexDirection> {
        proptest::sample::select(HexDirection::ALL.to_vec())
    }

    /// Strategy covering every variant of [`Action`], with payloads that
    /// span both the accepted and the rejected ranges.
    ///
    /// Built with an explicit `Union` rather than `prop_oneof!` because
    /// the branch strategies have heterogeneous types and there are more
    /// of them than the macro's fixed-arity forms support.
    fn action_strategy() -> impl Strategy<Value = Action> {
        proptest::strategy::Union::new(vec![
            Just(Action::Noop).boxed(),
            direction_strategy().prop_map(Action::Move).boxed(),
            Just(Action::PickUp).boxed(),
            (0u8..=MAX_GENERATED_SLOT).prop_map(Action::Drop).boxed(),
            (0u8..=MAX_GENERATED_SLOT).prop_map(Action::Use).boxed(),
            (0u16..=MAX_GENERATED_RECIPE)
                .prop_map(Action::Craft)
                .boxed(),
            direction_strategy().prop_map(Action::Push).boxed(),
            (0u16..=MAX_GENERATED_COMM_TOKEN)
                .prop_map(Action::Communicate)
                .boxed(),
            Just(Action::Interact).boxed(),
            Just(Action::Ascend).boxed(),
            Just(Action::Descend).boxed(),
            Just(Action::Hover).boxed(),
            Just(Action::TakeOff).boxed(),
            Just(Action::Land).boxed(),
            direction_strategy().prop_map(Action::Scan).boxed(),
            (0u8..=MAX_GENERATED_SLOT)
                .prop_map(Action::DropPayload)
                .boxed(),
            (0u8..=MAX_GENERATED_SLOT).prop_map(Action::Spray).boxed(),
            Just(Action::ScanMultispectral).boxed(),
            Just(Action::ScanThermal).boxed(),
            Just(Action::RelaySoilData).boxed(),
            Just(Action::GenerateReport).boxed(),
            hex_direction_strategy().prop_map(Action::MoveHex).boxed(),
        ])
    }

    /// Strategy producing a variable-length action sequence. The sequence
    /// is consumed `DETERMINISM_AGENT_COUNT` actions at a time, so its
    /// length also varies the number of simulation steps and — when the
    /// length is not a multiple of the agent count — exercises the
    /// short-slice padding path in `step_into`.
    fn action_sequence_strategy() -> impl Strategy<Value = Vec<Action>> {
        proptest::collection::vec(action_strategy(), 1..=MAX_ACTION_SEQUENCE_LEN)
    }

    /// Compares the `WorldState` fields that `SerializableWorldState`
    /// (and therefore [`WorldState::try_to_bytes`]) does **not** carry,
    /// so that a byte comparison plus this helper covers every field of
    /// the struct.
    ///
    /// Several of the underlying types (`ActiveTask`, `RecipeBook`,
    /// `GridTopologyKind`, `CropState`, `SoilSensorNode`, `AgriScratch`,
    /// `PhysicsScratch`, `AgentPushData`) do not implement `PartialEq`,
    /// so those are compared through their `Debug` rendering. That is
    /// structural, and unlike `PartialEq` on floats it treats `NaN` as
    /// equal to `NaN` — which is the behaviour a bit-reproducibility
    /// check wants. `config` is compared through its JSON encoding for
    /// the same reason (`ForgeConfig` has no `PartialEq` either).
    fn assert_untracked_fields_eq(a: &WorldState, b: &WorldState) -> Result<(), TestCaseError> {
        prop_assert_eq!(a.tasks.len(), b.tasks.len(), "tasks length diverged");
        prop_assert_eq!(
            format!("{:?}", a.tasks),
            format!("{:?}", b.tasks),
            "tasks diverged"
        );
        prop_assert_eq!(
            format!("{:?}", a.recipe_book),
            format!("{:?}", b.recipe_book),
            "recipe_book diverged"
        );
        prop_assert_eq!(
            serde_json::to_string(a.config.as_ref()).expect("config must serialize"),
            serde_json::to_string(b.config.as_ref()).expect("config must serialize"),
            "config diverged"
        );
        prop_assert_eq!(
            format!("{:?}", a.last_task_rewards),
            format!("{:?}", b.last_task_rewards),
            "last_task_rewards diverged"
        );
        prop_assert_eq!(
            format!("{:?}", a.physics_scratch),
            format!("{:?}", b.physics_scratch),
            "physics_scratch diverged"
        );
        prop_assert_eq!(
            format!("{:?}", a.topology),
            format!("{:?}", b.topology),
            "topology diverged"
        );
        prop_assert_eq!(
            a.crop_states.len(),
            b.crop_states.len(),
            "crop_states length diverged"
        );
        prop_assert_eq!(
            format!("{:?}", a.crop_states),
            format!("{:?}", b.crop_states),
            "crop_states diverged"
        );
        prop_assert_eq!(
            a.soil_nodes.len(),
            b.soil_nodes.len(),
            "soil_nodes length diverged"
        );
        prop_assert_eq!(
            format!("{:?}", a.soil_nodes),
            format!("{:?}", b.soil_nodes),
            "soil_nodes diverged"
        );
        prop_assert_eq!(
            format!("{:?}", a.agri_scratch),
            format!("{:?}", b.agri_scratch),
            "agri_scratch diverged"
        );
        prop_assert_eq!(
            &a.step_actions,
            &b.step_actions,
            "step_actions buffer diverged"
        );
        prop_assert_eq!(
            &a.validated_actions,
            &b.validated_actions,
            "validated_actions buffer diverged"
        );
        prop_assert_eq!(&a.near_station, &b.near_station, "near_station diverged");
        // `HashMap` implements order-independent `PartialEq`, so compare
        // directly rather than through `Debug` (whose iteration order is
        // not stable between two maps in the same process).
        prop_assert_eq!(
            &a.crafting_object_map,
            &b.crafting_object_map,
            "crafting_object_map diverged"
        );
        prop_assert_eq!(&a.comm_messages, &b.comm_messages, "comm_messages diverged");
        prop_assert_eq!(
            format!("{:?}", a.push_scratch),
            format!("{:?}", b.push_scratch),
            "push_scratch diverged"
        );
        Ok(())
    }

    proptest! {
        /// Same seed + same actions = identical world state
        /// (CHARTER.md Invariant 6).
        ///
        /// Both the seed *and* the action sequence are generated, so the
        /// property holds over a spread of the `Action` enum rather than
        /// a single hard-coded script. Coverage is whole-state: the
        /// bincode payload plus every field that payload omits.
        #[test]
        fn step_determinism(
            seed in 0u64..10_000,
            actions in action_sequence_strategy(),
        ) {
            let mut w1 = make_world_with_seed(seed);
            let mut w2 = make_world_with_seed(seed);

            for chunk in actions.chunks(DETERMINISM_AGENT_COUNT as usize) {
                w1.step(chunk);
                w2.step(chunk);
            }

            prop_assert_eq!(w1.tick, w2.tick, "tick diverged");

            // Length guards BEFORE any `zip`: `zip` truncates to the
            // shorter iterator, so a nondeterminism bug that changed an
            // entity count would otherwise be silently skipped.
            prop_assert_eq!(w1.agents.len(), w2.agents.len(), "agent count diverged");
            prop_assert_eq!(w1.objects.len(), w2.objects.len(), "object count diverged");
            prop_assert_eq!(
                w1.resources.len(),
                w2.resources.len(),
                "resource count diverged"
            );
            prop_assert_eq!(
                w1.grid.tiles.len(),
                w2.grid.tiles.len(),
                "grid tile count diverged"
            );

            // Whole-payload byte comparison. `try_to_bytes` is used
            // instead of `to_bytes` because the latter returns an empty
            // `Vec` on a serialization failure, which would make this
            // comparison pass vacuously; the non-empty assertions below
            // close that hole for good measure.
            let b1 = w1.try_to_bytes().expect("w1 must serialize");
            let b2 = w2.try_to_bytes().expect("w2 must serialize");
            prop_assert!(!b1.is_empty(), "serialized state of w1 must be non-empty");
            prop_assert!(!b2.is_empty(), "serialized state of w2 must be non-empty");
            prop_assert_eq!(b1.len(), b2.len(), "serialized state length diverged");
            // Reported as the first differing offset rather than via
            // `prop_assert_eq!(&b1, &b2)`, which would dump both
            // multi-kilobyte payloads into the failure message.
            let first_diff = b1.iter().zip(b2.iter()).position(|(x, y)| x != y);
            prop_assert!(
                first_diff.is_none(),
                "serialized world state diverged at byte {:?} of {}: w1={:?}, w2={:?}",
                first_diff,
                b1.len(),
                first_diff.map(|i| b1[i]),
                first_diff.map(|i| b2[i])
            );

            // The bincode payload omits roughly two thirds of
            // `WorldState`; compare those fields explicitly so coverage
            // is complete without changing the wire format.
            assert_untracked_fields_eq(&w1, &w2)?;

            // Per-agent comparison last: it adds nothing the byte
            // comparison misses, but it names the diverging agent and
            // field when a failure does occur.
            for (i, (a, b)) in w1.agents.iter().zip(w2.agents.iter()).enumerate() {
                prop_assert_eq!(a.position, b.position, "agent {} position diverged", i);
                prop_assert_eq!(a.health, b.health, "agent {} health diverged", i);
                prop_assert_eq!(a.stamina, b.stamina, "agent {} stamina diverged", i);
                prop_assert_eq!(a.alive, b.alive, "agent {} alive diverged", i);
                prop_assert_eq!(a.altitude, b.altitude, "agent {} altitude diverged", i);
                prop_assert_eq!(a.battery, b.battery, "agent {} battery diverged", i);
            }
        }

        /// After any number of steps, all agent positions are in bounds.
        #[test]
        fn agents_in_bounds_after_steps(
            seed in 0u64..10_000,
            steps in 1u32..20,
        ) {
            let mut world = make_world_with_seed(seed);
            for _ in 0..steps {
                world.step(&[Action::Move(Direction::Right), Action::Move(Direction::Down)]);
            }
            for agent in &world.agents {
                prop_assert!(agent.position.x < world.grid.width);
                prop_assert!(agent.position.y < world.grid.height);
            }
        }

        /// Serialization roundtrip preserves essential state.
        #[test]
        fn bytes_roundtrip(seed in 0u64..10_000) {
            let mut world = make_world_with_seed(seed);
            world.step(&[Action::Noop, Action::Noop]);

            let bytes = world.to_bytes();
            let restored = WorldState::from_bytes(&bytes, world.config.clone()).unwrap();

            prop_assert_eq!(restored.tick, world.tick);
            prop_assert_eq!(restored.agents.len(), world.agents.len());
            for (a, b) in restored.agents.iter().zip(world.agents.iter()) {
                prop_assert_eq!(a.position, b.position);
            }
        }
    }
}

#[cfg(test)]
mod proptests_tick {
    use super::*;
    use forge_types::grid::Direction;
    use forge_types::Action;
    use proptest::prelude::*;

    /// Build a minimal, valid `ForgeConfig` suitable for fast proptest runs.
    /// Width and height are kept small (8–16) so world creation is cheap.
    fn small_config(width: u16, height: u16, seed: u64, num_agents: u32) -> ForgeConfig {
        let mut config = ForgeConfig::default();
        config.world.width = width;
        config.world.height = height;
        config.world.seed = seed;
        config.agents.num_agents = num_agents;
        // Vision radius must fit: 2*r+1 <= min(width, height)
        let max_radius = ((width.min(height) - 1) / 2) as u8;
        config.agents.default_vision_radius = config.agents.default_vision_radius.min(max_radius);
        config.task.max_episode_length = 1000;
        config
    }

    /// Strategy for valid small world dimensions [8, 20].
    fn small_dim() -> impl Strategy<Value = u16> {
        8u16..=20u16
    }

    proptest! {
        /// After `n` calls to `step`, `world.tick` must equal `n`.
        #[test]
        fn prop_tick_increments_per_step(
            seed       in 0u64..=u64::MAX,
            num_steps  in 1usize..=8usize,
        ) {
            let config = small_config(8, 8, seed, 1);
            let mut world = WorldState::new(config).unwrap();
            prop_assert_eq!(world.tick, 0);

            for i in 1..=(num_steps as u64) {
                world.step(&[Action::Noop]);
                prop_assert_eq!(world.tick, i);
            }
        }

        /// `step` must return one observation and one reward per agent,
        /// regardless of the action slice length (pad/truncate rule).
        #[test]
        fn prop_step_result_length_matches_agents(
            seed       in 0u64..=u64::MAX,
            num_agents in 1u32..=4u32,
            width      in small_dim(),
            height     in small_dim(),
        ) {
            let config = small_config(width, height, seed, num_agents);
            let mut world = WorldState::new(config).unwrap();

            // Test with no actions (all padded to Noop)
            let result = world.step(&[]);
            prop_assert_eq!(result.observations.len(), num_agents as usize);
            prop_assert_eq!(result.rewards.len(),      num_agents as usize);
            prop_assert_eq!(result.info.agents_alive.len(), num_agents as usize);
        }

        /// Two worlds created with the same seed must produce identical agent
        /// positions after the same sequence of actions (determinism invariant).
        #[test]
        fn prop_same_seed_same_outcome(
            seed in 0u64..=u64::MAX,
        ) {
            let config1 = small_config(8, 8, seed, 1);
            let config2 = config1.clone();

            let mut world1 = WorldState::new(config1).unwrap();
            let mut world2 = WorldState::new(config2).unwrap();

            let actions = [
                Action::Move(Direction::Right),
                Action::Move(Direction::Down),
                Action::Noop,
                Action::Move(Direction::Left),
            ];

            for action in &actions {
                world1.step(std::slice::from_ref(action));
                world2.step(std::slice::from_ref(action));
            }

            prop_assert_eq!(world1.tick, world2.tick);
            prop_assert_eq!(
                world1.agents[0].position,
                world2.agents[0].position,
                "positions diverged after identical action sequences"
            );
        }
    }
}
