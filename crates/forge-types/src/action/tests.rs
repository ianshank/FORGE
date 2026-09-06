use super::*;

#[test]
fn test_action_discrete_roundtrip() {
    let vocab_size = 16;
    let actions = vec![
        Action::Noop,
        Action::Move(Direction::Up),
        Action::Move(Direction::Down),
        Action::Move(Direction::Left),
        Action::Move(Direction::Right),
        Action::PickUp,
        Action::Drop(0),
        Action::Drop(5),
        Action::Use(0),
        Action::Use(9),
        Action::Craft(0),
        Action::Craft(4),
        Action::Craft(8),
        Action::Push(Direction::Up),
        Action::Push(Direction::Right),
        Action::Interact,
        Action::Communicate(0),
        Action::Communicate(15),
    ];

    for action in actions {
        let discrete = action.to_discrete();
        let recovered = Action::from_discrete(discrete, vocab_size, false).unwrap();
        assert_eq!(action, recovered, "roundtrip failed for {:?}", action);
    }
}

#[test]
fn test_action_space_size() {
    assert_eq!(Action::space_size(0, false), 40);
    assert_eq!(Action::space_size(16, false), 56);
    assert_eq!(Action::space_size(256, false), 296);
}

#[test]
fn test_invalid_discrete_action() {
    assert!(Action::from_discrete(1000, 16, false).is_none());
}

#[test]
fn test_noop_is_zero() {
    assert_eq!(Action::Noop.to_discrete(), 0);
}

#[test]
fn test_drone_action_discrete_roundtrip() {
    let vocab_size = 16;
    let drone_base = 40 + vocab_size as u32;
    let drone_actions: Vec<(u32, Action)> = vec![
        (drone_base, Action::Ascend),
        (drone_base + 1, Action::Descend),
        (drone_base + 2, Action::Hover),
        (drone_base + 3, Action::TakeOff),
        (drone_base + 4, Action::Land),
        (drone_base + 5, Action::Scan(Direction::Up)),
        (drone_base + 6, Action::Scan(Direction::Down)),
        (drone_base + 7, Action::Scan(Direction::Left)),
        (drone_base + 8, Action::Scan(Direction::Right)),
        (drone_base + 9, Action::DropPayload(0)),
        (drone_base + 18, Action::DropPayload(9)),
    ];
    for (id, expected) in &drone_actions {
        let decoded = Action::from_discrete(*id, vocab_size, true);
        assert_eq!(decoded.as_ref(), Some(expected), "failed for id {}", id);
    }
}

#[test]
fn test_drone_to_discrete_full_roundtrip() {
    let vocab_size = 16;
    let actions = vec![
        Action::Ascend,
        Action::Descend,
        Action::Hover,
        Action::TakeOff,
        Action::Land,
        Action::Scan(Direction::Up),
        Action::Scan(Direction::Down),
        Action::Scan(Direction::Left),
        Action::Scan(Direction::Right),
        Action::DropPayload(0),
        Action::DropPayload(9),
    ];
    for action in &actions {
        let id = action.to_discrete_full(vocab_size);
        let decoded = Action::from_discrete(id, vocab_size, true).unwrap();
        assert_eq!(&decoded, action);
    }
}

#[test]
fn test_space_size_without_drones() {
    assert_eq!(Action::space_size(16, false), 56);
}

#[test]
fn test_space_size_with_drones() {
    assert_eq!(Action::space_size(16, true), 56 + 19);
}

#[test]
fn test_drone_actions_not_decoded_when_disabled() {
    let vocab_size = 16;
    let drone_base = 40 + vocab_size as u32;
    assert!(Action::from_discrete(drone_base, vocab_size, false).is_none());
}

#[test]
fn test_from_discrete_bounds_check() {
    // Beyond space_size should return None
    let max_id = Action::space_size(16, true);
    assert!(Action::from_discrete(max_id, 16, true).is_none());
    assert!(Action::from_discrete(max_id + 1, 16, true).is_none());

    // Last valid action should succeed
    assert!(Action::from_discrete(max_id - 1, 16, true).is_some());
}

#[test]
fn test_to_discrete_full_no_collision_with_comm_tokens() {
    let vocab_size = 16u16;
    // Verify drone actions don't collide with communication tokens
    let comm_ids: Vec<u32> = (0..vocab_size)
        .map(|t| Action::Communicate(t).to_discrete_full(vocab_size))
        .collect();
    let drone_actions = [
        Action::Ascend,
        Action::Descend,
        Action::Hover,
        Action::TakeOff,
        Action::Land,
        Action::Scan(Direction::Up),
        Action::Scan(Direction::Down),
        Action::Scan(Direction::Left),
        Action::Scan(Direction::Right),
        Action::DropPayload(0),
    ];
    for da in &drone_actions {
        let id = da.to_discrete_full(vocab_size);
        assert!(
            !comm_ids.contains(&id),
            "drone action {:?} (id={}) collides with communication token",
            da,
            id
        );
    }
}

#[test]
fn test_space_size_zero_vocab() {
    assert_eq!(Action::space_size(0, false), 40);
    assert_eq!(Action::space_size(0, true), 40 + 19);
}

#[test]
fn test_full_roundtrip_all_actions_with_drones() {
    let vocab_size = 8u16;
    let total = Action::space_size(vocab_size, true);
    for id in 0..total {
        let action = Action::from_discrete(id, vocab_size, true);
        assert!(action.is_some(), "id {} should decode to an action", id);
        let action = action.unwrap();
        let roundtrip_id = action.to_discrete_full(vocab_size);
        assert_eq!(
            roundtrip_id, id,
            "roundtrip failed for action {:?} (expected id={}, got id={})",
            action, id, roundtrip_id
        );
    }
}

#[test]
fn test_hex_roundtrip_without_optional_drone_blocks() {
    let vocab_size = 8u16;
    let base = 40 + vocab_size as u32;
    for (index, dir) in HexDirection::ALL.iter().copied().enumerate() {
        let action = Action::MoveHex(dir);
        let id = action.to_discrete_configured(vocab_size, false, false, true);
        assert_eq!(id, base + index as u32);
        let decoded = Action::from_discrete_full(id, vocab_size, false, false, true).unwrap();
        assert_eq!(decoded, action);
    }
}

#[test]
fn test_hex_roundtrip_after_drone_and_agri_blocks() {
    let vocab_size = 8u16;
    let base = 40
        + vocab_size as u32
        + crate::constants::DRONE_ACTION_COUNT
        + crate::constants::AGRI_ACTION_COUNT;
    for (index, dir) in HexDirection::ALL.iter().copied().enumerate() {
        let action = Action::MoveHex(dir);
        let id = action.to_discrete_configured(vocab_size, true, true, true);
        assert_eq!(id, base + index as u32);
        let decoded = Action::from_discrete_full(id, vocab_size, true, true, true).unwrap();
        assert_eq!(decoded, action);
    }
}

mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn from_discrete_roundtrip(
            vocab_size in 0u16..64,
            action_id in 0u32..200,
        ) {
            let space = Action::space_size(vocab_size, true);
            if action_id < space {
                let action = Action::from_discrete(action_id, vocab_size, true).unwrap();
                let recovered_id = action.to_discrete_full(vocab_size);
                prop_assert_eq!(recovered_id, action_id, "roundtrip failed");
            } else {
                prop_assert!(Action::from_discrete(action_id, vocab_size, true).is_none());
            }
        }

        #[test]
        fn space_size_monotonic_in_vocab(
            vocab_size in 0u16..1024,
        ) {
            let without = Action::space_size(vocab_size, false);
            let with = Action::space_size(vocab_size, true);
            prop_assert!(with > without, "drone actions should increase space size");
            prop_assert_eq!(with - without, crate::constants::DRONE_ACTION_COUNT);
        }

        /// `try_to_discrete_configured` and `to_discrete_configured` agree
        /// on every input that doesn't fail (i.e. their happy paths are
        /// byte-identical). This guards against the panicking path silently
        /// drifting from the fallible path.
        #[test]
        fn try_and_panic_agree_on_happy_path(
            vocab_size in 0u16..32,
            drone in any::<bool>(),
            agri in any::<bool>(),
            hex in any::<bool>(),
            action_id in 0u32..256,
        ) {
            let space = Action::space_size_full(vocab_size, drone, agri, hex);
            if action_id >= space {
                return Ok(());
            }
            let Some(action) = Action::from_discrete_full(action_id, vocab_size, drone, agri, hex) else {
                return Ok(());
            };
            // try_* must succeed for any action that round-trips through the
            // configured layout — those are by construction enabled.
            let try_id = action
                .try_to_discrete_configured(vocab_size, drone, agri, hex)
                .expect("decoded action must re-encode under the same layout");
            let panic_id = action.to_discrete_configured(vocab_size, drone, agri, hex);
            prop_assert_eq!(try_id, panic_id);
            prop_assert_eq!(try_id, action_id);
        }
    }
}

/// Tests for the fallible action encoders ([`Action::try_to_discrete`],
/// [`Action::try_to_discrete_configured`], [`Action::try_to_discrete_full`])
/// and their interplay with [`crate::error::ActionEncodingError`].
mod try_encoder_tests {
    use super::*;
    use crate::error::{ActionEncodingError, ForgeError};

    // ---- try_to_discrete: happy paths ----

    #[test]
    fn try_to_discrete_succeeds_for_base_actions() {
        let cases: &[(Action, u32)] = &[
            (Action::Noop, 0),
            (Action::Move(Direction::Up), 1),
            (Action::Move(Direction::Down), 2),
            (Action::PickUp, 5),
            (Action::Drop(0), 6),
            (Action::Drop(9), 15),
            (Action::Use(0), 16),
            (Action::Craft(0), 26),
            (Action::Push(Direction::Up), 35),
            (Action::Interact, 39),
            (Action::Communicate(0), 40),
            (Action::Communicate(15), 55),
        ];
        for (action, expected) in cases {
            let got = action
                .try_to_discrete()
                .expect("base action must encode without error");
            assert_eq!(got, *expected, "wrong id for {action:?}");
        }
    }

    // ---- try_to_discrete: error paths ----

    #[test]
    fn try_to_discrete_rejects_drone_actions() {
        let drone_actions = [
            Action::Ascend,
            Action::Descend,
            Action::Hover,
            Action::TakeOff,
            Action::Land,
            Action::Scan(Direction::Up),
            Action::DropPayload(0),
        ];
        for action in drone_actions {
            let err = action
                .try_to_discrete()
                .expect_err("drone action must not encode via try_to_discrete");
            assert!(
                matches!(
                    err,
                    ActionEncodingError::DroneActionRequiresFullEncoder { .. }
                ),
                "wrong error variant for {action:?}: {err:?}",
            );
        }
    }

    #[test]
    fn try_to_discrete_rejects_agri_actions() {
        let agri_actions = [
            Action::Spray(0),
            Action::ScanMultispectral,
            Action::ScanThermal,
            Action::RelaySoilData,
            Action::GenerateReport,
        ];
        for action in agri_actions {
            let err = action
                .try_to_discrete()
                .expect_err("agri action must not encode via try_to_discrete");
            assert!(
                matches!(err, ActionEncodingError::AgriActionUnsupported { .. }),
                "wrong error variant for {action:?}: {err:?}",
            );
        }
    }

    #[test]
    fn try_to_discrete_rejects_hex_actions() {
        let action = Action::MoveHex(HexDirection::E);
        let err = action
            .try_to_discrete()
            .expect_err("hex action must not encode via try_to_discrete");
        assert!(
            matches!(err, ActionEncodingError::HexActionUnsupported { .. }),
            "wrong error variant: {err:?}",
        );
    }

    // ---- try_to_discrete_configured: error paths ----

    #[test]
    fn try_configured_rejects_agri_when_agri_disabled() {
        let action = Action::Spray(2);
        let err = action
            .try_to_discrete_configured(8, true, false, false)
            .expect_err("agri must require both drone and agri flags");
        match err {
            ActionEncodingError::AgriActionUnsupported {
                action_name,
                drone_actions_enabled,
                agri_actions_enabled,
            } => {
                assert_eq!(action_name, "Spray");
                assert!(drone_actions_enabled);
                assert!(!agri_actions_enabled);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn try_configured_rejects_agri_when_drone_disabled() {
        let action = Action::ScanMultispectral;
        let err = action
            .try_to_discrete_configured(8, false, true, false)
            .expect_err("agri requires drone infrastructure");
        assert!(matches!(
            err,
            ActionEncodingError::AgriActionUnsupported {
                drone_actions_enabled: false,
                agri_actions_enabled: true,
                ..
            }
        ));
    }

    #[test]
    fn try_configured_rejects_hex_when_hex_disabled() {
        let action = Action::MoveHex(HexDirection::NW);
        let err = action
            .try_to_discrete_configured(8, true, true, false)
            .expect_err("hex must require hex_actions_enabled");
        assert!(matches!(
            err,
            ActionEncodingError::HexActionUnsupported {
                hex_actions_enabled: false,
            }
        ));
    }

    // ---- panicking variants delegate cleanly ----

    #[test]
    #[should_panic(expected = "Action::to_discrete:")]
    fn legacy_to_discrete_still_panics_on_drone() {
        let _ = Action::Ascend.to_discrete();
    }

    #[test]
    #[should_panic(expected = "Action::to_discrete_configured:")]
    fn legacy_to_discrete_configured_still_panics_when_disabled() {
        let _ = Action::Spray(0).to_discrete_configured(8, true, false, false);
    }

    // ---- error type plumbing ----

    #[test]
    fn action_encoding_error_converts_into_forge_error() {
        let err = Action::MoveHex(HexDirection::E)
            .try_to_discrete()
            .unwrap_err();
        let forge_err: ForgeError = err.into();
        assert!(matches!(forge_err, ForgeError::ActionEncoding(_)));
        let msg = forge_err.to_string();
        assert!(
            msg.contains("action encoding error"),
            "missing wrapper prefix in: {msg}",
        );
    }

    #[test]
    fn action_encoding_error_display_includes_action_name() {
        let err = Action::Hover.try_to_discrete().unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Hover"), "expected action name in: {msg}");
    }

    #[test]
    fn try_to_discrete_full_succeeds_for_every_variant() {
        // Every variant is enabled in the canonical full layout, so there
        // are no failure modes; this guards against regressions if a new
        // variant is added without wiring it through.
        let vocab_size = 16u16;
        let space = Action::space_size_full(vocab_size, true, true, true);
        for id in 0..space {
            let action = Action::from_discrete_full(id, vocab_size, true, true, true).unwrap();
            let roundtrip = action
                .try_to_discrete_full(vocab_size)
                .expect("full layout must accept every decoded action");
            assert_eq!(roundtrip, id, "roundtrip failed for {action:?}");
        }
    }

    // ---- Parameter-bounds validation (added per PR #39 review) ----

    /// Without bounds checks, `Action::Drop(slot=10)` would silently
    /// encode to ID 16, which is the slot for `Action::Use(0)` — a
    /// silent collision that corrupts the action distribution. The
    /// fallible encoder must reject the out-of-range slot instead.
    #[test]
    fn try_to_discrete_rejects_drop_slot_out_of_range() {
        let action = Action::Drop(crate::constants::ACTION_DROP_SLOTS as u8);
        let err = action
            .try_to_discrete()
            .expect_err("Drop(slot=ACTION_DROP_SLOTS) must error");
        match err {
            ActionEncodingError::ParameterOutOfRange {
                action_name,
                value,
                max,
            } => {
                assert_eq!(action_name, "Drop");
                assert_eq!(value, crate::constants::ACTION_DROP_SLOTS as u32);
                assert_eq!(max, crate::constants::ACTION_DROP_SLOTS as u32 - 1);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn try_to_discrete_rejects_use_slot_out_of_range() {
        let err = Action::Use(crate::constants::ACTION_USE_SLOTS as u8)
            .try_to_discrete()
            .expect_err("Use(slot=ACTION_USE_SLOTS) must error");
        assert!(matches!(
            err,
            ActionEncodingError::ParameterOutOfRange {
                action_name: "Use",
                ..
            }
        ));
    }

    #[test]
    fn try_to_discrete_rejects_craft_recipe_out_of_range() {
        let err = Action::Craft(crate::constants::ACTION_CRAFT_SLOTS as u16)
            .try_to_discrete()
            .expect_err("Craft(recipe=ACTION_CRAFT_SLOTS) must error");
        assert!(matches!(
            err,
            ActionEncodingError::ParameterOutOfRange {
                action_name: "Craft",
                ..
            }
        ));
    }

    #[test]
    fn try_to_discrete_configured_rejects_communicate_token_out_of_range() {
        let vocab_size = 8u16;
        let err = Action::Communicate(vocab_size)
            .try_to_discrete_configured(vocab_size, true, true, true)
            .expect_err("Communicate(vocab_size) must error");
        match err {
            ActionEncodingError::ParameterOutOfRange {
                action_name,
                value,
                max,
            } => {
                assert_eq!(action_name, "Communicate");
                assert_eq!(value, vocab_size as u32);
                assert_eq!(max, vocab_size as u32 - 1);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn try_to_discrete_configured_rejects_drop_payload_slot_out_of_range() {
        let err = Action::DropPayload(crate::constants::ACTION_DROP_PAYLOAD_SLOTS as u8)
            .try_to_discrete_configured(8, true, true, false)
            .expect_err("DropPayload(slot=10) must error");
        assert!(matches!(
            err,
            ActionEncodingError::ParameterOutOfRange {
                action_name: "DropPayload",
                ..
            }
        ));
    }

    #[test]
    fn try_to_discrete_configured_rejects_spray_slot_out_of_range() {
        let err = Action::Spray(crate::constants::ACTION_SPRAY_SLOTS as u8)
            .try_to_discrete_configured(8, true, true, false)
            .expect_err("Spray(slot=10) must error");
        assert!(matches!(
            err,
            ActionEncodingError::ParameterOutOfRange {
                action_name: "Spray",
                ..
            }
        ));
    }

    /// The `agri_actions_enabled=false` gate must fire BEFORE the slot
    /// bounds check — callers who haven't enabled agri shouldn't see a
    /// confusing "spray slot out of range" error.
    #[test]
    fn agri_unsupported_takes_precedence_over_slot_bounds() {
        let err = Action::Spray(99)
            .try_to_discrete_configured(8, true, false, false)
            .expect_err("must error before reaching slot bounds check");
        assert!(matches!(
            err,
            ActionEncodingError::AgriActionUnsupported { .. }
        ));
    }

    /// Every in-bounds parameter value must encode successfully under the
    /// canonical full layout. Guards against off-by-one regressions in
    /// `param_check`.
    #[test]
    fn try_to_discrete_accepts_every_in_bounds_parameter() {
        let vocab_size = 16u16;
        for slot in 0..crate::constants::ACTION_DROP_SLOTS as u8 {
            Action::Drop(slot)
                .try_to_discrete()
                .expect("Drop(in-bounds) must succeed");
        }
        for slot in 0..crate::constants::ACTION_USE_SLOTS as u8 {
            Action::Use(slot)
                .try_to_discrete()
                .expect("Use(in-bounds) must succeed");
        }
        for recipe in 0..crate::constants::ACTION_CRAFT_SLOTS as u16 {
            Action::Craft(recipe)
                .try_to_discrete()
                .expect("Craft(in-bounds) must succeed");
        }
        for token in 0..vocab_size {
            Action::Communicate(token)
                .try_to_discrete_configured(vocab_size, true, true, true)
                .expect("Communicate(in-bounds) must succeed");
        }
        for slot in 0..crate::constants::ACTION_DROP_PAYLOAD_SLOTS as u8 {
            Action::DropPayload(slot)
                .try_to_discrete_configured(vocab_size, true, true, true)
                .expect("DropPayload(in-bounds) must succeed");
        }
        for slot in 0..crate::constants::ACTION_SPRAY_SLOTS as u8 {
            Action::Spray(slot)
                .try_to_discrete_configured(vocab_size, true, true, true)
                .expect("Spray(in-bounds) must succeed");
        }
    }

    /// The legacy panicking entrypoint must now panic on out-of-range
    /// parameters instead of silently producing a colliding ID.
    #[test]
    #[should_panic(expected = "Action::to_discrete:")]
    fn legacy_to_discrete_panics_on_drop_slot_out_of_range() {
        let _ = Action::Drop(crate::constants::ACTION_DROP_SLOTS as u8).to_discrete();
    }

    #[test]
    #[should_panic(expected = "Action::to_discrete_configured:")]
    fn legacy_to_discrete_configured_panics_on_communicate_token_out_of_range() {
        let _ = Action::Communicate(99).to_discrete_configured(8, true, true, true);
    }

    // ---- Drone-gating regression (Copilot/Devin reviews on PR #39) ----

    /// Without the `drone_check` gate, every drone variant silently
    /// encoded to `40 + comm_vocab_size + offset` even when
    /// `drone_actions_enabled=false` — that ID is *outside* the configured
    /// action space (since `space_size_full(_, false, false, false) ==
    /// 40 + comm_vocab_size`) and disagrees with `from_discrete_full`,
    /// which decodes the same input to `None`. This regression test
    /// pins the contract: every drone variant must come back as
    /// `DroneActionRequiresFullEncoder` when drone actions are disabled.
    #[test]
    fn try_configured_rejects_every_drone_variant_when_drone_disabled() {
        let drone_actions = [
            Action::Ascend,
            Action::Descend,
            Action::Hover,
            Action::TakeOff,
            Action::Land,
            Action::Scan(Direction::Up),
            Action::Scan(Direction::Down),
            Action::Scan(Direction::Left),
            Action::Scan(Direction::Right),
            Action::DropPayload(0),
            Action::DropPayload(9),
        ];
        for action in drone_actions {
            let err = action
                .try_to_discrete_configured(8, false, false, false)
                .expect_err("drone variant must error when drone_actions_enabled=false");
            assert!(
                matches!(
                    err,
                    ActionEncodingError::DroneActionRequiresFullEncoder { .. }
                ),
                "wrong variant for {action:?}: {err:?}",
            );
        }
    }

    /// `try_to_discrete_configured(_, false, _, _)` must agree with
    /// `from_discrete_full(_, false, _, _)`: both encoder and decoder
    /// reject every drone action when drone support is disabled. Without
    /// this guard the encoder produced `Ok(48)` for `Ascend` while the
    /// decoder returned `None` for `from_discrete_full(48, 8, false, _, _)`.
    #[test]
    fn drone_disabled_encoder_decoder_agree() {
        let vocab_size = 8u16;
        // Encoder rejects every drone action.
        for action in [Action::Ascend, Action::Hover, Action::DropPayload(3)] {
            assert!(action
                .try_to_discrete_configured(vocab_size, false, false, false)
                .is_err());
        }
        // Decoder rejects every ID inside the would-be drone block.
        let drone_base = 40 + vocab_size as u32;
        for offset in 0..crate::constants::DRONE_ACTION_COUNT {
            let id = drone_base + offset;
            assert!(
                Action::from_discrete_full(id, vocab_size, false, false, false).is_none(),
                "decoder must return None for id={id} when drone_actions_enabled=false"
            );
        }
    }

    /// The legacy panicking entrypoint must now panic on drone-disabled
    /// configurations instead of silently producing an out-of-space ID.
    #[test]
    #[should_panic(expected = "Action::to_discrete_configured:")]
    fn legacy_to_discrete_configured_panics_when_drone_disabled() {
        let _ = Action::Ascend.to_discrete_configured(8, false, false, false);
    }
}
