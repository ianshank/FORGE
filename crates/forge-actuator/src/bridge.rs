//! Bridge traits and the generic [`MappedActuator`] implementation.

use tracing::{debug, info, instrument, warn};

use crate::command::ActuatorCommand;
use crate::error::ActuatorError;
use crate::mapping::{ActionMapping, LookupSource};

/// Anything that can translate a FORGE discrete `action_id` into hardware
/// effects.
///
/// Implementors are expected to be stateful: they track command history and
/// fail-fast on driver errors so the upstream policy loop can fall back to
/// the EdgeAgent's safe-pose action.
pub trait ActuatorBridge {
    /// Dispatch a single discrete action id, executing the mapped command
    /// sequence on the underlying driver.
    ///
    /// Returns a [`DispatchResult`] describing the action id, the command
    /// sequence that was actually executed, and whether it came from an
    /// explicit mapping entry or the default fallback.
    ///
    /// # Errors
    ///
    /// Returns an [`ActuatorError`] if the mapping lookup fails (strict
    /// mode + unknown id) or if the underlying driver rejects any command.
    /// In the latter case, commands executed before the failure are still
    /// recorded in the bridge's history.
    fn dispatch(&mut self, action_id: u32) -> Result<DispatchResult, ActuatorError>;

    /// Returns the most recent dispatched commands, oldest-first, up to the
    /// bridge's history capacity.
    fn history(&self) -> &[ActuatorCommand];
}

/// A hardware (or simulated) effector that knows how to execute a single
/// [`ActuatorCommand`].
///
/// Implementations marshal commands onto whatever interface the device
/// requires (serial, GPIO, ROS topic, MQTT, etc.). This crate ships a
/// [`MockDriver`] for tests; production drivers live outside the workspace
/// or in a sibling crate.
pub trait ActuatorDriver {
    /// Execute the command on the underlying hardware.
    ///
    /// Returns [`ActuatorError::DriverFailure`] for transient errors so the
    /// caller can route to a safe-pose action without crashing the agent
    /// loop.
    fn execute(&mut self, command: &ActuatorCommand) -> Result<(), ActuatorError>;
}

/// Whether a dispatched command sequence came from an explicit mapping entry
/// or the configured default fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchSource {
    /// The action id had an explicit entry in the [`ActionMapping`].
    Mapped,
    /// The action id was unmapped and the default sequence was used
    /// (only possible in non-strict mode).
    Default,
}

impl From<LookupSource> for DispatchSource {
    fn from(source: LookupSource) -> Self {
        match source {
            LookupSource::Mapped => DispatchSource::Mapped,
            LookupSource::Default => DispatchSource::Default,
        }
    }
}

/// Outcome of a single [`ActuatorBridge::dispatch`] call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DispatchResult {
    /// FORGE discrete action id that was dispatched.
    pub action_id: u32,
    /// The full command sequence that the driver executed.
    pub commands: Vec<ActuatorCommand>,
    /// Whether the mapping returned an explicit entry or the default.
    pub source: DispatchSource,
}

/// Generic config-driven actuator bridge.
///
/// Holds an [`ActionMapping`] (loaded from TOML) plus an [`ActuatorDriver`]
/// (typically wrapping a serial bus or GPIO). On each `dispatch`, looks the
/// action id up in the mapping and runs the resulting command sequence on
/// the driver. Maintains a bounded ring buffer of recent commands for
/// debugging.
///
/// This is the only bridge implementation in the crate — domain
/// configuration (kitchen counter, patrol pickup, agri sampling) is encoded
/// in the mapping TOML, not in separate bridge subtypes.
pub struct MappedActuator<D: ActuatorDriver> {
    mapping: ActionMapping,
    driver: D,
    history: Vec<ActuatorCommand>,
    history_capacity: usize,
}

impl<D: ActuatorDriver> MappedActuator<D> {
    /// Construct a new bridge with the given mapping, driver, and history
    /// capacity. A capacity of zero disables history tracking entirely.
    pub fn new(mapping: ActionMapping, driver: D, history_capacity: usize) -> Self {
        Self {
            mapping,
            driver,
            history: Vec::with_capacity(history_capacity),
            history_capacity,
        }
    }

    /// Borrow the underlying mapping (e.g. for diagnostics).
    pub fn mapping(&self) -> &ActionMapping {
        &self.mapping
    }

    /// Replace the mapping at runtime (e.g. after an OTA update). Clears the
    /// command history so subsequent debugging output reflects only the new
    /// mapping's behaviour.
    pub fn replace_mapping(&mut self, mapping: ActionMapping) {
        info!(
            old_entries = self.mapping.len(),
            new_entries = mapping.len(),
            "MappedActuator: replacing action mapping"
        );
        self.mapping = mapping;
        self.history.clear();
    }

    /// Borrow the underlying driver immutably.
    pub fn driver(&self) -> &D {
        &self.driver
    }

    /// Borrow the underlying driver mutably.
    pub fn driver_mut(&mut self) -> &mut D {
        &mut self.driver
    }

    /// Returns the configured history capacity.
    pub fn history_capacity(&self) -> usize {
        self.history_capacity
    }

    fn record(&mut self, command: &ActuatorCommand) {
        if self.history_capacity == 0 {
            return;
        }
        if self.history.len() >= self.history_capacity {
            // Drop the oldest entry to make room. History capacities are
            // expected to be small (4–64), so the O(n) shift is negligible
            // relative to the actuator dispatch itself.
            self.history.remove(0);
        }
        self.history.push(command.clone());
    }
}

impl<D: ActuatorDriver> ActuatorBridge for MappedActuator<D> {
    #[instrument(skip(self), fields(action_id))]
    fn dispatch(&mut self, action_id: u32) -> Result<DispatchResult, ActuatorError> {
        let (commands_slice, lookup_source) = self.mapping.resolve(action_id)?;
        let source = DispatchSource::from(lookup_source);
        let commands: Vec<ActuatorCommand> = commands_slice.to_vec();
        debug!(
            action_id,
            ?source,
            command_count = commands.len(),
            "MappedActuator::dispatch"
        );
        for command in &commands {
            if let Err(e) = self.driver.execute(command) {
                warn!(
                    action_id,
                    ?command,
                    error = %e,
                    "MappedActuator: driver failed mid-sequence"
                );
                self.record(command);
                return Err(e);
            }
            self.record(command);
        }
        Ok(DispatchResult {
            action_id,
            commands,
            source,
        })
    }

    fn history(&self) -> &[ActuatorCommand] {
        &self.history
    }
}

// ---- Mock driver for tests and examples ----

/// Closure type used by [`MockDriver`] to inject command failures.
///
/// Returning `Some(reason)` causes the next `execute` call to fail with
/// [`ActuatorError::DriverFailure(reason)`]; `None` accepts the command.
pub type MockFailurePredicate = Box<dyn FnMut(&ActuatorCommand) -> Option<String> + Send>;

/// In-memory [`ActuatorDriver`] that records every command it receives.
///
/// Useful for unit tests, examples, and deterministic CI runs where no
/// physical hardware is available. Optionally injects failures for chosen
/// commands so callers can exercise error paths.
#[derive(Default)]
pub struct MockDriver {
    received: Vec<ActuatorCommand>,
    failure_predicate: Option<MockFailurePredicate>,
}

impl std::fmt::Debug for MockDriver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockDriver")
            .field("received", &self.received)
            .field(
                "failure_predicate",
                &self.failure_predicate.as_ref().map(|_| "<closure>"),
            )
            .finish()
    }
}

impl MockDriver {
    /// Construct a new mock driver that accepts all commands.
    pub fn new() -> Self {
        Self::default()
    }

    /// Install a predicate that returns `Some(reason)` to fail the command
    /// or `None` to accept it.
    pub fn with_failure_predicate<F>(mut self, f: F) -> Self
    where
        F: FnMut(&ActuatorCommand) -> Option<String> + Send + 'static,
    {
        self.failure_predicate = Some(Box::new(f));
        self
    }

    /// Returns every command received, in dispatch order.
    pub fn received(&self) -> &[ActuatorCommand] {
        &self.received
    }

    /// Clears the received-command buffer.
    pub fn clear(&mut self) {
        self.received.clear();
    }
}

impl ActuatorDriver for MockDriver {
    fn execute(&mut self, command: &ActuatorCommand) -> Result<(), ActuatorError> {
        if let Some(pred) = &mut self.failure_predicate {
            if let Some(reason) = pred(command) {
                return Err(ActuatorError::DriverFailure(reason));
            }
        }
        self.received.push(command.clone());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::CardinalDirection;

    fn kitchen_mapping() -> ActionMapping {
        ActionMapping::try_from_entries(
            [
                (0u32, vec![ActuatorCommand::Halt]),
                (
                    1,
                    vec![ActuatorCommand::DriveDirection {
                        direction: CardinalDirection::Up,
                        distance_mm: 30,
                    }],
                ),
                (5, vec![ActuatorCommand::CloseGripper]),
                (16, vec![ActuatorCommand::EngageSweeper]),
                (
                    17,
                    vec![ActuatorCommand::DisengageSweeper, ActuatorCommand::Halt],
                ),
                (
                    35,
                    vec![ActuatorCommand::DriveDirection {
                        direction: CardinalDirection::Up,
                        distance_mm: 30,
                    }],
                ),
                (
                    39,
                    vec![ActuatorCommand::Custom {
                        name: "sink_edge_sweep".to_string(),
                        payload: None,
                    }],
                ),
            ],
            vec![ActuatorCommand::Halt],
            false,
        )
        .unwrap()
    }

    #[test]
    fn dispatch_runs_mapped_sequence_in_order() {
        let mut bridge = MappedActuator::new(kitchen_mapping(), MockDriver::new(), 16);
        let result = bridge.dispatch(17).unwrap();
        assert_eq!(result.action_id, 17);
        assert_eq!(result.source, DispatchSource::Mapped);
        assert_eq!(
            result.commands,
            vec![ActuatorCommand::DisengageSweeper, ActuatorCommand::Halt]
        );
        assert_eq!(
            bridge.driver().received(),
            &[ActuatorCommand::DisengageSweeper, ActuatorCommand::Halt]
        );
    }

    #[test]
    fn dispatch_falls_through_to_default_for_unmapped_id() {
        let mut bridge = MappedActuator::new(kitchen_mapping(), MockDriver::new(), 16);
        let result = bridge.dispatch(999).unwrap();
        assert_eq!(result.source, DispatchSource::Default);
        assert_eq!(result.commands, vec![ActuatorCommand::Halt]);
    }

    #[test]
    fn dispatch_strict_mode_returns_error_for_unmapped_id() {
        let mapping = ActionMapping::try_from_entries(
            [(1u32, vec![ActuatorCommand::OpenGripper])],
            vec![],
            true,
        )
        .unwrap();
        let mut bridge = MappedActuator::new(mapping, MockDriver::new(), 4);
        let err = bridge.dispatch(2).unwrap_err();
        match err {
            ActuatorError::UnknownActionId { action_id } => assert_eq!(action_id, 2),
            other => panic!("expected UnknownActionId, got {other:?}"),
        }
    }

    #[test]
    fn driver_error_aborts_sequence_and_records_failed_command() {
        let mapping = ActionMapping::try_from_entries(
            [(
                17u32,
                vec![ActuatorCommand::DisengageSweeper, ActuatorCommand::Halt],
            )],
            vec![ActuatorCommand::Halt],
            false,
        )
        .unwrap();

        // Fail the second command in the sequence.
        let mut count = 0;
        let driver = MockDriver::new().with_failure_predicate(move |_cmd| {
            count += 1;
            if count == 2 {
                Some("simulated torque trip".to_string())
            } else {
                None
            }
        });

        let mut bridge = MappedActuator::new(mapping, driver, 16);
        let err = bridge.dispatch(17).unwrap_err();
        assert!(matches!(err, ActuatorError::DriverFailure(_)));
        // The first command succeeded and was recorded by the driver; the
        // second was attempted (recorded by the bridge history) but failed.
        assert_eq!(
            bridge.driver().received(),
            &[ActuatorCommand::DisengageSweeper]
        );
        assert_eq!(
            bridge.history(),
            &[ActuatorCommand::DisengageSweeper, ActuatorCommand::Halt,]
        );
    }

    #[test]
    fn history_respects_capacity() {
        let mut bridge = MappedActuator::new(kitchen_mapping(), MockDriver::new(), 3);
        for _ in 0..5 {
            // Action 17 emits 2 commands; 5 dispatches → 10 commands → trimmed to 3.
            bridge.dispatch(17).unwrap();
        }
        assert_eq!(bridge.history().len(), 3);
    }

    #[test]
    fn history_capacity_zero_disables_tracking() {
        let mut bridge = MappedActuator::new(kitchen_mapping(), MockDriver::new(), 0);
        bridge.dispatch(17).unwrap();
        assert!(bridge.history().is_empty());
        // Driver still received the commands.
        assert_eq!(bridge.driver().received().len(), 2);
    }

    #[test]
    fn replace_mapping_clears_history() {
        let mut bridge = MappedActuator::new(kitchen_mapping(), MockDriver::new(), 16);
        bridge.dispatch(17).unwrap();
        assert!(!bridge.history().is_empty());

        bridge.replace_mapping(ActionMapping::new());
        assert!(bridge.history().is_empty());
        // New mapping (the empty default with Halt) should still dispatch.
        let result = bridge.dispatch(17).unwrap();
        assert_eq!(result.source, DispatchSource::Default);
        assert_eq!(result.commands, vec![ActuatorCommand::Halt]);
    }

    #[test]
    fn dispatch_records_source_for_observability() {
        let mut bridge = MappedActuator::new(kitchen_mapping(), MockDriver::new(), 16);
        let mapped = bridge.dispatch(5).unwrap();
        let defaulted = bridge.dispatch(255).unwrap();
        assert_eq!(mapped.source, DispatchSource::Mapped);
        assert_eq!(defaulted.source, DispatchSource::Default);
    }

    #[test]
    fn driver_mut_allows_clearing_received_buffer_between_tests() {
        let mut bridge = MappedActuator::new(kitchen_mapping(), MockDriver::new(), 16);
        bridge.dispatch(5).unwrap();
        assert_eq!(bridge.driver().received().len(), 1);
        bridge.driver_mut().clear();
        assert!(bridge.driver().received().is_empty());
    }

    // ---- Property tests ----

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn dispatch_never_panics_for_arbitrary_action_id(action_id in 0u32..200u32) {
            let mut bridge = MappedActuator::new(kitchen_mapping(), MockDriver::new(), 8);
            // Either Ok (mapped or default) or a recognised error variant.
            let _ = bridge.dispatch(action_id);
        }

        #[test]
        fn unmapped_ids_in_permissive_mode_always_use_default(action_id in 100u32..u32::MAX) {
            // Action ids in this range are never in the kitchen mapping, so
            // they must always come back as DispatchSource::Default with the
            // configured Halt sequence.
            let mut bridge = MappedActuator::new(kitchen_mapping(), MockDriver::new(), 4);
            let result = bridge.dispatch(action_id).unwrap();
            prop_assert_eq!(result.source, DispatchSource::Default);
            prop_assert_eq!(result.commands, vec![ActuatorCommand::Halt]);
        }
    }
}
