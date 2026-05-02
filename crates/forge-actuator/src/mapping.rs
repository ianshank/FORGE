//! Action-id → command-sequence mapping loaded from TOML.
//!
//! This is the only place that holds the deployment's action vocabulary, so
//! reconfiguring the robot for a new domain is a TOML edit rather than a
//! code change.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tracing::{debug, instrument, warn};

use crate::command::ActuatorCommand;
use crate::error::ActuatorError;

/// Whether a [`ActionMapping::resolve`] hit an explicit entry or the
/// configured default fallback.
///
/// Mirrors [`crate::DispatchSource`] but lives on the mapping so callers
/// that don't go through [`crate::MappedActuator`] can still observe the
/// distinction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LookupSource {
    /// The action id had an explicit entry in the mapping.
    Mapped,
    /// The action id was unmapped and the default sequence was returned
    /// (only possible in non-strict mode).
    Default,
}

/// A mapping from FORGE discrete action ids to ordered actuator command
/// sequences.
///
/// # TOML schema
///
/// ```toml
/// [mapping]
/// strict = false                     # if true, unmapped ids return UnknownActionId
///
/// [[mapping.action]]
/// id = 0                             # FORGE action id (Noop here)
/// commands = [
///     { kind = "halt" },
/// ]
///
/// [[mapping.action]]
/// id = 1                             # MoveUp
/// commands = [
///     { kind = "drive_direction", direction = "up", distance_mm = 30 },
/// ]
///
/// [mapping.default]
/// commands = [
///     { kind = "halt" },             # what to do for unmapped ids when strict = false
/// ]
/// ```
///
/// Duplicate `id` entries are rejected at load time so a typo can't silently
/// override the previous mapping.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "ActionMappingFile", into = "ActionMappingFile")]
pub struct ActionMapping {
    entries: BTreeMap<u32, Vec<ActuatorCommand>>,
    default: Vec<ActuatorCommand>,
    strict: bool,
}

impl ActionMapping {
    /// Construct an empty mapping with `Halt` as the default and `strict = false`.
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
            default: vec![ActuatorCommand::Halt],
            strict: false,
        }
    }

    /// Construct a mapping from raw entries. Returns
    /// [`ActuatorError::InvalidMapping`] if the same action id is supplied
    /// more than once.
    ///
    /// `default` is the command sequence emitted for action ids that have no
    /// entry. Ignored when `strict = true` (unknown ids return
    /// [`ActuatorError::UnknownActionId`] instead).
    pub fn try_from_entries(
        entries: impl IntoIterator<Item = (u32, Vec<ActuatorCommand>)>,
        default: Vec<ActuatorCommand>,
        strict: bool,
    ) -> Result<Self, ActuatorError> {
        let entries = collect_unique_entries(entries)?;
        Ok(Self {
            entries,
            default,
            strict,
        })
    }

    /// Parse a mapping from a TOML string.
    #[instrument(skip_all)]
    pub fn from_toml_str(toml_str: &str) -> Result<Self, ActuatorError> {
        let parsed: ActionMappingFile = toml::from_str(toml_str).map_err(|e| {
            warn!(error = %e, "ActionMapping::from_toml_str: parse failure");
            ActuatorError::ParseToml(e)
        })?;
        Self::try_from_file(parsed)
    }

    /// Load a mapping from a TOML file on disk.
    #[instrument(skip_all, fields(path = %path.as_ref().display()))]
    pub fn from_toml_file<P: AsRef<Path>>(path: P) -> Result<Self, ActuatorError> {
        let path_buf: PathBuf = path.as_ref().to_path_buf();
        let display_path = path_buf.display().to_string();
        let contents = std::fs::read_to_string(&path_buf).map_err(|e| {
            warn!(path = %display_path, error = %e, "ActionMapping: failed to read file");
            ActuatorError::Io {
                path: display_path.clone(),
                source: e,
            }
        })?;
        let mapping = Self::from_toml_str(&contents)?;
        debug!(path = %display_path, entries = mapping.len(), strict = mapping.strict, "ActionMapping: loaded mapping from file");
        Ok(mapping)
    }

    /// Look up the command sequence for an action id.
    ///
    /// In strict mode, unmapped ids yield [`ActuatorError::UnknownActionId`].
    /// In permissive mode, unmapped ids yield the default sequence.
    pub fn commands_for(&self, action_id: u32) -> Result<&[ActuatorCommand], ActuatorError> {
        Ok(self.resolve(action_id)?.0)
    }

    /// Look up the command sequence and report whether it came from an
    /// explicit mapping entry or the default fallback.
    ///
    /// One [`BTreeMap`] lookup per call — used by [`crate::MappedActuator`] on
    /// its dispatch hot path so the source flag and the command slice come
    /// from a single tree probe.
    pub fn resolve(
        &self,
        action_id: u32,
    ) -> Result<(&[ActuatorCommand], LookupSource), ActuatorError> {
        match self.entries.get(&action_id) {
            Some(commands) => Ok((commands, LookupSource::Mapped)),
            None if self.strict => Err(ActuatorError::UnknownActionId { action_id }),
            None => Ok((&self.default, LookupSource::Default)),
        }
    }

    /// Returns whether the mapping is in strict mode.
    pub fn is_strict(&self) -> bool {
        self.strict
    }

    /// Returns the number of explicit action-id entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true when there are no explicit entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn try_from_file(file: ActionMappingFile) -> Result<Self, ActuatorError> {
        let entries =
            collect_unique_entries(file.mapping.action.into_iter().map(|e| (e.id, e.commands)))?;
        let default = file.mapping.default.commands;
        if default.is_empty() && !file.mapping.strict {
            return Err(ActuatorError::InvalidMapping(
                "non-strict mapping requires a non-empty [mapping.default] commands list"
                    .to_string(),
            ));
        }
        Ok(Self {
            entries,
            default,
            strict: file.mapping.strict,
        })
    }
}

/// Collect action-id entries into a `BTreeMap`, rejecting duplicates with a
/// uniform error message. Shared by `try_from_entries` and `try_from_file`
/// so the duplicate-detection rule is defined exactly once.
fn collect_unique_entries(
    entries: impl IntoIterator<Item = (u32, Vec<ActuatorCommand>)>,
) -> Result<BTreeMap<u32, Vec<ActuatorCommand>>, ActuatorError> {
    let mut map: BTreeMap<u32, Vec<ActuatorCommand>> = BTreeMap::new();
    for (id, commands) in entries {
        if map.insert(id, commands).is_some() {
            return Err(ActuatorError::InvalidMapping(format!(
                "duplicate action id {id} in mapping"
            )));
        }
    }
    Ok(map)
}

impl Default for ActionMapping {
    fn default() -> Self {
        Self::new()
    }
}

// ---- Internal serde-only file shape ----

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ActionMappingFile {
    mapping: MappingBlock,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct MappingBlock {
    #[serde(default)]
    strict: bool,
    #[serde(default)]
    action: Vec<MappingEntry>,
    #[serde(default)]
    default: DefaultBlock,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct MappingEntry {
    id: u32,
    commands: Vec<ActuatorCommand>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
struct DefaultBlock {
    #[serde(default)]
    commands: Vec<ActuatorCommand>,
}

impl TryFrom<ActionMappingFile> for ActionMapping {
    type Error = ActuatorError;

    fn try_from(file: ActionMappingFile) -> Result<Self, Self::Error> {
        // serde uses this via `#[serde(try_from = "ActionMappingFile")]`, so a
        // validation failure surfaces as a deserialization error to the
        // caller instead of being silently downgraded to an empty mapping.
        Self::try_from_file(file)
    }
}

impl From<ActionMapping> for ActionMappingFile {
    fn from(mapping: ActionMapping) -> Self {
        Self {
            mapping: MappingBlock {
                strict: mapping.strict,
                action: mapping
                    .entries
                    .into_iter()
                    .map(|(id, commands)| MappingEntry { id, commands })
                    .collect(),
                default: DefaultBlock {
                    commands: mapping.default,
                },
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::CardinalDirection;

    fn sample_kitchen_toml() -> &'static str {
        r#"
[mapping]
strict = false

[[mapping.action]]
id = 0
commands = [{ kind = "halt" }]

[[mapping.action]]
id = 1
commands = [{ kind = "drive_direction", direction = "up", distance_mm = 30 }]

[[mapping.action]]
id = 5
commands = [{ kind = "close_gripper" }]

[[mapping.action]]
id = 16
commands = [{ kind = "engage_sweeper" }]

[[mapping.action]]
id = 17
commands = [
    { kind = "disengage_sweeper" },
    { kind = "halt" },
]

[mapping.default]
commands = [{ kind = "halt" }]
"#
    }

    #[test]
    fn from_toml_str_parses_kitchen_sample() {
        let mapping = ActionMapping::from_toml_str(sample_kitchen_toml()).unwrap();
        assert_eq!(mapping.len(), 5);
        assert!(!mapping.is_strict());
        assert_eq!(
            mapping.commands_for(1).unwrap(),
            &[ActuatorCommand::DriveDirection {
                direction: CardinalDirection::Up,
                distance_mm: 30,
            }]
        );
        assert_eq!(
            mapping.commands_for(17).unwrap(),
            &[ActuatorCommand::DisengageSweeper, ActuatorCommand::Halt,]
        );
    }

    #[test]
    fn unmapped_id_in_permissive_mode_returns_default() {
        let mapping = ActionMapping::from_toml_str(sample_kitchen_toml()).unwrap();
        // Action id 999 is not in the sample mapping.
        let cmds = mapping.commands_for(999).unwrap();
        assert_eq!(cmds, &[ActuatorCommand::Halt]);
    }

    #[test]
    fn unmapped_id_in_strict_mode_returns_unknown_action_id() {
        let toml_str = r#"
[mapping]
strict = true

[[mapping.action]]
id = 1
commands = [{ kind = "drive_direction", direction = "up", distance_mm = 30 }]

[mapping.default]
commands = []
"#;
        let mapping = ActionMapping::from_toml_str(toml_str).unwrap();
        assert!(mapping.is_strict());
        let err = mapping.commands_for(2).unwrap_err();
        match err {
            ActuatorError::UnknownActionId { action_id } => assert_eq!(action_id, 2),
            other => panic!("expected UnknownActionId, got {other:?}"),
        }
    }

    #[test]
    fn duplicate_action_id_is_rejected() {
        let toml_str = r#"
[mapping]
strict = false

[[mapping.action]]
id = 1
commands = [{ kind = "halt" }]

[[mapping.action]]
id = 1
commands = [{ kind = "open_gripper" }]

[mapping.default]
commands = [{ kind = "halt" }]
"#;
        let err = ActionMapping::from_toml_str(toml_str).unwrap_err();
        match err {
            ActuatorError::InvalidMapping(msg) => assert!(msg.contains("duplicate")),
            other => panic!("expected InvalidMapping, got {other:?}"),
        }
    }

    #[test]
    fn permissive_mapping_without_default_is_rejected() {
        let toml_str = r#"
[mapping]
strict = false

[[mapping.action]]
id = 1
commands = [{ kind = "halt" }]

[mapping.default]
commands = []
"#;
        let err = ActionMapping::from_toml_str(toml_str).unwrap_err();
        match err {
            ActuatorError::InvalidMapping(msg) => assert!(msg.contains("default")),
            other => panic!("expected InvalidMapping, got {other:?}"),
        }
    }

    #[test]
    fn strict_mapping_without_default_is_accepted() {
        let toml_str = r#"
[mapping]
strict = true

[[mapping.action]]
id = 1
commands = [{ kind = "halt" }]

[mapping.default]
commands = []
"#;
        let mapping = ActionMapping::from_toml_str(toml_str).unwrap();
        assert!(mapping.is_strict());
        // Strict mode reaches the unknown-id error path instead of the
        // default sequence; verify that contract here rather than poking at
        // the internal default field.
        let err = mapping.resolve(99).unwrap_err();
        assert!(matches!(
            err,
            ActuatorError::UnknownActionId { action_id: 99 }
        ));
    }

    #[test]
    fn try_from_entries_rejects_duplicates() {
        let err = ActionMapping::try_from_entries(
            [
                (1, vec![ActuatorCommand::Halt]),
                (1, vec![ActuatorCommand::OpenGripper]),
            ],
            vec![ActuatorCommand::Halt],
            false,
        )
        .unwrap_err();
        assert!(matches!(err, ActuatorError::InvalidMapping(_)));
    }

    #[test]
    fn from_toml_file_reads_disk() {
        let toml_str = sample_kitchen_toml();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mapping.toml");
        std::fs::write(&path, toml_str).unwrap();
        let mapping = ActionMapping::from_toml_file(&path).unwrap();
        assert_eq!(mapping.len(), 5);
    }

    #[test]
    fn from_toml_file_missing_file_is_io_error() {
        let err = ActionMapping::from_toml_file("/nonexistent/path/mapping.toml").unwrap_err();
        match err {
            ActuatorError::Io { path, .. } => assert!(path.contains("nonexistent")),
            other => panic!("expected Io, got {other:?}"),
        }
    }

    #[test]
    fn malformed_toml_returns_parse_error() {
        let err = ActionMapping::from_toml_str("not [valid toml").unwrap_err();
        assert!(matches!(err, ActuatorError::ParseToml(_)));
    }

    #[test]
    fn default_constructor_is_permissive_with_halt() {
        let mapping = ActionMapping::new();
        assert!(!mapping.is_strict());
        assert!(mapping.is_empty());
        assert_eq!(mapping.commands_for(0).unwrap(), &[ActuatorCommand::Halt]);
    }

    #[test]
    fn resolve_reports_lookup_source_in_a_single_call() {
        let mapping = ActionMapping::from_toml_str(sample_kitchen_toml()).unwrap();

        let (cmds, source) = mapping.resolve(17).unwrap();
        assert_eq!(source, LookupSource::Mapped);
        assert_eq!(
            cmds,
            &[ActuatorCommand::DisengageSweeper, ActuatorCommand::Halt]
        );

        let (cmds, source) = mapping.resolve(999).unwrap();
        assert_eq!(source, LookupSource::Default);
        assert_eq!(cmds, &[ActuatorCommand::Halt]);
    }

    #[test]
    fn resolve_strict_mode_returns_error_for_unmapped_id() {
        let mapping = ActionMapping::try_from_entries(
            [(1u32, vec![ActuatorCommand::OpenGripper])],
            vec![],
            true,
        )
        .unwrap();
        let err = mapping.resolve(99).unwrap_err();
        assert!(matches!(
            err,
            ActuatorError::UnknownActionId { action_id: 99 }
        ));
    }

    #[test]
    fn action_mapping_round_trips_through_toml_serialization() {
        // Serialise → parse must be lossless. This exercises both
        // `From<ActionMapping> for ActionMappingFile` (the `into = ...`
        // direction of the serde wiring) and the matching `try_from` path
        // on the way back. Without this test the `into` path is dead code.
        let original = ActionMapping::try_from_entries(
            [
                (1u32, vec![ActuatorCommand::OpenGripper]),
                (
                    17,
                    vec![ActuatorCommand::DisengageSweeper, ActuatorCommand::Halt],
                ),
            ],
            vec![ActuatorCommand::Halt],
            false,
        )
        .unwrap();

        let serialised = toml::to_string(&original).expect("ActionMapping must serialise to TOML");
        let restored: ActionMapping = toml::from_str(&serialised)
            .expect("serialised TOML must parse back into ActionMapping");

        assert_eq!(restored.len(), original.len());
        assert!(!restored.is_strict());
        assert_eq!(
            restored.commands_for(17).unwrap(),
            &[ActuatorCommand::DisengageSweeper, ActuatorCommand::Halt]
        );
        // Permissive default survives the round trip.
        assert_eq!(
            restored.commands_for(999).unwrap(),
            &[ActuatorCommand::Halt]
        );
    }

    #[test]
    fn serde_try_from_surfaces_validation_errors_instead_of_silent_fallback() {
        // A round-trip through `toml::from_str::<ActionMapping>` (skipping our
        // own validating loader) must surface a deserialization error when
        // the input is invalid — not silently produce an empty mapping.
        let bad_toml = r#"
[mapping]
strict = false

[[mapping.action]]
id = 1
commands = [{ kind = "halt" }]

[[mapping.action]]
id = 1
commands = [{ kind = "open_gripper" }]

[mapping.default]
commands = [{ kind = "halt" }]
"#;
        let result: Result<ActionMapping, _> = toml::from_str(bad_toml);
        assert!(
            result.is_err(),
            "duplicate-id TOML must surface as a serde error, got {:?}",
            result.map(|m| m.len())
        );
        // The error message must mention the validation failure (duplicate id),
        // not just a generic structural parse error.
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("duplicate"),
            "expected validation message in serde error, got: {msg}"
        );
    }
}
