//! Action map — loaded from `configs/minecraft/action_map.toml`.
//!
//! The action map is the single source of truth for discrete action
//! semantics. Both the Rust client and the Node bot load the same file;
//! the SHA256 of its canonical form is folded into `schema_id` so a
//! mismatch on either side aborts at the handshake.
//!
//! Example TOML:
//! ```toml
//! schema_version = 1
//!
//! [[action]]
//! id = 0
//! kind = "noop"
//! ticks = 1
//!
//! [[action]]
//! id = 1
//! kind = "move"
//! direction = "forward"
//! ticks = 4
//!
//! [[action]]
//! id = 7
//! kind = "place"
//! hotbar_slot = 0
//! ```

use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::McEnvError;

/// Concrete action kinds the bot knows how to execute.
///
/// Keep variants additive — new kinds are a backwards-compatible
/// schema bump only if existing ids retain their meaning. Any change
/// to the canonical form changes the `schema_id` hash.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ActionKind {
    /// Do nothing for `ticks` server ticks.
    Noop {
        /// Number of ticks to idle.
        #[serde(default = "default_ticks")]
        ticks: u32,
    },
    /// Walk in a cardinal direction for `ticks` ticks.
    Move {
        /// "forward" | "back" | "left" | "right".
        direction: String,
        /// Hold the control for this many ticks.
        #[serde(default = "default_ticks")]
        ticks: u32,
    },
    /// Jump for one tick.
    Jump,
    /// Attack (left-click) for one tick.
    Attack,
    /// Use/place (right-click) for one tick.
    Use,
    /// Place a block from the given hotbar slot.
    Place {
        /// Hotbar slot index 0..=8.
        hotbar_slot: u8,
    },
    /// Switch the selected hotbar slot.
    SelectSlot {
        /// Hotbar slot index 0..=8.
        hotbar_slot: u8,
    },
    /// Look (turn camera) by deltas. Continuous control under a discrete cap.
    Look {
        /// Yaw delta in degrees.
        yaw_deg: f32,
        /// Pitch delta in degrees.
        pitch_deg: f32,
    },
}

fn default_ticks() -> u32 {
    1
}

/// One entry in the action map.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionEntry {
    /// Stable id used on the wire. MUST be unique within the file.
    pub id: u32,
    /// What the bot should do for this id.
    #[serde(flatten)]
    pub kind: ActionKind,
}

/// Parsed action map loaded from disk.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionMap {
    /// Map schema version. v1 = the format documented here.
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    /// Action entries — id MUST be unique.
    #[serde(rename = "action", default)]
    pub entries: Vec<ActionEntry>,
}

fn default_schema_version() -> u32 {
    1
}

impl ActionMap {
    /// Load and validate from a TOML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, McEnvError> {
        let path = path.as_ref();
        let raw = std::fs::read_to_string(path)
            .map_err(|e| McEnvError::Config(format!("read {}: {e}", path.display())))?;
        Self::parse_toml(&raw)
    }

    /// Parse from a TOML string. (Named `parse_toml` to avoid the
    /// `FromStr` trait collision flagged by clippy::should_implement_trait.)
    pub fn parse_toml(raw: &str) -> Result<Self, McEnvError> {
        let map: Self = toml::from_str(raw)
            .map_err(|e| McEnvError::Config(format!("parse action map: {e}")))?;
        map.validate()?;
        Ok(map)
    }

    /// Number of distinct actions = max id + 1. Asserts dense ids
    /// starting at 0 to keep the wire format trivial.
    pub fn action_count(&self) -> u32 {
        self.entries.iter().map(|e| e.id + 1).max().unwrap_or(0)
    }

    /// Look up an entry by id.
    pub fn get(&self, id: u32) -> Option<&ActionEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// Validate: ids are unique, dense, and starting at 0.
    pub fn validate(&self) -> Result<(), McEnvError> {
        if self.entries.is_empty() {
            return Err(McEnvError::Config("action map has no entries".into()));
        }
        let mut seen = std::collections::HashSet::new();
        for e in &self.entries {
            if !seen.insert(e.id) {
                return Err(McEnvError::Config(format!("duplicate action id: {}", e.id)));
            }
        }
        // Require a dense id space starting at 0 — the schema_id hash
        // is sensitive to ordering and we don't want callers paving
        // over gaps later.
        let n = self.entries.len() as u32;
        for i in 0..n {
            if !seen.contains(&i) {
                return Err(McEnvError::Config(format!(
                    "action ids must be dense 0..{n}, missing {i}"
                )));
            }
        }
        Ok(())
    }

    /// SHA256 of the canonical (sorted-by-id, serde-json) form. Folded
    /// into the global `schema_id`.
    pub fn canonical_sha256(&self) -> String {
        let mut sorted = self.entries.clone();
        sorted.sort_by_key(|e| e.id);
        let canonical = serde_json::to_string(&sorted).expect("entries serialise");
        let mut hasher = Sha256::new();
        hasher.update(canonical.as_bytes());
        let digest = hasher.finalize();
        hex_encode(&digest)
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_map() -> ActionMap {
        ActionMap {
            schema_version: 1,
            entries: vec![
                ActionEntry {
                    id: 0,
                    kind: ActionKind::Noop { ticks: 1 },
                },
                ActionEntry {
                    id: 1,
                    kind: ActionKind::Move {
                        direction: "forward".into(),
                        ticks: 4,
                    },
                },
                ActionEntry {
                    id: 2,
                    kind: ActionKind::Jump,
                },
            ],
        }
    }

    #[test]
    fn action_count_returns_max_id_plus_one() {
        let m = sample_map();
        assert_eq!(m.action_count(), 3);
    }

    #[test]
    fn validate_rejects_duplicate_ids() {
        let mut m = sample_map();
        m.entries.push(ActionEntry {
            id: 1,
            kind: ActionKind::Attack,
        });
        let err = m.validate().unwrap_err();
        assert!(matches!(err, McEnvError::Config(_)));
    }

    #[test]
    fn validate_rejects_sparse_ids() {
        let m = ActionMap {
            schema_version: 1,
            entries: vec![
                ActionEntry {
                    id: 0,
                    kind: ActionKind::Noop { ticks: 1 },
                },
                ActionEntry {
                    id: 2,
                    kind: ActionKind::Jump,
                },
            ],
        };
        assert!(matches!(m.validate(), Err(McEnvError::Config(_))));
    }

    #[test]
    fn validate_rejects_empty() {
        let m = ActionMap {
            schema_version: 1,
            entries: vec![],
        };
        assert!(matches!(m.validate(), Err(McEnvError::Config(_))));
    }

    #[test]
    fn canonical_sha256_is_stable() {
        let m = sample_map();
        let a = m.canonical_sha256();
        let b = m.canonical_sha256();
        assert_eq!(a, b);
        assert_eq!(a.len(), 64); // hex sha256
    }

    #[test]
    fn canonical_sha256_invariant_under_reorder() {
        let m1 = sample_map();
        let mut m2 = sample_map();
        m2.entries.reverse();
        assert_eq!(m1.canonical_sha256(), m2.canonical_sha256());
    }

    #[test]
    fn loads_from_toml_string() {
        let toml = r#"
schema_version = 1

[[action]]
id = 0
kind = "noop"
ticks = 1

[[action]]
id = 1
kind = "jump"
"#;
        let m = ActionMap::parse_toml(toml).unwrap();
        assert_eq!(m.action_count(), 2);
        assert!(matches!(m.get(1).unwrap().kind, ActionKind::Jump));
    }

    #[test]
    fn get_returns_none_for_unknown_id() {
        let m = sample_map();
        assert!(m.get(99).is_none());
    }

    /// The shipped default action map (`configs/minecraft/action_map.toml`)
    /// MUST parse and validate. This guards against typos in the
    /// canonical config that downstream test/build configs depend on.
    #[test]
    fn ships_default_action_map_parses() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("configs/minecraft/action_map.toml");
        let raw = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let map = ActionMap::parse_toml(&raw).expect("parse default action_map");
        map.validate().unwrap();
        assert!(map.action_count() >= 1);
    }
}
