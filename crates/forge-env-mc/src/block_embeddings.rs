//! Block-id embedding vocabulary loaded from
//! `configs/minecraft/block_embeddings.toml`.
//!
//! This is an **observation-layout** pin, not part of today's two-input
//! `schema_id` (action map + rewards). Changing the table changes channel
//! semantics of the grid without invalidating reward-only replay. The
//! JS twin is `mc-bot/src/block_embeddings.ts`; the Python twin lives
//! in `python/forge/training/muzero_mc/schema_id.py`.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::McEnvError;
use crate::hash_util::canonical_json_sha256;

/// Named fallback when the TOML is missing. Matches the shipped
/// `unknown = 35` vocab (`max(index) + 1`).
pub const DEFAULT_NUM_BLOCK_EMBEDDINGS: usize = 36;

/// Parsed `[blocks]` table: Minecraft block name → contiguous embedding index.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockEmbeddings {
    /// Map of Minecraft block name to embedding index.
    #[serde(default)]
    pub blocks: BTreeMap<String, u32>,
}

impl BlockEmbeddings {
    /// Load and validate from a TOML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, McEnvError> {
        let path = path.as_ref();
        let raw = std::fs::read_to_string(path)
            .map_err(|e| McEnvError::Config(format!("read {}: {e}", path.display())))?;
        Self::parse_toml(&raw)
    }

    /// Parse from a TOML string. Requires a non-empty `[blocks]` table.
    pub fn parse_toml(raw: &str) -> Result<Self, McEnvError> {
        let cfg: Self = toml::from_str(raw)
            .map_err(|e| McEnvError::Config(format!("parse block embeddings toml: {e}")))?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Reject an empty vocabulary.
    pub fn validate(&self) -> Result<(), McEnvError> {
        if self.blocks.is_empty() {
            return Err(McEnvError::Config(
                "block embeddings config has no [blocks] entries".into(),
            ));
        }
        Ok(())
    }

    /// Vocab size: `max(index) + 1` so sparse tables still cover every
    /// index the encoder can emit.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.blocks
            .values()
            .copied()
            .max()
            .map(|m| m as usize + 1)
            .unwrap_or(0)
    }

    /// SHA256 of the canonical JSON form of the `[blocks]` table.
    ///
    /// Keys are alphabetical (`BTreeMap`); values are integers. JS
    /// `sortKeysDeep` + `JSON.stringify` and Python
    /// `_toml_to_canonical_json` + `json.dumps(separators=(",", ":"))`
    /// must produce the same bytes.
    #[must_use]
    pub fn canonical_sha256(&self) -> String {
        let table: toml::map::Map<String, toml::Value> = self
            .blocks
            .iter()
            .map(|(k, v)| (k.clone(), toml::Value::Integer(i64::from(*v))))
            .collect();
        canonical_json_sha256(&toml::Value::Table(table))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_toml() -> &'static str {
        r#"
[blocks]
air = 0
stone = 1
unknown = 2
"#
    }

    #[test]
    fn parses_sample_toml() {
        let cfg = BlockEmbeddings::parse_toml(sample_toml()).unwrap();
        assert_eq!(cfg.blocks.len(), 3);
        assert_eq!(cfg.vocab_size(), 3);
    }

    #[test]
    fn validate_rejects_empty_blocks() {
        let raw = "schema_version = 1\n";
        assert!(matches!(
            BlockEmbeddings::parse_toml(raw),
            Err(McEnvError::Config(_))
        ));
    }

    #[test]
    fn vocab_size_uses_max_index_plus_one() {
        let cfg =
            BlockEmbeddings::parse_toml("[blocks]\nair = 0\nunknown = 7\nstone = 1\n").unwrap();
        assert_eq!(cfg.vocab_size(), 8);
    }

    #[test]
    fn canonical_sha256_is_stable_and_order_invariant() {
        let a = BlockEmbeddings::parse_toml(sample_toml())
            .unwrap()
            .canonical_sha256();
        let reordered = "[blocks]\nunknown = 2\nair = 0\nstone = 1\n";
        let b = BlockEmbeddings::parse_toml(reordered)
            .unwrap()
            .canonical_sha256();
        assert_eq!(a, b);
        assert_eq!(a.len(), 64);
    }

    #[test]
    fn canonical_sha256_changes_when_an_index_changes() {
        let a = BlockEmbeddings::parse_toml(sample_toml())
            .unwrap()
            .canonical_sha256();
        let b = BlockEmbeddings::parse_toml("[blocks]\nair = 0\nstone = 1\nunknown = 3\n")
            .unwrap()
            .canonical_sha256();
        assert_ne!(a, b);
    }

    /// Pinned cross-language regression gate for the **shipped**
    /// `configs/minecraft/block_embeddings.toml`. Pair tests in
    /// `mc-bot/test/block_embeddings.test.ts` and
    /// `tests/python/training/test_muzero_mc_schema_id.py`.
    #[test]
    fn xlang_block_embeddings_pinned_to_known_good() {
        let raw =
            crate::test_support::read_workspace_config("configs/minecraft/block_embeddings.toml");
        let cfg = BlockEmbeddings::parse_toml(&raw).expect("parse shipped block embeddings");
        assert_eq!(cfg.vocab_size(), DEFAULT_NUM_BLOCK_EMBEDDINGS);
        let h = cfg.canonical_sha256();
        assert_eq!(
            h, "b5aef9f434474c17ffbdee7fe894ae93ada0f4e6477cb51b0a8cf4fc0d7a7a7e",
            "block-embeddings obs-layout pin drift — update Rust/JS/Python pins together.",
        );
    }

    #[test]
    fn ships_default_block_embeddings_parses() {
        let path =
            crate::test_support::workspace_config_path("configs/minecraft/block_embeddings.toml");
        let cfg = BlockEmbeddings::load(&path).expect("load shipped block embeddings");
        cfg.validate().unwrap();
        assert_eq!(cfg.vocab_size(), DEFAULT_NUM_BLOCK_EMBEDDINGS);
        assert_eq!(cfg.blocks.get("unknown").copied(), Some(35));
        assert_eq!(cfg.blocks.get("air").copied(), Some(0));
    }
}
