//! Reward configuration loaded from `configs/minecraft/rewards.toml`.
//!
//! `forge-env-mc` does NOT execute reward functions — that's the Node
//! bot's job. This module exists solely to (a) parse and validate the
//! file so a typo is caught at runner startup, and (b) compute a
//! canonical SHA256 that's folded into the global `schema_id`.
//!
//! Nested path keys ([`NESTED_REWARD_PATH_KEY_CONFIG`] /
//! [`NESTED_REWARD_PATH_KEY_CRAFTING`]) are resolved relative to the
//! rewards.toml parent and **replaced in the hash tree** with the nested
//! file's own canonical SHA. Rename-without-content-change does not bump
//! `schema_id`. A set path key whose file is missing fails closed
//! (the bot's hardcoded milestone fallback must not be hashed as if it
//! were the file).
//!
//! String-parsed fixtures (no load directory) hash path strings as-is
//! so the fixture xlang pin stays stable.
//!
//! The bot's JS-side loader (`mc-bot/src/reward_config.ts`) MUST
//! produce the same canonical hash from the same file — enforced by
//! the `xlang_rewards_schema_id_pinned_to_known_good` test below and
//! the shipped nested-fold pin.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::McEnvError;
use crate::hash_util::{canonical_json_sha256, sha256_hex, toml_to_canonical_json};

/// Nested milestone-rewards TOML path key under a `[[reward]]` table.
/// Twin of JS/Python `NESTED_REWARD_PATH_KEYS`.
pub const NESTED_REWARD_PATH_KEY_CONFIG: &str = "config_path";

/// Nested crafting-rewards TOML path key under a `[[reward]]` table.
/// Twin of JS/Python `NESTED_REWARD_PATH_KEYS`.
pub const NESTED_REWARD_PATH_KEY_CRAFTING: &str = "crafting_config_path";

/// Path-valued keys whose **file contents** (not the path string) fold
/// into [`RewardConfig::canonical_sha256`] when the config was loaded
/// from disk.
pub const NESTED_REWARD_PATH_KEYS: &[&str] = &[
    NESTED_REWARD_PATH_KEY_CONFIG,
    NESTED_REWARD_PATH_KEY_CRAFTING,
];

/// Parsed reward config. `entries` order matters for the canonical
/// hash — both sides MUST iterate in file order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RewardConfig {
    /// Map schema version. v1 = the format documented here.
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    /// Reward entries. Multiple top-level entries are summed by the bot.
    #[serde(rename = "reward", default)]
    pub entries: Vec<toml::Value>,
    /// Directory of the rewards.toml this was loaded from, if any.
    /// Nested path keys are resolved against this directory and folded
    /// into [`Self::canonical_sha256`]. `None` for string-parsed fixtures
    /// (path strings hashed as-is so the xlang fixture pin stays stable).
    #[serde(skip)]
    load_dir: Option<PathBuf>,
}

fn default_schema_version() -> u32 {
    1
}

/// True when `key` is a nested reward-file path that must be folded.
#[must_use]
pub fn is_nested_reward_path_key(key: &str) -> bool {
    NESTED_REWARD_PATH_KEYS.contains(&key)
}

impl RewardConfig {
    /// Load and validate from a TOML file.
    ///
    /// Nested path keys are resolved relative to `parent(path)` and
    /// hashed immediately so a missing nested file fails closed at
    /// load rather than at the next handshake.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, McEnvError> {
        let path = path.as_ref();
        let raw = std::fs::read_to_string(path)
            .map_err(|e| McEnvError::Config(format!("read {}: {e}", path.display())))?;
        let mut cfg = Self::parse_toml(&raw)?;
        let load_dir = path.parent().ok_or_else(|| {
            McEnvError::Config(format!(
                "rewards path {} has no parent directory",
                path.display()
            ))
        })?;
        cfg.load_dir = Some(load_dir.to_path_buf());
        // Fail closed: hashing (and therefore handshake) must not
        // succeed when a nested path key is set and the file is missing.
        let _ = cfg.canonical_sha256()?;
        Ok(cfg)
    }

    /// Parse from a TOML string. Validates that at least one `[[reward]]`
    /// entry is present and every entry has a string `kind`.
    ///
    /// Does **not** fold nested path keys — those stay as path strings
    /// in the hash, matching the fixture xlang pin.
    pub fn parse_toml(raw: &str) -> Result<Self, McEnvError> {
        let cfg: Self = toml::from_str(raw)
            .map_err(|e| McEnvError::Config(format!("parse rewards toml: {e}")))?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Validate that the config has at least one entry and every entry
    /// has a non-empty `kind` string.
    pub fn validate(&self) -> Result<(), McEnvError> {
        if self.entries.is_empty() {
            return Err(McEnvError::Config(
                "rewards config has no [[reward]] entries".into(),
            ));
        }
        for (i, e) in self.entries.iter().enumerate() {
            let kind = e
                .get("kind")
                .and_then(|v| v.as_str())
                .ok_or_else(|| McEnvError::Config(format!("reward[{i}] missing string `kind`")))?;
            if kind.is_empty() {
                return Err(McEnvError::Config(format!("reward[{i}].kind is empty")));
            }
        }
        Ok(())
    }

    /// SHA256 of the canonical-form serialisation.
    ///
    /// Canonical-form rules — JS and Rust MUST both follow:
    /// - Object keys sorted alphabetically at every depth (Rust's
    ///   `toml::Value::Table` is `BTreeMap`-backed, so this is automatic
    ///   on the Rust side; the JS side does an explicit
    ///   recursive sort).
    /// - Whole-number floats normalised to integer JSON (`100.0 → 100`)
    ///   because `JSON.stringify(100.0) === "100"` in V8. Without this,
    ///   TOML floats with no fractional part disagree across languages.
    /// - File-order preservation for top-level entries (the
    ///   `[[reward]]` array order is meaningful).
    /// - When loaded from disk, nested path keys are replaced with the
    ///   nested file's own canonical SHA (whole parsed TOML document).
    ///
    /// Regression-tested via `xlang_rewards_schema_id_pinned_to_known_good`
    /// (fixture, no nested fold) and
    /// `xlang_shipped_rewards_schema_id_folds_nested_files` (shipped
    /// configs). Pair tests in `mc-bot/test/reward_config.test.ts`.
    pub fn canonical_sha256(&self) -> Result<String, McEnvError> {
        let normalised: Vec<serde_json::Value> = self
            .entries
            .iter()
            .map(|e| fold_toml_to_canonical_json(e, self.load_dir.as_deref()))
            .collect::<Result<_, _>>()?;
        let canonical =
            serde_json::to_string(&normalised).expect("serde_json::Value always serialises");
        Ok(sha256_hex(canonical.as_bytes()))
    }
}

/// Resolve a nested reward path relative to `base_dir`.
///
/// Candidates, first existing file wins:
/// 1. `value` if it is an absolute path to a file
/// 2. `base_dir.join(filename)` — sibling lookup (local + docker
///    mounts of `configs/minecraft/` as a flat dir)
/// 3. `base_dir.join(value)`
/// 4. cwd-relative `value`
fn resolve_nested_reward_path(
    base_dir: &Path,
    key: &str,
    value: &str,
) -> Result<PathBuf, McEnvError> {
    if value.is_empty() {
        return Err(McEnvError::Config(format!(
            "nested reward path key `{key}` is empty"
        )));
    }
    let given = Path::new(value);
    let mut candidates: Vec<PathBuf> = Vec::new();
    if given.is_absolute() {
        candidates.push(given.to_path_buf());
    }
    if let Some(name) = given.file_name() {
        candidates.push(base_dir.join(name));
    }
    candidates.push(base_dir.join(given));
    if !given.is_absolute() {
        candidates.push(given.to_path_buf());
    }
    for candidate in &candidates {
        if candidate.is_file() {
            return Ok(candidate.clone());
        }
    }
    Err(McEnvError::Config(format!(
        "nested reward file not found for `{key}` = `{value}` \
         (searched sibling, base_dir-relative, and cwd-relative; \
         hashing fails closed rather than using hardcoded milestone defaults)"
    )))
}

fn nested_file_canonical_sha256(path: &Path) -> Result<String, McEnvError> {
    let raw = std::fs::read_to_string(path).map_err(|e| {
        McEnvError::Config(format!("read nested reward file {}: {e}", path.display()))
    })?;
    let value: toml::Value = toml::from_str(&raw).map_err(|e| {
        McEnvError::Config(format!("parse nested reward file {}: {e}", path.display()))
    })?;
    Ok(canonical_json_sha256(&value))
}

/// Convert a TOML value to canonical JSON, substituting nested path
/// keys with the nested file's canonical SHA when `base_dir` is set.
fn fold_toml_to_canonical_json(
    v: &toml::Value,
    base_dir: Option<&Path>,
) -> Result<serde_json::Value, McEnvError> {
    use serde_json::Value as J;
    match v {
        toml::Value::Table(t) => {
            let mut map = serde_json::Map::new();
            for (k, val) in t {
                if is_nested_reward_path_key(k) {
                    let path_str = val.as_str().ok_or_else(|| {
                        McEnvError::Config(format!("nested reward path key `{k}` must be a string"))
                    })?;
                    let folded = match base_dir {
                        None => path_str.to_string(),
                        Some(dir) => {
                            let nested = resolve_nested_reward_path(dir, k, path_str)?;
                            nested_file_canonical_sha256(&nested)?
                        }
                    };
                    map.insert(k.clone(), J::String(folded));
                } else {
                    map.insert(k.clone(), fold_toml_to_canonical_json(val, base_dir)?);
                }
            }
            Ok(J::Object(map))
        }
        toml::Value::Array(a) => {
            let items = a
                .iter()
                .map(|item| fold_toml_to_canonical_json(item, base_dir))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(J::Array(items))
        }
        other => Ok(toml_to_canonical_json(other)),
    }
}

/// Combined `schema_id` folding action map + rewards + (future)
/// observation layout. This is the hash both sides agree on at the
/// `Hello` handshake.
///
/// Order is fixed: `sha256(action_map_hash || ":" || rewards_hash)`.
/// Changing the formula bumps every existing replay's `schema_id`.
/// Nested reward **file contents** are already inside `rewards_hash`
/// when the rewards config was loaded from disk. Block embeddings are
/// a **separate** obs-layout pin — do not fold them into this
/// two-input formula.
pub fn combined_schema_id(action_map_hash: &str, rewards_hash: &str) -> String {
    let combined = format!("{action_map_hash}:{rewards_hash}");
    sha256_hex(combined.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn sample_toml() -> &'static str {
        r#"
schema_version = 1

[[reward]]
kind = "survival"
value = 0.01

[[reward]]
kind = "distance_to_goal"
clip = 100.0

[reward.target]
x = 0
y = 64
z = 0
"#
    }

    #[test]
    fn parses_sample_toml() {
        let cfg = RewardConfig::parse_toml(sample_toml()).unwrap();
        assert_eq!(cfg.entries.len(), 2);
    }

    #[test]
    fn validate_rejects_empty_entries() {
        let raw = "schema_version = 1\n";
        assert!(matches!(
            RewardConfig::parse_toml(raw),
            Err(McEnvError::Config(_))
        ));
    }

    #[test]
    fn validate_rejects_missing_kind() {
        let raw = r#"
schema_version = 1
[[reward]]
value = 1.0
"#;
        assert!(matches!(
            RewardConfig::parse_toml(raw),
            Err(McEnvError::Config(_))
        ));
    }

    #[test]
    fn canonical_sha256_is_stable() {
        let cfg = RewardConfig::parse_toml(sample_toml()).unwrap();
        let a = cfg.canonical_sha256().unwrap();
        let b = cfg.canonical_sha256().unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 64);
    }

    #[test]
    fn canonical_sha256_changes_with_value_changes() {
        let a = RewardConfig::parse_toml(sample_toml())
            .unwrap()
            .canonical_sha256()
            .unwrap();
        let modified = sample_toml().replace("clip = 100.0", "clip = 50.0");
        let b = RewardConfig::parse_toml(&modified)
            .unwrap()
            .canonical_sha256()
            .unwrap();
        assert_ne!(a, b, "different reward params must yield different hash");
    }

    #[test]
    fn combined_schema_id_is_deterministic_and_unique() {
        let c1 = combined_schema_id("aaa", "bbb");
        let c2 = combined_schema_id("aaa", "bbb");
        let c3 = combined_schema_id("aaa", "ccc");
        let c4 = combined_schema_id("bbb", "aaa"); // swap order
        assert_eq!(c1, c2);
        assert_ne!(c1, c3);
        assert_ne!(c1, c4, "argument order must matter to combined hash");
    }

    /// Pinned cross-language regression gate — JS side hash for the
    /// `sample_toml()` fixture above MUST equal this value. Pair test
    /// in `mc-bot/test/reward_config.test.ts`.
    #[test]
    fn xlang_rewards_schema_id_pinned_to_known_good() {
        let cfg = RewardConfig::parse_toml(sample_toml()).unwrap();
        let h = cfg.canonical_sha256().unwrap();
        assert_eq!(
            h, "451b10f995371924a374633e5c42deab35c137fbbc65bc8f551bf2bd7844b478",
            "rewards-config schema_id drift — JS test in mc-bot/test/reward_config.test.ts will also fail. \
             If you intentionally bumped the format, update BOTH pinned constants together.",
        );
    }

    #[test]
    fn load_reads_from_disk_and_validates() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("r.toml");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(sample_toml().as_bytes()).unwrap();
        drop(f);
        let cfg = RewardConfig::load(&path).expect("load");
        assert_eq!(cfg.entries.len(), 2);
    }

    #[test]
    fn load_missing_file_returns_config_error() {
        let err = RewardConfig::load("/nonexistent/path/rewards.toml").unwrap_err();
        assert!(matches!(err, McEnvError::Config(_)));
    }

    #[test]
    fn parse_toml_rejects_garbage() {
        let err = RewardConfig::parse_toml("not = valid !").unwrap_err();
        assert!(matches!(err, McEnvError::Config(_)));
    }

    /// The shipped default config MUST parse + validate; protects
    /// against typos that would break runner startup.
    #[test]
    fn ships_default_rewards_config_parses() {
        let raw = crate::test_support::read_workspace_config("configs/minecraft/rewards.toml");
        let cfg = RewardConfig::parse_toml(&raw).expect("parse default rewards");
        cfg.validate().unwrap();
        assert!(!cfg.entries.is_empty());
    }

    fn write_nested_fixture(dir: &Path, nested_name: &str, nested_body: &str) -> PathBuf {
        let nested = dir.join(nested_name);
        std::fs::write(&nested, nested_body).unwrap();
        let rewards = dir.join("rewards.toml");
        let body = format!(
            "schema_version = 1\n\n[[reward]]\nkind = \"milestone\"\nconfig_path = \"{nested_name}\"\n"
        );
        std::fs::write(&rewards, body).unwrap();
        rewards
    }

    #[test]
    fn nested_content_change_bumps_hash() {
        let dir = tempfile::tempdir().unwrap();
        let rewards = write_nested_fixture(
            dir.path(),
            "milestones.toml",
            "[milestones]\nfirst_wood = { reward = 10.0, once = true }\n",
        );
        let a = RewardConfig::load(&rewards)
            .unwrap()
            .canonical_sha256()
            .unwrap();
        std::fs::write(
            dir.path().join("milestones.toml"),
            "[milestones]\nfirst_wood = { reward = 11.0, once = true }\n",
        )
        .unwrap();
        let b = RewardConfig::load(&rewards)
            .unwrap()
            .canonical_sha256()
            .unwrap();
        assert_ne!(a, b, "nested file content must fold into schema_id");
    }

    #[test]
    fn nested_rename_without_content_change_does_not_bump_hash() {
        let dir = tempfile::tempdir().unwrap();
        let body = "[milestones]\nfirst_wood = { reward = 10.0, once = true }\n";
        let rewards_a = write_nested_fixture(dir.path(), "mil_a.toml", body);
        let hash_a = RewardConfig::load(&rewards_a)
            .unwrap()
            .canonical_sha256()
            .unwrap();
        std::fs::copy(dir.path().join("mil_a.toml"), dir.path().join("mil_b.toml")).unwrap();
        let rewards_b = dir.path().join("rewards_b.toml");
        std::fs::write(
            &rewards_b,
            "schema_version = 1\n\n[[reward]]\nkind = \"milestone\"\nconfig_path = \"mil_b.toml\"\n",
        )
        .unwrap();
        let hash_b = RewardConfig::load(&rewards_b)
            .unwrap()
            .canonical_sha256()
            .unwrap();
        assert_eq!(
            hash_a, hash_b,
            "path-string rename with identical nested content must not bump schema_id"
        );
    }

    #[test]
    fn missing_nested_file_fails_closed_on_load() {
        let dir = tempfile::tempdir().unwrap();
        let rewards = dir.path().join("rewards.toml");
        std::fs::write(
            &rewards,
            "schema_version = 1\n\n[[reward]]\nkind = \"milestone\"\nconfig_path = \"missing.toml\"\n",
        )
        .unwrap();
        let err = RewardConfig::load(&rewards).unwrap_err();
        match err {
            McEnvError::Config(msg) => {
                assert!(
                    msg.contains("nested reward file not found"),
                    "expected fail-closed nested-path error, got: {msg}"
                );
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn parse_toml_without_load_dir_hashes_path_strings() {
        // Fixtures that mention a path but are not loaded from disk
        // must keep hashing the path string (xlang fixture pin).
        let raw = r#"
schema_version = 1
[[reward]]
kind = "milestone"
config_path = "configs/minecraft/milestone_rewards.toml"
"#;
        let cfg = RewardConfig::parse_toml(raw).unwrap();
        let h = cfg.canonical_sha256().unwrap();
        assert_eq!(h.len(), 64);
        // Must succeed even though the nested file is not resolved.
    }

    /// Pinned cross-language regression gate over the **shipped**
    /// `rewards.toml` + nested milestone/crafting files. Pair tests in
    /// JS and Python. Bumping nested content without updating all three
    /// pins fails CI simultaneously.
    #[test]
    fn xlang_shipped_rewards_schema_id_folds_nested_files() {
        let path = crate::test_support::workspace_config_path("configs/minecraft/rewards.toml");
        let cfg = RewardConfig::load(&path).expect("load shipped rewards");
        let h = cfg.canonical_sha256().expect("hash shipped rewards");
        assert_eq!(
            h, "78f96c103767f3db7280175e92b8564937bcb5d505e4d75e8aab0c570d237f4b",
            "shipped rewards schema_id drift (nested files folded) — \
             update Rust/JS/Python pins together.",
        );
    }
}
