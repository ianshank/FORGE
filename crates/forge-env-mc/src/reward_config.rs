//! Reward configuration loaded from `configs/minecraft/rewards.toml`.
//!
//! `forge-env-mc` does NOT execute reward functions — that's the Node
//! bot's job. This module exists solely to (a) parse and validate the
//! file so a typo is caught at runner startup, and (b) compute a
//! canonical SHA256 that's folded into the global `schema_id`.
//!
//! The bot's JS-side loader (`mc-bot/src/reward_config.ts`) MUST
//! produce the same canonical hash from the same file — enforced by
//! the `xlang_rewards_schema_id_pinned_to_known_good` test below.

use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::McEnvError;

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
}

fn default_schema_version() -> u32 {
    1
}

impl RewardConfig {
    /// Load and validate from a TOML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, McEnvError> {
        let path = path.as_ref();
        let raw = std::fs::read_to_string(path)
            .map_err(|e| McEnvError::Config(format!("read {}: {e}", path.display())))?;
        Self::parse_toml(&raw)
    }

    /// Parse from a TOML string. Validates that at least one `[[reward]]`
    /// entry is present and every entry has a string `kind`.
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
    ///
    /// Regression-tested via `xlang_rewards_schema_id_pinned_to_known_good`
    /// and its JS twin in `mc-bot/test/reward_config.test.ts`.
    pub fn canonical_sha256(&self) -> String {
        let normalised: Vec<serde_json::Value> =
            self.entries.iter().map(toml_to_canonical_json).collect();
        // serde_json::Value → string is infallible.
        let canonical =
            serde_json::to_string(&normalised).expect("serde_json::Value always serialises");
        let mut hasher = Sha256::new();
        hasher.update(canonical.as_bytes());
        hex_encode(&hasher.finalize())
    }
}

/// Convert a `toml::Value` to a `serde_json::Value`, normalising:
/// - whole floats → integers (to match JS `JSON.stringify(100.0) == "100"`),
/// - datetimes → ISO strings (rejected for reward configs but kept for
///   forward compatibility),
/// - tables → JSON objects (sorted alphabetically — already true via
///   `toml::Table`'s `BTreeMap` backing).
fn toml_to_canonical_json(v: &toml::Value) -> serde_json::Value {
    use serde_json::Value as J;
    match v {
        toml::Value::String(s) => J::String(s.clone()),
        toml::Value::Integer(i) => J::Number((*i).into()),
        toml::Value::Float(f) => {
            // Normalise whole floats to integers to match JS behaviour.
            // NaN/Infinity are not representable in TOML so unwrap is safe.
            if f.is_finite() && f.floor() == *f && f.abs() < (i64::MAX as f64) {
                J::Number((*f as i64).into())
            } else {
                J::Number(
                    serde_json::Number::from_f64(*f)
                        .expect("toml floats are finite by construction"),
                )
            }
        }
        toml::Value::Boolean(b) => J::Bool(*b),
        toml::Value::Datetime(d) => J::String(d.to_string()),
        toml::Value::Array(a) => J::Array(a.iter().map(toml_to_canonical_json).collect()),
        toml::Value::Table(t) => {
            // toml::Table is BTreeMap-backed → already alphabetically sorted.
            let mut map = serde_json::Map::new();
            for (k, val) in t {
                map.insert(k.clone(), toml_to_canonical_json(val));
            }
            J::Object(map)
        }
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

/// Combined `schema_id` folding action map + rewards + (future)
/// observation layout. This is the hash both sides agree on at the
/// `Hello` handshake.
///
/// Order is fixed: `sha256(action_map_hash || ":" || rewards_hash)`.
/// Changing the formula bumps every existing replay's `schema_id`.
pub fn combined_schema_id(action_map_hash: &str, rewards_hash: &str) -> String {
    let combined = format!("{action_map_hash}:{rewards_hash}");
    let mut hasher = Sha256::new();
    hasher.update(combined.as_bytes());
    hex_encode(&hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let a = cfg.canonical_sha256();
        let b = cfg.canonical_sha256();
        assert_eq!(a, b);
        assert_eq!(a.len(), 64);
    }

    #[test]
    fn canonical_sha256_changes_with_value_changes() {
        let a = RewardConfig::parse_toml(sample_toml())
            .unwrap()
            .canonical_sha256();
        let modified = sample_toml().replace("clip = 100.0", "clip = 50.0");
        let b = RewardConfig::parse_toml(&modified)
            .unwrap()
            .canonical_sha256();
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
        let h = cfg.canonical_sha256();
        assert_eq!(
            h, "451b10f995371924a374633e5c42deab35c137fbbc65bc8f551bf2bd7844b478",
            "rewards-config schema_id drift — JS test in mc-bot/test/reward_config.test.ts will also fail. \
             If you intentionally bumped the format, update BOTH pinned constants together.",
        );
    }

    #[test]
    fn load_reads_from_disk_and_validates() {
        use std::io::Write;
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
}
