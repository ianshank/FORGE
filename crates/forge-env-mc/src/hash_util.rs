//! Shared hashing helper for [`crate::action_map`], [`crate::reward_config`],
//! and [`crate::block_embeddings`] canonical SHA-256 methods — all feed a
//! SHA-256 digest through the same hex encoding on the way to a
//! cross-language pin.

use sha2::{Digest, Sha256};

/// Lowercase-hex-encodes `bytes` (e.g. a SHA-256 digest).
pub(crate) fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

/// SHA-256 of `bytes`, lowercase hex.
pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex_encode(&hasher.finalize())
}

/// Convert a `toml::Value` to a `serde_json::Value`, normalising:
/// - whole floats → integers (to match JS `JSON.stringify(100.0) == "100"`),
/// - datetimes → ISO strings,
/// - tables → JSON objects (sorted alphabetically — already true via
///   `toml::Table`'s `BTreeMap` backing).
pub(crate) fn toml_to_canonical_json(v: &toml::Value) -> serde_json::Value {
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
            let mut map = serde_json::Map::new();
            for (k, val) in t {
                map.insert(k.clone(), toml_to_canonical_json(val));
            }
            J::Object(map)
        }
    }
}

/// SHA-256 of the canonical JSON form of a TOML value.
pub(crate) fn canonical_json_sha256(v: &toml::Value) -> String {
    let json = toml_to_canonical_json(v);
    // serde_json::Value → string is infallible.
    let canonical = serde_json::to_string(&json).expect("serde_json::Value always serialises");
    sha256_hex(canonical.as_bytes())
}
