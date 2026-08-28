//! Shared hashing helper for [`crate::action_map`] and
//! [`crate::reward_config`]'s `canonical_sha256`-style methods — both feed
//! a SHA-256 digest through the same hex encoding on the way to a
//! cross-language `schema_id`.

/// Lowercase-hex-encodes `bytes` (e.g. a SHA-256 digest).
pub(crate) fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}
