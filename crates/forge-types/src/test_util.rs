//! Shared test utilities for config struct validation.
//!
//! Provides macros for common config test patterns used across FORGE crates.
//! These reduce boilerplate in config modules that follow the standard
//! `Clone + Debug + Serialize + Deserialize + Default` pattern.

/// Asserts that a config type survives a JSON serialization roundtrip.
///
/// Serializes the `Default` instance to JSON, deserializes it, re-serializes,
/// and verifies the two JSON strings are identical.
///
/// # Example
///
/// ```ignore
/// use forge_types::assert_config_serde_roundtrip;
///
/// #[test]
/// fn test_roundtrip() {
///     assert_config_serde_roundtrip!(MyConfig);
/// }
/// ```
#[macro_export]
macro_rules! assert_config_serde_roundtrip {
    ($config_type:ty) => {{
        let config = <$config_type>::default();
        let json =
            serde_json::to_string_pretty(&config).expect("config serialization should not fail");
        let deser: $config_type =
            serde_json::from_str(&json).expect("config deserialization should not fail");
        let json2 =
            serde_json::to_string_pretty(&deser).expect("config re-serialization should not fail");
        assert_eq!(
            json,
            json2,
            "serialization roundtrip produced different JSON for {}",
            stringify!($config_type)
        );
    }};
}

/// Asserts that a config type's `Default` implementation produces valid values.
///
/// Verifies the default config can be constructed and serialized without panicking,
/// and that `Clone` produces an equal JSON representation.
///
/// # Example
///
/// ```ignore
/// use forge_types::assert_config_defaults_valid;
///
/// #[test]
/// fn test_defaults() {
///     assert_config_defaults_valid!(MyConfig);
/// }
/// ```
#[macro_export]
macro_rules! assert_config_defaults_valid {
    ($config_type:ty) => {{
        let config = <$config_type>::default();
        // Ensure Debug works
        let _debug = format!("{:?}", config);
        // Ensure Clone works and produces same serialization
        let cloned = config.clone();
        let json1 = serde_json::to_string(&config).expect("default config serialization failed");
        let json2 = serde_json::to_string(&cloned).expect("cloned config serialization failed");
        assert_eq!(
            json1,
            json2,
            "cloned {} should serialize identically",
            stringify!($config_type)
        );
    }};
}
