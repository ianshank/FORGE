use std::sync::Arc;

use object_store::memory::InMemory;

use super::*;
use crate::traits::{ModelStore, ReplayStore};
use forge_replay::compact::CompactReplay;
use forge_types::config::ForgeConfig;
use forge_types::transport::{
    ModelStore as TypesModelStore, ReplayTransport as TypesReplayTransport,
};

/// Builds a minimal `CompactReplay` for use in tests.
fn make_replay(seed: u64) -> CompactReplay {
    CompactReplay::builder(ForgeConfig::default(), seed).build()
}

/// Shared in-memory `ObjectStore` for isolated test scenarios.
fn mem_store() -> Arc<dyn ObjectStore> {
    Arc::new(InMemory::new())
}

// --- GcsReplayStore -------------------------------------------------------

#[test]
fn test_replay_store_roundtrip() {
    let store = GcsReplayStore::with_object_store(mem_store(), "forge/");
    let replay = make_replay(42);
    store.store(&replay, "ep-001").unwrap();
    let loaded = store.load("ep-001").unwrap();
    assert_eq!(replay.seed, loaded.seed);
}

#[test]
fn test_replay_store_exists_true_and_false() {
    let store = GcsReplayStore::with_object_store(mem_store(), "forge/");
    let replay = make_replay(1);
    assert!(!store.exists("ep-000").unwrap());
    store.store(&replay, "ep-000").unwrap();
    assert!(store.exists("ep-000").unwrap());
}

#[test]
fn test_replay_store_load_missing_returns_error() {
    let store = GcsReplayStore::with_object_store(mem_store(), "forge/");
    let result = store.load("no-such-key");
    assert!(result.is_err());
}

#[test]
fn test_replay_store_delete_removes_object() {
    let store = GcsReplayStore::with_object_store(mem_store(), "forge/");
    let replay = make_replay(7);
    store.store(&replay, "del-me").unwrap();
    assert!(store.exists("del-me").unwrap());
    store.delete("del-me").unwrap();
    assert!(!store.exists("del-me").unwrap());
}

#[test]
fn test_replay_store_list_empty() {
    let store = GcsReplayStore::with_object_store(mem_store(), "forge/");
    let keys = store.list("").unwrap();
    assert!(keys.is_empty());
}

#[test]
fn test_replay_store_list_with_prefix_filter() {
    let store = GcsReplayStore::with_object_store(mem_store(), "forge/");
    let replay = make_replay(0);
    store.store(&replay, "alpha-1").unwrap();
    store.store(&replay, "alpha-2").unwrap();
    store.store(&replay, "beta-1").unwrap();

    let alpha_keys = store.list("alpha").unwrap();
    assert_eq!(alpha_keys.len(), 2);
    assert!(alpha_keys.iter().all(|k| k.starts_with("alpha")));

    let all_keys = store.list("").unwrap();
    assert_eq!(all_keys.len(), 3);
}

#[test]
fn test_replay_store_list_keys_sorted() {
    let store = GcsReplayStore::with_object_store(mem_store(), "forge/");
    let replay = make_replay(0);
    store.store(&replay, "ep-003").unwrap();
    store.store(&replay, "ep-001").unwrap();
    store.store(&replay, "ep-002").unwrap();
    let keys = store.list("").unwrap();
    assert_eq!(keys, vec!["ep-001", "ep-002", "ep-003"]);
}

#[test]
fn test_replay_store_prefix_isolation() {
    // Two stores with different prefixes share the same bucket but
    // must not see each other's data.
    let bucket = mem_store();
    let store_a = GcsReplayStore::with_object_store(Arc::clone(&bucket), "ns-a/");
    let store_b = GcsReplayStore::with_object_store(bucket, "ns-b/");
    let replay = make_replay(99);

    store_a.store(&replay, "ep-x").unwrap();
    assert!(!store_b.exists("ep-x").unwrap());
    assert!(store_a.exists("ep-x").unwrap());
}

// --- GcsModelStore (u32 trait) --------------------------------------------

#[test]
fn test_model_store_u32_roundtrip() {
    let store = GcsModelStore::with_object_store(mem_store(), "forge/");
    let data = b"model-weights-v1".to_vec();
    ModelStore::store_model(&store, "policy", 1, &data).unwrap();
    let loaded = ModelStore::load_model(&store, "policy", 1).unwrap();
    assert_eq!(data, loaded);
}

#[test]
fn test_model_store_latest_version_none_when_empty() {
    let store = GcsModelStore::with_object_store(mem_store(), "forge/");
    let latest = ModelStore::latest_version(&store, "policy").unwrap();
    assert!(latest.is_none());
}

#[test]
fn test_model_store_latest_version_updates_on_store() {
    let store = GcsModelStore::with_object_store(mem_store(), "forge/");
    ModelStore::store_model(&store, "policy", 1, b"v1").unwrap();
    assert_eq!(
        ModelStore::latest_version(&store, "policy").unwrap(),
        Some(1)
    );
    ModelStore::store_model(&store, "policy", 3, b"v3").unwrap();
    assert_eq!(
        ModelStore::latest_version(&store, "policy").unwrap(),
        Some(3)
    );
}

#[test]
fn test_model_store_latest_version_does_not_regress() {
    let store = GcsModelStore::with_object_store(mem_store(), "forge/");
    ModelStore::store_model(&store, "policy", 5, b"v5").unwrap();
    ModelStore::store_model(&store, "policy", 2, b"v2").unwrap();
    assert_eq!(
        ModelStore::latest_version(&store, "policy").unwrap(),
        Some(5)
    );
}

#[test]
fn test_model_store_list_versions() {
    let store = GcsModelStore::with_object_store(mem_store(), "forge/");
    ModelStore::store_model(&store, "net", 1, b"v1").unwrap();
    ModelStore::store_model(&store, "net", 3, b"v3").unwrap();
    ModelStore::store_model(&store, "net", 2, b"v2").unwrap();
    let mut versions = ModelStore::list_versions(&store, "net").unwrap();
    versions.sort();
    assert_eq!(versions, vec![1, 2, 3]);
}

#[test]
fn test_model_store_load_missing_returns_error() {
    let store = GcsModelStore::with_object_store(mem_store(), "forge/");
    let result = ModelStore::load_model(&store, "nonexistent", 99);
    assert!(result.is_err());
}

// --- GcsModelStore (str trait) --------------------------------------------

#[test]
fn test_model_store_str_roundtrip() {
    let store = GcsModelStore::with_object_store(mem_store(), "forge/");
    let data = b"str-version-weights".to_vec();
    TypesModelStore::store_model(&store, "value", "20240101", &data).unwrap();
    let loaded = TypesModelStore::load_model(&store, "value", "20240101").unwrap();
    assert_eq!(data, loaded);
}

#[test]
fn test_model_store_str_latest_version_none_when_empty() {
    let store = GcsModelStore::with_object_store(mem_store(), "forge/");
    let latest = TypesModelStore::latest_version(&store, "value").unwrap();
    assert!(latest.is_none());
}

#[test]
fn test_model_store_str_latest_version_updates() {
    let store = GcsModelStore::with_object_store(mem_store(), "forge/");
    TypesModelStore::store_model(&store, "value", "20240101", b"v1").unwrap();
    TypesModelStore::store_model(&store, "value", "20240201", b"v2").unwrap();
    let latest = TypesModelStore::latest_version(&store, "value").unwrap();
    assert_eq!(latest.as_deref(), Some("20240201"));
}

#[test]
fn test_model_store_str_list_versions_excludes_latest_marker() {
    let store = GcsModelStore::with_object_store(mem_store(), "forge/");
    TypesModelStore::store_model(&store, "value", "v1", b"data").unwrap();
    TypesModelStore::store_model(&store, "value", "v2", b"data").unwrap();
    let versions = TypesModelStore::list_versions(&store, "value").unwrap();
    assert!(!versions
        .iter()
        .any(|v| v == crate::constants::LATEST_VERSION_FILENAME));
    assert_eq!(versions.len(), 2);
}

#[test]
fn test_model_store_backend_name() {
    let store = GcsModelStore::with_object_store(mem_store(), "forge/");
    assert_eq!(TypesModelStore::backend_name(&store), "gcs");
}

// --- GcsReplayTransport ---------------------------------------------------

#[test]
fn test_transport_receive_empty_returns_none() {
    let transport = GcsReplayTransport::with_object_store(mem_store(), "forge/");
    let result = TypesReplayTransport::receive(&transport).unwrap();
    assert!(result.is_none());
}

#[test]
fn test_transport_send_receive_roundtrip() {
    let transport = GcsReplayTransport::with_object_store(mem_store(), "forge/");
    let payload = b"serialized-replay-bytes".to_vec();
    TypesReplayTransport::send(&transport, "batch-001", &payload).unwrap();
    let received = TypesReplayTransport::receive(&transport).unwrap();
    let (key, data) = received.expect("should have received a message");
    assert_eq!(key, "batch-001");
    assert_eq!(data, payload);
}

#[test]
fn test_transport_receive_deletes_after_consume() {
    let transport = GcsReplayTransport::with_object_store(mem_store(), "forge/");
    let payload = b"once".to_vec();
    TypesReplayTransport::send(&transport, "single", &payload).unwrap();
    // first receive returns the message
    let first = TypesReplayTransport::receive(&transport).unwrap();
    assert!(first.is_some());
    // second receive finds nothing (at-most-once delivery)
    let second = TypesReplayTransport::receive(&transport).unwrap();
    assert!(second.is_none());
}

#[test]
fn test_transport_backend_name() {
    let transport = GcsReplayTransport::with_object_store(mem_store(), "forge/");
    assert_eq!(TypesReplayTransport::backend_name(&transport), "gcs");
}

// --- Path-traversal validation -------------------------------------------

#[test]
fn test_model_store_rejects_traversal_in_model_id() {
    let store = GcsModelStore::with_object_store(mem_store(), "forge/");
    assert!(ModelStore::store_model(&store, "../escape", 1, b"x").is_err());
    assert!(ModelStore::load_model(&store, "a/b", 1).is_err());
    assert!(ModelStore::latest_version(&store, "a\\b").is_err());
    assert!(ModelStore::list_versions(&store, "../up").is_err());
}

#[test]
fn test_model_store_str_rejects_traversal_in_name_or_version() {
    let store = GcsModelStore::with_object_store(mem_store(), "forge/");
    assert!(TypesModelStore::store_model(&store, "../x", "v1", b"d").is_err());
    assert!(TypesModelStore::store_model(&store, "ok", "../v", b"d").is_err());
    assert!(TypesModelStore::load_model(&store, "a/b", "v1").is_err());
    assert!(TypesModelStore::load_model(&store, "ok", "a/b").is_err());
    assert!(TypesModelStore::latest_version(&store, "../x").is_err());
    assert!(TypesModelStore::list_versions(&store, "../x").is_err());
}

#[test]
fn test_transport_send_rejects_traversal_key() {
    let transport = GcsReplayTransport::with_object_store(mem_store(), "forge/");
    assert!(TypesReplayTransport::send(&transport, "../escape", b"x").is_err());
    assert!(TypesReplayTransport::send(&transport, "a/b", b"x").is_err());
}

// --- receive() .bin filtering --------------------------------------------

#[test]
fn test_transport_receive_ignores_non_bin_objects() {
    let bucket = mem_store();
    let transport = GcsReplayTransport::with_object_store(Arc::clone(&bucket), "forge/");

    // Manually write a non-.bin file under the transport prefix.
    let non_bin = ObjectPath::from("forge/transport/stray.txt");
    fallback_runtime()
        .block_on(bucket.put(&non_bin, b"junk".to_vec().into()))
        .unwrap();

    // Also write a valid .bin file.
    TypesReplayTransport::send(&transport, "valid-key", b"payload").unwrap();

    // receive should return the .bin file, not the .txt file.
    let (key, data) = TypesReplayTransport::receive(&transport)
        .unwrap()
        .expect("should receive valid .bin object");
    assert_eq!(key, "valid-key");
    assert_eq!(data, b"payload");
}
