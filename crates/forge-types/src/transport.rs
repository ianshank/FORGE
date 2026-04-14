//! Transport and storage traits for cloud-edge data flow.
//!
//! These traits abstract the communication and storage backends used
//! by the cloud training pipeline and edge deployment runtime. Concrete
//! implementations live in downstream crates (e.g., `forge-cloud`).
//!
//! # Design
//!
//! All traits are `Send + Sync` to support async and multi-threaded contexts.
//! Methods return [`ForgeResult`] for uniform error handling across backends.

use crate::error::ForgeResult;

/// Trait for replay data transport between workers and coordinators.
///
/// Implementations handle sending compressed `CompactReplay` bytes from
/// rollout workers to the training coordinator, and receiving them on the
/// coordinator side.
///
/// # Backend Examples
///
/// - `InMemoryTransport` for unit testing
/// - `LocalTransport` for single-machine pipelines (filesystem)
/// - `GcsTransport` for Google Cloud Storage (behind feature flag)
/// - `PubSubTransport` for Google Cloud Pub/Sub (behind feature flag)
pub trait ReplayTransport: Send + Sync {
    /// Sends a serialized replay payload to the transport backend.
    ///
    /// The `key` is a unique identifier for the replay (e.g., seed-based).
    fn send(&self, key: &str, payload: &[u8]) -> ForgeResult<()>;

    /// Receives the next available replay payload, if any.
    ///
    /// Returns `Ok(None)` when no replay is currently available.
    fn receive(&self) -> ForgeResult<Option<(String, Vec<u8>)>>;

    /// Returns the transport backend name for logging and diagnostics.
    fn backend_name(&self) -> &str;
}

/// Trait for model artifact storage and retrieval.
///
/// Implementations handle storing trained model artifacts (e.g., ONNX files)
/// with version tracking, and retrieving them for edge deployment.
///
/// # Backend Examples
///
/// - `LocalModelStore` for filesystem-based storage
/// - `GcsModelStore` for Google Cloud Storage (behind feature flag)
pub trait ModelStore: Send + Sync {
    /// Stores a model artifact with the given name and version.
    fn store_model(&self, name: &str, version: &str, data: &[u8]) -> ForgeResult<()>;

    /// Loads a model artifact by name and version.
    fn load_model(&self, name: &str, version: &str) -> ForgeResult<Vec<u8>>;

    /// Returns the latest version string for a model, if any versions exist.
    fn latest_version(&self, name: &str) -> ForgeResult<Option<String>>;

    /// Lists all available versions for a model, ordered by version string.
    fn list_versions(&self, name: &str) -> ForgeResult<Vec<String>>;

    /// Returns the storage backend name for logging and diagnostics.
    fn backend_name(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::Mutex;

    /// A minimal in-memory transport for testing trait bounds.
    struct MockTransport {
        queue: Mutex<VecDeque<(String, Vec<u8>)>>,
    }

    impl MockTransport {
        fn new() -> Self {
            Self {
                queue: Mutex::new(VecDeque::new()),
            }
        }
    }

    impl ReplayTransport for MockTransport {
        fn send(&self, key: &str, payload: &[u8]) -> ForgeResult<()> {
            self.queue
                .lock()
                .unwrap()
                .push_back((key.to_string(), payload.to_vec()));
            Ok(())
        }

        fn receive(&self) -> ForgeResult<Option<(String, Vec<u8>)>> {
            Ok(self.queue.lock().unwrap().pop_front())
        }

        fn backend_name(&self) -> &str {
            "mock"
        }
    }

    /// A minimal in-memory model store for testing trait bounds.
    struct MockModelStore {
        models: Mutex<Vec<(String, String, Vec<u8>)>>,
    }

    impl MockModelStore {
        fn new() -> Self {
            Self {
                models: Mutex::new(Vec::new()),
            }
        }
    }

    impl ModelStore for MockModelStore {
        fn store_model(&self, name: &str, version: &str, data: &[u8]) -> ForgeResult<()> {
            self.models.lock().unwrap().push((
                name.to_string(),
                version.to_string(),
                data.to_vec(),
            ));
            Ok(())
        }

        fn load_model(&self, name: &str, version: &str) -> ForgeResult<Vec<u8>> {
            let models = self.models.lock().unwrap();
            models
                .iter()
                .find(|(n, v, _)| n == name && v == version)
                .map(|(_, _, d)| d.clone())
                .ok_or_else(|| {
                    crate::error::ForgeError::Cloud(crate::error::CloudError::ModelRegistry(
                        format!("model {name}@{version} not found"),
                    ))
                })
        }

        fn latest_version(&self, name: &str) -> ForgeResult<Option<String>> {
            let models = self.models.lock().unwrap();
            Ok(models
                .iter()
                .filter(|(n, _, _)| n == name)
                .next_back()
                .map(|(_, v, _)| v.clone()))
        }

        fn list_versions(&self, name: &str) -> ForgeResult<Vec<String>> {
            let models = self.models.lock().unwrap();
            Ok(models
                .iter()
                .filter(|(n, _, _)| n == name)
                .map(|(_, v, _)| v.clone())
                .collect())
        }

        fn backend_name(&self) -> &str {
            "mock"
        }
    }

    #[test]
    fn test_replay_transport_send_receive_roundtrip() {
        let transport = MockTransport::new();
        transport.send("seed_42", b"replay_data").unwrap();
        let received = transport.receive().unwrap();
        assert!(received.is_some());
        let (key, data) = received.unwrap();
        assert_eq!(key, "seed_42");
        assert_eq!(data, b"replay_data");
    }

    #[test]
    fn test_replay_transport_receive_empty() {
        let transport = MockTransport::new();
        let received = transport.receive().unwrap();
        assert!(received.is_none());
    }

    #[test]
    fn test_replay_transport_fifo_ordering() {
        let transport = MockTransport::new();
        transport.send("first", b"1").unwrap();
        transport.send("second", b"2").unwrap();
        let (key1, _) = transport.receive().unwrap().unwrap();
        let (key2, _) = transport.receive().unwrap().unwrap();
        assert_eq!(key1, "first");
        assert_eq!(key2, "second");
    }

    #[test]
    fn test_replay_transport_backend_name() {
        let transport = MockTransport::new();
        assert_eq!(transport.backend_name(), "mock");
    }

    #[test]
    fn test_model_store_roundtrip() {
        let store = MockModelStore::new();
        store.store_model("muzero", "v1", b"weights").unwrap();
        let data = store.load_model("muzero", "v1").unwrap();
        assert_eq!(data, b"weights");
    }

    #[test]
    fn test_model_store_not_found() {
        let store = MockModelStore::new();
        let result = store.load_model("missing", "v1");
        assert!(result.is_err());
    }

    #[test]
    fn test_model_store_latest_version() {
        let store = MockModelStore::new();
        assert!(store.latest_version("muzero").unwrap().is_none());

        store.store_model("muzero", "v1", b"w1").unwrap();
        store.store_model("muzero", "v2", b"w2").unwrap();
        assert_eq!(
            store.latest_version("muzero").unwrap(),
            Some("v2".to_string())
        );
    }

    #[test]
    fn test_model_store_list_versions() {
        let store = MockModelStore::new();
        store.store_model("muzero", "v1", b"w1").unwrap();
        store.store_model("muzero", "v2", b"w2").unwrap();
        store.store_model("other", "v1", b"w3").unwrap();
        let versions = store.list_versions("muzero").unwrap();
        assert_eq!(versions, vec!["v1", "v2"]);
    }

    #[test]
    fn test_model_store_backend_name() {
        let store = MockModelStore::new();
        assert_eq!(store.backend_name(), "mock");
    }

    #[test]
    fn test_transport_trait_is_object_safe() {
        // Verify ReplayTransport can be used as a trait object
        let transport: Box<dyn ReplayTransport> = Box::new(MockTransport::new());
        assert_eq!(transport.backend_name(), "mock");
    }

    #[test]
    fn test_model_store_trait_is_object_safe() {
        // Verify ModelStore can be used as a trait object
        let store: Box<dyn ModelStore> = Box::new(MockModelStore::new());
        assert_eq!(store.backend_name(), "mock");
    }
}
