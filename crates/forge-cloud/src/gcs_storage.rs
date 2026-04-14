//! Google Cloud Storage implementations of [`ReplayStore`], [`ModelStore`],
//! and [`ReplayTransport`].
//!
//! All types in this module are gated behind the `gcs` Cargo feature flag.
//! They use the Apache Arrow `object_store` crate for async GCS operations,
//! wrapped in a blocking Tokio runtime for compatibility with FORGE's
//! synchronous trait interfaces.
//!
//! Authentication uses Application Default Credentials (ADC) by default.
//! For local development and CI, set `STORAGE_EMULATOR_HOST` to point at
//! a `fake-gcs-server` instance — no real credentials needed.

use std::sync::Arc;

use futures::StreamExt;
use object_store::gcp::GoogleCloudStorageBuilder;
use object_store::path::Path as ObjectPath;
use object_store::{ObjectMeta, ObjectStore};
use tracing::{debug, info, instrument};

use crate::error::{CloudError, CloudResult, StorageError};
use crate::traits;
use forge_replay::compact::CompactReplay;

/// Creates a [`tokio::runtime::Runtime`] for blocking on async `object_store`
/// calls within FORGE's synchronous trait methods.
fn blocking_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("failed to create tokio runtime for GCS blocking ops")
}

/// Builds a GCS [`ObjectStore`] client from the given bucket name.
///
/// Respects `STORAGE_EMULATOR_HOST` for local dev / CI with `fake-gcs-server`.
fn build_gcs_store(bucket: &str) -> CloudResult<Arc<dyn ObjectStore>> {
    let store = GoogleCloudStorageBuilder::new()
        .with_bucket_name(bucket)
        .build()
        .map_err(|e| {
            CloudError::Storage(StorageError::ReadFailed {
                path: format!("gs://{bucket}"),
                reason: format!("failed to build GCS client: {e}"),
            })
        })?;
    Ok(Arc::new(store))
}

// ---------------------------------------------------------------------------
// GcsReplayStore
// ---------------------------------------------------------------------------

/// GCS-backed implementation of [`traits::ReplayStore`].
///
/// Stores compact replays as bincode-serialized objects at
/// `{prefix}/replays/{key}.bin` within the configured bucket.
pub struct GcsReplayStore {
    store: Arc<dyn ObjectStore>,
    prefix: String,
    rt: tokio::runtime::Runtime,
}

impl GcsReplayStore {
    /// Creates a new GCS replay store.
    ///
    /// # Arguments
    ///
    /// * `bucket` — GCS bucket name
    /// * `prefix` — key prefix (e.g. `"forge/"`)
    pub fn new(bucket: &str, prefix: &str) -> CloudResult<Self> {
        let store = build_gcs_store(bucket)?;
        let rt = blocking_runtime();
        info!(bucket, prefix, "Creating GCS replay store");
        Ok(Self {
            store,
            prefix: prefix.to_string(),
            rt,
        })
    }

    /// Creates a store backed by the given [`ObjectStore`] — for testing.
    #[cfg(test)]
    fn with_object_store(store: Arc<dyn ObjectStore>, prefix: &str) -> Self {
        Self {
            store,
            prefix: prefix.to_string(),
            rt: blocking_runtime(),
        }
    }

    /// Returns the full object path for a replay key.
    fn object_path(&self, key: &str) -> ObjectPath {
        ObjectPath::from(format!("{}replays/{}.bin", self.prefix, key))
    }
}

impl traits::ReplayStore for GcsReplayStore {
    #[instrument(skip(self, replay), fields(key = %key))]
    fn store(&self, replay: &CompactReplay, key: &str) -> CloudResult<()> {
        let path = self.object_path(key);
        let bytes = replay.to_bytes().map_err(|e| {
            StorageError::WriteFailed {
                path: path.to_string(),
                reason: e,
            }
        })?;
        self.rt
            .block_on(self.store.put(&path, bytes.into()))
            .map_err(|e| {
                CloudError::Storage(StorageError::WriteFailed {
                    path: path.to_string(),
                    reason: e.to_string(),
                })
            })?;
        debug!(path = %path, "Stored replay to GCS");
        Ok(())
    }

    #[instrument(skip(self), fields(key = %key))]
    fn load(&self, key: &str) -> CloudResult<CompactReplay> {
        let path = self.object_path(key);
        let result = self.rt.block_on(self.store.get(&path)).map_err(|_| {
            CloudError::Storage(StorageError::NotFound {
                path: path.to_string(),
            })
        })?;
        let bytes = self.rt.block_on(result.bytes()).map_err(|e| {
            CloudError::Storage(StorageError::ReadFailed {
                path: path.to_string(),
                reason: e.to_string(),
            })
        })?;
        let replay = CompactReplay::from_bytes(&bytes).map_err(|e| {
            StorageError::InvalidFormat {
                path: path.to_string(),
                reason: e,
            }
        })?;
        debug!(path = %path, "Loaded replay from GCS");
        Ok(replay)
    }

    #[instrument(skip(self), fields(prefix = %prefix))]
    fn list(&self, prefix: &str) -> CloudResult<Vec<String>> {
        let list_prefix = ObjectPath::from(format!("{}replays/", self.prefix));
        let objects: Vec<Result<ObjectMeta, _>> = self
            .rt
            .block_on(async {
                let mut items = Vec::new();
                let mut stream = self.store.list(Some(&list_prefix));
                while let Some(meta) = stream.next().await {
                    items.push(meta);
                }
                items
            });
        let mut keys = Vec::new();
        for item in objects {
            let meta = item.map_err(|e| {
                CloudError::Storage(StorageError::ReadFailed {
                    path: list_prefix.to_string(),
                    reason: e.to_string(),
                })
            })?;
            let name = meta.location.filename().unwrap_or_default();
            if let Some(key) = name.strip_suffix(".bin") {
                if key.starts_with(prefix) {
                    keys.push(key.to_string());
                }
            }
        }
        keys.sort();
        debug!(count = keys.len(), "Listed replay keys from GCS");
        Ok(keys)
    }

    #[instrument(skip(self), fields(key = %key))]
    fn delete(&self, key: &str) -> CloudResult<()> {
        let path = self.object_path(key);
        self.rt.block_on(self.store.delete(&path)).map_err(|_| {
            CloudError::Storage(StorageError::NotFound {
                path: path.to_string(),
            })
        })?;
        debug!(path = %path, "Deleted replay from GCS");
        Ok(())
    }

    #[instrument(skip(self), fields(key = %key))]
    fn exists(&self, key: &str) -> CloudResult<bool> {
        let path = self.object_path(key);
        match self.rt.block_on(self.store.head(&path)) {
            Ok(_) => Ok(true),
            Err(object_store::Error::NotFound { .. }) => Ok(false),
            Err(e) => Err(CloudError::Storage(StorageError::ReadFailed {
                path: path.to_string(),
                reason: e.to_string(),
            })),
        }
    }
}

// ---------------------------------------------------------------------------
// GcsModelStore
// ---------------------------------------------------------------------------

/// GCS-backed implementation of [`traits::ModelStore`] (u32 versions) and
/// [`forge_types::transport::ModelStore`] (string versions).
///
/// Model artifacts are stored at
/// `{prefix}/models/{model_id}/{version}/model.bin` with a `latest.txt`
/// tracking file at `{prefix}/models/{model_id}/latest.txt`.
pub struct GcsModelStore {
    store: Arc<dyn ObjectStore>,
    prefix: String,
    rt: tokio::runtime::Runtime,
}

impl GcsModelStore {
    /// Creates a new GCS model store.
    pub fn new(bucket: &str, prefix: &str) -> CloudResult<Self> {
        let store = build_gcs_store(bucket)?;
        let rt = blocking_runtime();
        info!(bucket, prefix, "Creating GCS model store");
        Ok(Self {
            store,
            prefix: prefix.to_string(),
            rt,
        })
    }

    /// Creates a store backed by the given [`ObjectStore`] — for testing.
    #[cfg(test)]
    fn with_object_store(store: Arc<dyn ObjectStore>, prefix: &str) -> Self {
        Self {
            store,
            prefix: prefix.to_string(),
            rt: blocking_runtime(),
        }
    }

    /// Object path for a model binary.
    fn model_path(&self, model_id: &str, version: &str) -> ObjectPath {
        ObjectPath::from(format!(
            "{}models/{}/{}/model.bin",
            self.prefix, model_id, version
        ))
    }

    /// Object path for the `latest.txt` tracking file.
    fn latest_path(&self, model_id: &str) -> ObjectPath {
        ObjectPath::from(format!("{}models/{}/latest.txt", self.prefix, model_id))
    }

    /// Reads the current `latest.txt` content, if it exists.
    fn read_latest(&self, model_id: &str) -> CloudResult<Option<String>> {
        let path = self.latest_path(model_id);
        match self.rt.block_on(self.store.get(&path)) {
            Ok(result) => {
                let bytes = self.rt.block_on(result.bytes()).map_err(|e| {
                    CloudError::Storage(StorageError::ReadFailed {
                        path: path.to_string(),
                        reason: e.to_string(),
                    })
                })?;
                Ok(Some(
                    String::from_utf8_lossy(&bytes).trim().to_string(),
                ))
            }
            Err(object_store::Error::NotFound { .. }) => Ok(None),
            Err(e) => Err(CloudError::Storage(StorageError::ReadFailed {
                path: path.to_string(),
                reason: e.to_string(),
            })),
        }
    }

    /// Writes `latest.txt` with the given version string.
    fn write_latest(&self, model_id: &str, version: &str) -> CloudResult<()> {
        let path = self.latest_path(model_id);
        self.rt
            .block_on(self.store.put(&path, version.as_bytes().to_vec().into()))
            .map_err(|e| {
                CloudError::Storage(StorageError::WriteFailed {
                    path: path.to_string(),
                    reason: e.to_string(),
                })
            })?;
        Ok(())
    }
}

impl traits::ModelStore for GcsModelStore {
    #[instrument(skip(self, data), fields(model_id = %model_id, version))]
    fn store_model(&self, model_id: &str, version: u32, data: &[u8]) -> CloudResult<()> {
        let version_str = version.to_string();
        let path = self.model_path(model_id, &version_str);
        self.rt
            .block_on(self.store.put(&path, data.to_vec().into()))
            .map_err(|e| {
                CloudError::Storage(StorageError::WriteFailed {
                    path: path.to_string(),
                    reason: e.to_string(),
                })
            })?;
        // Update latest.txt only if this version is newer.
        let should_update = match self.read_latest(model_id)? {
            Some(current) => current.trim().parse::<u32>().map_or(true, |v| version > v),
            None => true,
        };
        if should_update {
            self.write_latest(model_id, &version_str)?;
        }
        debug!(model_id, version, size = data.len(), "Stored model to GCS");
        Ok(())
    }

    #[instrument(skip(self), fields(model_id = %model_id, version))]
    fn load_model(&self, model_id: &str, version: u32) -> CloudResult<Vec<u8>> {
        let version_str = version.to_string();
        let path = self.model_path(model_id, &version_str);
        let result = self.rt.block_on(self.store.get(&path)).map_err(|_| {
            CloudError::Storage(StorageError::NotFound {
                path: path.to_string(),
            })
        })?;
        let bytes = self.rt.block_on(result.bytes()).map_err(|e| {
            CloudError::Storage(StorageError::ReadFailed {
                path: path.to_string(),
                reason: e.to_string(),
            })
        })?;
        debug!(
            model_id,
            version,
            size = bytes.len(),
            "Loaded model from GCS"
        );
        Ok(bytes.to_vec())
    }

    #[instrument(skip(self), fields(model_id = %model_id))]
    fn latest_version(&self, model_id: &str) -> CloudResult<Option<u32>> {
        match self.read_latest(model_id)? {
            Some(s) => {
                let version: u32 =
                    s.parse().map_err(|e: std::num::ParseIntError| {
                        CloudError::Storage(StorageError::InvalidFormat {
                            path: format!("{}models/{}/latest.txt", self.prefix, model_id),
                            reason: e.to_string(),
                        })
                    })?;
                Ok(Some(version))
            }
            None => Ok(None),
        }
    }

    #[instrument(skip(self), fields(model_id = %model_id))]
    fn list_versions(&self, model_id: &str) -> CloudResult<Vec<u32>> {
        let list_prefix =
            ObjectPath::from(format!("{}models/{}/", self.prefix, model_id));
        let objects: Vec<Result<ObjectMeta, _>> = self.rt.block_on(async {
            let mut items = Vec::new();
            let mut stream = self.store.list(Some(&list_prefix));
            while let Some(meta) = stream.next().await {
                items.push(meta);
            }
            items
        });
        let mut versions = Vec::new();
        for item in objects {
            let meta = item.map_err(|e| {
                CloudError::Storage(StorageError::ReadFailed {
                    path: list_prefix.to_string(),
                    reason: e.to_string(),
                })
            })?;
            // Path: {prefix}/models/{model_id}/{version}/model.bin
            // We want the version segment.
            let parts: Vec<_> = meta.location.as_ref().split('/').collect();
            // Find model_id in parts, take the next segment as version.
            if let Some(pos) = parts.iter().position(|&p| p == model_id) {
                if let Some(ver_str) = parts.get(pos + 1) {
                    if let Ok(v) = ver_str.parse::<u32>() {
                        if !versions.contains(&v) {
                            versions.push(v);
                        }
                    }
                }
            }
        }
        versions.sort();
        debug!(
            model_id,
            count = versions.len(),
            "Listed model versions from GCS"
        );
        Ok(versions)
    }
}

impl forge_types::transport::ModelStore for GcsModelStore {
    #[instrument(skip(self, data), fields(name = %name, version = %version))]
    fn store_model(
        &self,
        name: &str,
        version: &str,
        data: &[u8],
    ) -> forge_types::error::ForgeResult<()> {
        let path = self.model_path(name, version);
        self.rt
            .block_on(self.store.put(&path, data.to_vec().into()))
            .map_err(|e| {
                forge_types::error::ForgeError::Cloud(
                    forge_types::error::CloudError::Storage(format!(
                        "GCS write failed for {}: {e}",
                        path
                    )),
                )
            })?;
        // Update latest.txt only if this version sorts after the current latest.
        let should_update = match self.read_latest(name) {
            Ok(Some(current)) => version > current.trim(),
            _ => true,
        };
        if should_update {
            let _ = self.write_latest(name, version);
        }
        debug!(name, version, size = data.len(), "Stored model to GCS (str)");
        Ok(())
    }

    #[instrument(skip(self), fields(name = %name, version = %version))]
    fn load_model(
        &self,
        name: &str,
        version: &str,
    ) -> forge_types::error::ForgeResult<Vec<u8>> {
        let path = self.model_path(name, version);
        let result = self.rt.block_on(self.store.get(&path)).map_err(|_| {
            forge_types::error::ForgeError::Cloud(
                forge_types::error::CloudError::ModelRegistry(format!(
                    "model {name}@{version} not found at {path}"
                )),
            )
        })?;
        let bytes = self.rt.block_on(result.bytes()).map_err(|e| {
            forge_types::error::ForgeError::Cloud(
                forge_types::error::CloudError::Storage(format!(
                    "GCS read failed for {}: {e}",
                    path
                )),
            )
        })?;
        debug!(name, version, size = bytes.len(), "Loaded model from GCS (str)");
        Ok(bytes.to_vec())
    }

    #[instrument(skip(self), fields(name = %name))]
    fn latest_version(
        &self,
        name: &str,
    ) -> forge_types::error::ForgeResult<Option<String>> {
        self.read_latest(name).map_err(|e| {
            forge_types::error::ForgeError::Cloud(
                forge_types::error::CloudError::Storage(e.to_string()),
            )
        })
    }

    #[instrument(skip(self), fields(name = %name))]
    fn list_versions(
        &self,
        name: &str,
    ) -> forge_types::error::ForgeResult<Vec<String>> {
        let list_prefix =
            ObjectPath::from(format!("{}models/{}/", self.prefix, name));
        let objects: Vec<Result<ObjectMeta, _>> = self.rt.block_on(async {
            let mut items = Vec::new();
            let mut stream = self.store.list(Some(&list_prefix));
            while let Some(meta) = stream.next().await {
                items.push(meta);
            }
            items
        });
        let mut versions = Vec::new();
        for item in objects {
            let meta = item.map_err(|e| {
                forge_types::error::ForgeError::Cloud(
                    forge_types::error::CloudError::Storage(format!(
                        "GCS list failed for {}: {e}",
                        list_prefix
                    )),
                )
            })?;
            let parts: Vec<_> = meta.location.as_ref().split('/').collect();
            if let Some(pos) = parts.iter().position(|&p| p == name) {
                if let Some(ver_str) = parts.get(pos + 1) {
                    let v = (*ver_str).to_string();
                    if !versions.contains(&v)
                        && v != crate::constants::LATEST_VERSION_FILENAME
                    {
                        versions.push(v);
                    }
                }
            }
        }
        versions.sort();
        debug!(name, count = versions.len(), "Listed model versions from GCS (str)");
        Ok(versions)
    }

    fn backend_name(&self) -> &str {
        "gcs"
    }
}

// ---------------------------------------------------------------------------
// GcsReplayTransport
// ---------------------------------------------------------------------------

/// GCS-backed implementation of [`forge_types::transport::ReplayTransport`].
///
/// Uses a pull-based model: `send()` uploads to a transport prefix,
/// `receive()` lists the oldest object, downloads it, then deletes it.
pub struct GcsReplayTransport {
    store: Arc<dyn ObjectStore>,
    prefix: String,
    rt: tokio::runtime::Runtime,
}

impl GcsReplayTransport {
    /// Creates a new GCS replay transport.
    pub fn new(bucket: &str, prefix: &str) -> CloudResult<Self> {
        let store = build_gcs_store(bucket)?;
        let rt = blocking_runtime();
        info!(bucket, prefix, "Creating GCS replay transport");
        Ok(Self {
            store,
            prefix: prefix.to_string(),
            rt,
        })
    }

    /// Creates a transport backed by the given [`ObjectStore`] — for testing.
    #[cfg(test)]
    fn with_object_store(store: Arc<dyn ObjectStore>, prefix: &str) -> Self {
        Self {
            store,
            prefix: prefix.to_string(),
            rt: blocking_runtime(),
        }
    }

    /// Object path for a transport payload.
    fn transport_path(&self, key: &str) -> ObjectPath {
        ObjectPath::from(format!("{}transport/{}.bin", self.prefix, key))
    }
}

impl forge_types::transport::ReplayTransport for GcsReplayTransport {
    #[instrument(skip(self, payload), fields(key = %key))]
    fn send(&self, key: &str, payload: &[u8]) -> forge_types::error::ForgeResult<()> {
        let path = self.transport_path(key);
        self.rt
            .block_on(self.store.put(&path, payload.to_vec().into()))
            .map_err(|e| {
                forge_types::error::ForgeError::Cloud(
                    forge_types::error::CloudError::Storage(format!(
                        "GCS transport send failed for {}: {e}",
                        path
                    )),
                )
            })?;
        debug!(path = %path, size = payload.len(), "Sent replay via GCS transport");
        Ok(())
    }

    #[instrument(skip(self))]
    fn receive(&self) -> forge_types::error::ForgeResult<Option<(String, Vec<u8>)>> {
        let list_prefix = ObjectPath::from(format!("{}transport/", self.prefix));

        // Find the oldest object (by last_modified).
        let oldest = self.rt.block_on(async {
            let mut stream = self.store.list(Some(&list_prefix));
            let mut oldest: Option<ObjectMeta> = None;
            while let Some(result) = stream.next().await {
                if let Ok(meta) = result {
                    if oldest
                        .as_ref()
                        .map_or(true, |o| meta.last_modified < o.last_modified)
                    {
                        oldest = Some(meta);
                    }
                }
            }
            oldest
        });

        let meta = match oldest {
            Some(m) => m,
            None => return Ok(None),
        };

        // Download the object.
        let result =
            self.rt
                .block_on(self.store.get(&meta.location))
                .map_err(|e| {
                    forge_types::error::ForgeError::Cloud(
                        forge_types::error::CloudError::Storage(format!(
                            "GCS transport receive failed for {}: {e}",
                            meta.location
                        )),
                    )
                })?;
        let bytes = self.rt.block_on(result.bytes()).map_err(|e| {
            forge_types::error::ForgeError::Cloud(
                forge_types::error::CloudError::Storage(format!(
                    "GCS transport read failed for {}: {e}",
                    meta.location
                )),
            )
        })?;

        // Extract key from filename.
        let key = meta
            .location
            .filename()
            .unwrap_or_default()
            .strip_suffix(".bin")
            .unwrap_or_default()
            .to_string();

        // Delete after successful download (at-most-once delivery).
        let _ = self.rt.block_on(self.store.delete(&meta.location));

        debug!(key = %key, size = bytes.len(), "Received replay via GCS transport");
        Ok(Some((key, bytes.to_vec())))
    }

    fn backend_name(&self) -> &str {
        "gcs"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
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
        assert_eq!(ModelStore::latest_version(&store, "policy").unwrap(), Some(1));
        ModelStore::store_model(&store, "policy", 3, b"v3").unwrap();
        assert_eq!(ModelStore::latest_version(&store, "policy").unwrap(), Some(3));
    }

    #[test]
    fn test_model_store_latest_version_does_not_regress() {
        let store = GcsModelStore::with_object_store(mem_store(), "forge/");
        ModelStore::store_model(&store, "policy", 5, b"v5").unwrap();
        ModelStore::store_model(&store, "policy", 2, b"v2").unwrap();
        assert_eq!(ModelStore::latest_version(&store, "policy").unwrap(), Some(5));
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
}

