//! Google Cloud Storage implementations of [`ReplayStore`], [`ModelStore`],
//! and [`ReplayTransport`].
//!
//! All types in this module are gated behind the `gcs` Cargo feature flag.
//! They use the Apache Arrow `object_store` crate for async GCS operations,
//! wrapped in a [`GcsExecutor`] that safely dispatches work whether or not
//! the caller is already inside a Tokio runtime (see [`gcs_runtime`]).
//!
//! Authentication uses Application Default Credentials (ADC) by default.
//! For local development and CI, set `STORAGE_EMULATOR_HOST` to point at
//! a `fake-gcs-server` instance — no real credentials needed.
//!
//! ## Concurrency notes
//!
//! * **`store_model` (latest.txt):** The read-modify-write cycle on
//!   `latest.txt` is not atomic. Concurrent callers may race; the last
//!   writer wins. True CAS semantics would require GCS preconditions,
//!   which `object_store` 0.11 does not expose.
//!
//! * **`receive` (transport):** Multiple concurrent consumers may
//!   download the same object before any of them deletes it, leading to
//!   duplicate processing. For strict exactly-once delivery, use a
//!   dedicated message queue (e.g. Cloud Pub/Sub) instead of
//!   object-storage polling.

use std::sync::{Arc, OnceLock};

use futures::StreamExt;
use object_store::gcp::GoogleCloudStorageBuilder;
use object_store::path::Path as ObjectPath;
use object_store::{ObjectMeta, ObjectStore, ObjectStoreExt};
use tracing::{debug, info, instrument, warn};

use crate::error::{CloudError, CloudResult, StorageError};
use crate::storage::sanitize_key;
use crate::traits;
use forge_replay::compact::CompactReplay;

/// A thin executor whose [`block_on`](Self::block_on) method is safe to call
/// regardless of whether the current thread is already inside a Tokio runtime.
///
/// All GCS trait methods obtain an instance via [`gcs_runtime()`] and call
/// `block_on` to drive `object_store` futures synchronously.
struct GcsExecutor;

/// Returns a [`GcsExecutor`] whose [`block_on`](GcsExecutor::block_on) method
/// handles nested-runtime detection automatically.
///
/// Call sites use `gcs_runtime().block_on(future)` uniformly — the executor
/// picks the correct strategy at run time.
fn gcs_runtime() -> GcsExecutor {
    GcsExecutor
}

impl GcsExecutor {
    /// Drives `future` to completion on the current thread.
    ///
    /// * If called from within an existing **multi-threaded** Tokio runtime
    ///   (e.g. an `axum` handler), uses [`tokio::task::block_in_place`] +
    ///   [`tokio::runtime::Handle::block_on`] to avoid the nested-runtime
    ///   panic.
    /// * Otherwise, creates (once) and reuses a lightweight current-thread
    ///   runtime via [`fallback_runtime`].
    fn block_on<F: std::future::Future>(&self, future: F) -> F::Output {
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => tokio::task::block_in_place(|| handle.block_on(future)),
            Err(_) => fallback_runtime().block_on(future),
        }
    }
}

/// Shared fallback runtime for when no Tokio runtime is active on the
/// current thread (e.g. standalone CLI usage).
fn fallback_runtime() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to create tokio runtime for GCS blocking ops")
    })
}

/// Builds a GCS [`ObjectStore`] client from the given parameters.
///
/// Uses [`GoogleCloudStorageBuilder::from_env`] as a starting point so that
/// standard Google Cloud environment variables (e.g. `GOOGLE_SERVICE_ACCOUNT`,
/// `STORAGE_EMULATOR_HOST`) are honoured automatically.  Explicit non-empty
/// values for `gcp_service_account` override the environment.
///
/// `gcp_project` is accepted for logging and forward-compatibility but is
/// **not** passed to the builder — `object_store` 0.11 does not expose a
/// project-ID setter; the project is inferred from bucket ownership and
/// service-account credentials.
fn build_gcs_store(
    bucket: &str,
    gcp_project: &str,
    gcp_service_account: &str,
) -> CloudResult<Arc<dyn ObjectStore>> {
    let mut builder = GoogleCloudStorageBuilder::from_env().with_bucket_name(bucket);
    // object_store 0.11 has no project-ID setter; log it for traceability.
    if !gcp_project.is_empty() {
        debug!(
            gcp_project,
            "GCS project configured (informational — project is determined by credentials/bucket)"
        );
    }
    if !gcp_service_account.is_empty() {
        builder = builder.with_service_account_key(gcp_service_account);
    }
    let store = builder.build().map_err(|e| {
        CloudError::Storage(StorageError::ReadFailed {
            path: format!("gs://{bucket}"),
            reason: format!("failed to build GCS client: {e}"),
        })
    })?;
    Ok(Arc::new(store))
}

/// Normalizes a prefix so it is either empty or ends with `/`.
fn normalize_prefix(prefix: &str) -> String {
    if prefix.is_empty() || prefix.ends_with('/') {
        prefix.to_string()
    } else {
        format!("{prefix}/")
    }
}

/// Maps an [`object_store::Error`] to the appropriate [`StorageError`] variant.
fn classify_get_error(e: object_store::Error, path: &str) -> CloudError {
    match e {
        object_store::Error::NotFound { .. } => CloudError::Storage(StorageError::NotFound {
            path: path.to_string(),
        }),
        _ => CloudError::Storage(StorageError::ReadFailed {
            path: path.to_string(),
            reason: e.to_string(),
        }),
    }
}

/// Maps an [`object_store::Error`] from a delete call to [`StorageError`].
fn classify_delete_error(e: object_store::Error, path: &str) -> CloudError {
    match e {
        object_store::Error::NotFound { .. } => CloudError::Storage(StorageError::NotFound {
            path: path.to_string(),
        }),
        _ => CloudError::Storage(StorageError::WriteFailed {
            path: path.to_string(),
            reason: e.to_string(),
        }),
    }
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
}

impl GcsReplayStore {
    /// Creates a new GCS replay store.
    ///
    /// # Arguments
    ///
    /// * `bucket` — GCS bucket name
    /// * `prefix` — key prefix (e.g. `"forge/"`); normalized to include trailing `/`
    /// * `gcp_project` — optional GCP project ID (empty string = use ADC)
    /// * `gcp_service_account` — optional service account key (empty string = use ADC)
    pub fn new(
        bucket: &str,
        prefix: &str,
        gcp_project: &str,
        gcp_service_account: &str,
    ) -> CloudResult<Self> {
        let store = build_gcs_store(bucket, gcp_project, gcp_service_account)?;
        let prefix = normalize_prefix(prefix);
        info!(bucket, %prefix, "Creating GCS replay store");
        Ok(Self { store, prefix })
    }

    /// Creates a store backed by the given [`ObjectStore`] — for testing.
    #[cfg(test)]
    fn with_object_store(store: Arc<dyn ObjectStore>, prefix: &str) -> Self {
        Self {
            store,
            prefix: normalize_prefix(prefix),
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
        sanitize_key(key)?;
        let path = self.object_path(key);
        let bytes = replay.to_bytes().map_err(|e| StorageError::WriteFailed {
            path: path.to_string(),
            reason: e,
        })?;
        gcs_runtime()
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
        sanitize_key(key)?;
        let path = self.object_path(key);
        let result = gcs_runtime()
            .block_on(self.store.get(&path))
            .map_err(|e| classify_get_error(e, path.as_ref()))?;
        let bytes = gcs_runtime().block_on(result.bytes()).map_err(|e| {
            CloudError::Storage(StorageError::ReadFailed {
                path: path.to_string(),
                reason: e.to_string(),
            })
        })?;
        let replay =
            CompactReplay::from_bytes(&bytes).map_err(|e| StorageError::InvalidFormat {
                path: path.to_string(),
                reason: e,
            })?;
        debug!(path = %path, "Loaded replay from GCS");
        Ok(replay)
    }

    #[instrument(skip(self), fields(prefix = %prefix))]
    fn list(&self, prefix: &str) -> CloudResult<Vec<String>> {
        let list_prefix = ObjectPath::from(format!("{}replays/", self.prefix));
        let objects: Vec<Result<ObjectMeta, _>> = gcs_runtime().block_on(async {
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
        sanitize_key(key)?;
        let path = self.object_path(key);
        gcs_runtime()
            .block_on(self.store.delete(&path))
            .map_err(|e| classify_delete_error(e, path.as_ref()))?;
        debug!(path = %path, "Deleted replay from GCS");
        Ok(())
    }

    #[instrument(skip(self), fields(key = %key))]
    fn exists(&self, key: &str) -> CloudResult<bool> {
        sanitize_key(key)?;
        let path = self.object_path(key);
        match gcs_runtime().block_on(self.store.head(&path)) {
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
/// `{prefix}/models/{model_id}/{version}/{MODEL_BINARY_FILENAME}` with a
/// `{LATEST_VERSION_FILENAME}` tracking file at
/// `{prefix}/models/{model_id}/{LATEST_VERSION_FILENAME}`.
///
/// ## Concurrency
///
/// The `latest.txt` pointer is updated via a non-atomic read-modify-write.
/// Concurrent writers may race; the last writer wins.
pub struct GcsModelStore {
    store: Arc<dyn ObjectStore>,
    prefix: String,
}

impl GcsModelStore {
    /// Creates a new GCS model store.
    ///
    /// `prefix` is normalized to include a trailing `/`.
    pub fn new(
        bucket: &str,
        prefix: &str,
        gcp_project: &str,
        gcp_service_account: &str,
    ) -> CloudResult<Self> {
        let store = build_gcs_store(bucket, gcp_project, gcp_service_account)?;
        let prefix = normalize_prefix(prefix);
        info!(bucket, %prefix, "Creating GCS model store");
        Ok(Self { store, prefix })
    }

    /// Creates a store backed by the given [`ObjectStore`] — for testing.
    #[cfg(test)]
    fn with_object_store(store: Arc<dyn ObjectStore>, prefix: &str) -> Self {
        Self {
            store,
            prefix: normalize_prefix(prefix),
        }
    }

    /// Object path for a model binary.
    fn model_path(&self, model_id: &str, version: &str) -> ObjectPath {
        ObjectPath::from(format!(
            "{}models/{}/{}/{}",
            self.prefix,
            model_id,
            version,
            crate::constants::MODEL_BINARY_FILENAME,
        ))
    }

    /// Object path for the latest-version tracking file.
    fn latest_path(&self, model_id: &str) -> ObjectPath {
        ObjectPath::from(format!(
            "{}models/{}/{}",
            self.prefix,
            model_id,
            crate::constants::LATEST_VERSION_FILENAME,
        ))
    }

    /// Reads the current latest-version content, if it exists.
    fn read_latest(&self, model_id: &str) -> CloudResult<Option<String>> {
        let path = self.latest_path(model_id);
        match gcs_runtime().block_on(self.store.get(&path)) {
            Ok(result) => {
                let bytes = gcs_runtime().block_on(result.bytes()).map_err(|e| {
                    CloudError::Storage(StorageError::ReadFailed {
                        path: path.to_string(),
                        reason: e.to_string(),
                    })
                })?;
                Ok(Some(String::from_utf8_lossy(&bytes).trim().to_string()))
            }
            Err(object_store::Error::NotFound { .. }) => Ok(None),
            Err(e) => Err(CloudError::Storage(StorageError::ReadFailed {
                path: path.to_string(),
                reason: e.to_string(),
            })),
        }
    }

    /// Writes the latest-version tracking file with the given version string.
    fn write_latest(&self, model_id: &str, version: &str) -> CloudResult<()> {
        let path = self.latest_path(model_id);
        gcs_runtime()
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
        sanitize_key(model_id)?;
        let version_str = version.to_string();
        let path = self.model_path(model_id, &version_str);
        gcs_runtime()
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
        sanitize_key(model_id)?;
        let version_str = version.to_string();
        let path = self.model_path(model_id, &version_str);
        let result = gcs_runtime()
            .block_on(self.store.get(&path))
            .map_err(|e| classify_get_error(e, path.as_ref()))?;
        let bytes = gcs_runtime().block_on(result.bytes()).map_err(|e| {
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
        sanitize_key(model_id)?;
        match self.read_latest(model_id)? {
            Some(s) => {
                let version: u32 = s.parse().map_err(|e: std::num::ParseIntError| {
                    CloudError::Storage(StorageError::InvalidFormat {
                        path: self.latest_path(model_id).to_string(),
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
        sanitize_key(model_id)?;
        let list_prefix = ObjectPath::from(format!("{}models/{}/", self.prefix, model_id));
        let objects: Vec<Result<ObjectMeta, _>> = gcs_runtime().block_on(async {
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
            // Use rposition to find the *last* occurrence of model_id in the
            // path segments, avoiding false matches if the prefix contains
            // the same string.
            let parts: Vec<_> = meta.location.as_ref().split('/').collect();
            if let Some(pos) = parts.iter().rposition(|&p| p == model_id) {
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
        sanitize_key(name).map_err(|e| {
            forge_types::error::ForgeError::Cloud(forge_types::error::CloudError::Storage(
                e.to_string(),
            ))
        })?;
        sanitize_key(version).map_err(|e| {
            forge_types::error::ForgeError::Cloud(forge_types::error::CloudError::Storage(
                e.to_string(),
            ))
        })?;
        let path = self.model_path(name, version);
        gcs_runtime()
            .block_on(self.store.put(&path, data.to_vec().into()))
            .map_err(|e| {
                forge_types::error::ForgeError::Cloud(forge_types::error::CloudError::Storage(
                    format!("GCS write failed for {}: {e}", path),
                ))
            })?;
        // Update latest.txt only if this version sorts after the current latest.
        // For purely numeric version strings, compare numerically to avoid
        // lexicographic issues (e.g. "10" < "9" lexicographically).
        let should_update = match self.read_latest(name) {
            Ok(Some(current)) => {
                let current = current.trim();
                match (version.parse::<u64>(), current.parse::<u64>()) {
                    (Ok(new_v), Ok(cur_v)) => new_v > cur_v,
                    _ => version > current,
                }
            }
            _ => true,
        };
        if should_update {
            self.write_latest(name, version).map_err(|e| {
                forge_types::error::ForgeError::Cloud(forge_types::error::CloudError::Storage(
                    format!("GCS latest.txt update failed for model {name} version {version}: {e}"),
                ))
            })?;
        }
        debug!(
            name,
            version,
            size = data.len(),
            "Stored model to GCS (str)"
        );
        Ok(())
    }

    #[instrument(skip(self), fields(name = %name, version = %version))]
    fn load_model(&self, name: &str, version: &str) -> forge_types::error::ForgeResult<Vec<u8>> {
        sanitize_key(name).map_err(|e| {
            forge_types::error::ForgeError::Cloud(forge_types::error::CloudError::Storage(
                e.to_string(),
            ))
        })?;
        sanitize_key(version).map_err(|e| {
            forge_types::error::ForgeError::Cloud(forge_types::error::CloudError::Storage(
                e.to_string(),
            ))
        })?;
        let path = self.model_path(name, version);
        let result = gcs_runtime()
            .block_on(self.store.get(&path))
            .map_err(|e| match e {
                object_store::Error::NotFound { .. } => forge_types::error::ForgeError::Cloud(
                    forge_types::error::CloudError::ModelRegistry(format!(
                        "model {name}@{version} not found at {path}"
                    )),
                ),
                _ => {
                    forge_types::error::ForgeError::Cloud(forge_types::error::CloudError::Storage(
                        format!("GCS read failed for {path}: {e}"),
                    ))
                }
            })?;
        let bytes = gcs_runtime().block_on(result.bytes()).map_err(|e| {
            forge_types::error::ForgeError::Cloud(forge_types::error::CloudError::Storage(format!(
                "GCS read failed for {}: {e}",
                path
            )))
        })?;
        debug!(
            name,
            version,
            size = bytes.len(),
            "Loaded model from GCS (str)"
        );
        Ok(bytes.to_vec())
    }

    #[instrument(skip(self), fields(name = %name))]
    fn latest_version(&self, name: &str) -> forge_types::error::ForgeResult<Option<String>> {
        sanitize_key(name).map_err(|e| {
            forge_types::error::ForgeError::Cloud(forge_types::error::CloudError::Storage(
                e.to_string(),
            ))
        })?;
        self.read_latest(name).map_err(|e| {
            forge_types::error::ForgeError::Cloud(forge_types::error::CloudError::Storage(
                e.to_string(),
            ))
        })
    }

    #[instrument(skip(self), fields(name = %name))]
    fn list_versions(&self, name: &str) -> forge_types::error::ForgeResult<Vec<String>> {
        sanitize_key(name).map_err(|e| {
            forge_types::error::ForgeError::Cloud(forge_types::error::CloudError::Storage(
                e.to_string(),
            ))
        })?;
        let list_prefix = ObjectPath::from(format!("{}models/{}/", self.prefix, name));
        let objects: Vec<Result<ObjectMeta, _>> = gcs_runtime().block_on(async {
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
                forge_types::error::ForgeError::Cloud(forge_types::error::CloudError::Storage(
                    format!("GCS list failed for {}: {e}", list_prefix),
                ))
            })?;
            let parts: Vec<_> = meta.location.as_ref().split('/').collect();
            if let Some(pos) = parts.iter().rposition(|&p| p == name) {
                if let Some(ver_str) = parts.get(pos + 1) {
                    let v = (*ver_str).to_string();
                    if !versions.contains(&v) && v != crate::constants::LATEST_VERSION_FILENAME {
                        versions.push(v);
                    }
                }
            }
        }
        versions.sort();
        debug!(
            name,
            count = versions.len(),
            "Listed model versions from GCS (str)"
        );
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
///
/// ## Concurrency
///
/// Multiple concurrent consumers can download the same "oldest" object
/// before any of them deletes it, leading to duplicate processing.  For
/// strict exactly-once semantics, use a message queue (e.g. Cloud Pub/Sub)
/// rather than GCS polling.
pub struct GcsReplayTransport {
    store: Arc<dyn ObjectStore>,
    prefix: String,
}

impl GcsReplayTransport {
    /// Creates a new GCS replay transport.
    ///
    /// `prefix` is normalized to include a trailing `/`.
    pub fn new(
        bucket: &str,
        prefix: &str,
        gcp_project: &str,
        gcp_service_account: &str,
    ) -> CloudResult<Self> {
        let store = build_gcs_store(bucket, gcp_project, gcp_service_account)?;
        let prefix = normalize_prefix(prefix);
        info!(bucket, %prefix, "Creating GCS replay transport");
        Ok(Self { store, prefix })
    }

    /// Creates a transport backed by the given [`ObjectStore`] — for testing.
    #[cfg(test)]
    fn with_object_store(store: Arc<dyn ObjectStore>, prefix: &str) -> Self {
        Self {
            store,
            prefix: normalize_prefix(prefix),
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
        sanitize_key(key).map_err(|e| {
            forge_types::error::ForgeError::Cloud(forge_types::error::CloudError::Storage(
                e.to_string(),
            ))
        })?;
        let path = self.transport_path(key);
        gcs_runtime()
            .block_on(self.store.put(&path, payload.to_vec().into()))
            .map_err(|e| {
                forge_types::error::ForgeError::Cloud(forge_types::error::CloudError::Storage(
                    format!("GCS transport send failed for {}: {e}", path),
                ))
            })?;
        debug!(path = %path, size = payload.len(), "Sent replay via GCS transport");
        Ok(())
    }

    #[instrument(skip(self))]
    fn receive(&self) -> forge_types::error::ForgeResult<Option<(String, Vec<u8>)>> {
        let list_prefix = ObjectPath::from(format!("{}transport/", self.prefix));

        // Find the oldest .bin object (by last_modified), failing fast on errors.
        // Non-.bin objects are skipped to avoid processing unexpected files.
        let oldest: Option<ObjectMeta> = gcs_runtime().block_on(async {
            let mut stream = self.store.list(Some(&list_prefix));
            let mut oldest: Option<ObjectMeta> = None;
            while let Some(result) = stream.next().await {
                let meta = result.map_err(|e| {
                    forge_types::error::ForgeError::Cloud(forge_types::error::CloudError::Storage(
                        format!("GCS transport list failed for {list_prefix}: {e}"),
                    ))
                })?;
                let filename = meta.location.filename().unwrap_or_default();
                if !filename.ends_with(".bin") {
                    continue;
                }
                if oldest
                    .as_ref()
                    .is_none_or(|o| meta.last_modified < o.last_modified)
                {
                    oldest = Some(meta);
                }
            }
            Ok::<_, forge_types::error::ForgeError>(oldest)
        })?;

        let meta = match oldest {
            Some(m) => m,
            None => return Ok(None),
        };

        // Download the object.
        let result = gcs_runtime()
            .block_on(self.store.get(&meta.location))
            .map_err(|e| {
                forge_types::error::ForgeError::Cloud(forge_types::error::CloudError::Storage(
                    format!("GCS transport receive failed for {}: {e}", meta.location),
                ))
            })?;
        let bytes = gcs_runtime().block_on(result.bytes()).map_err(|e| {
            forge_types::error::ForgeError::Cloud(forge_types::error::CloudError::Storage(format!(
                "GCS transport read failed for {}: {e}",
                meta.location
            )))
        })?;

        // Extract key from filename (guaranteed to end with .bin by the filter).
        let filename = meta.location.filename().unwrap_or_default();
        let key = filename
            .strip_suffix(".bin")
            .unwrap_or(filename)
            .to_string();
        if key.is_empty() {
            warn!(path = %meta.location, "Skipping transport object with empty key");
            return Ok(None);
        }

        // Delete after successful download (at-most-once delivery).
        // Propagate delete errors — if deletion fails, the message may be
        // delivered again, so we surface the failure to the caller.
        gcs_runtime()
            .block_on(self.store.delete(&meta.location))
            .map_err(|e| {
                warn!(path = %meta.location, error = %e, "GCS transport delete failed");
                forge_types::error::ForgeError::Cloud(forge_types::error::CloudError::Storage(
                    format!("GCS transport delete failed for {}: {e}", meta.location),
                ))
            })?;

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
#[path = "gcs_storage/tests.rs"]
mod tests;
