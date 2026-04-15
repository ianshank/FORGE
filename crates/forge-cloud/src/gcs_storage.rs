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
use object_store::{ObjectMeta, ObjectStore};
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
                    .map_or(true, |o| meta.last_modified < o.last_modified)
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
}
