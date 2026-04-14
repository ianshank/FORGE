//! Local filesystem implementations of [`ReplayStore`] and [`ModelStore`].
//!
//! These stores persist data to the local filesystem and are suitable for
//! single-machine pipelines, development, and testing. For distributed or
//! cloud-backed storage, see the GCS variants (behind feature flags).

use std::fs;
use std::path::PathBuf;

use forge_replay::compact::CompactReplay;
use tracing::{debug, info, instrument};

use crate::error::{CloudResult, StorageError};
use crate::traits;

/// Sanitizes a storage key to prevent path traversal.
fn sanitize_key(key: &str) -> Result<&str, StorageError> {
    if key.contains("..") || key.contains('/') || key.contains('\\') || key.contains('\0') {
        return Err(StorageError::WriteFailed {
            path: key.to_string(),
            reason: "key contains invalid characters (path traversal attempt)".to_string(),
        });
    }
    if key.is_empty() {
        return Err(StorageError::WriteFailed {
            path: key.to_string(),
            reason: "key must not be empty".to_string(),
        });
    }
    Ok(key)
}

/// Local filesystem implementation of [`traits::ReplayStore`].
///
/// Stores compact replays as bincode-serialized files at
/// `{base_path}/{key}.bin`.
pub struct LocalReplayStore {
    /// Root directory for replay storage.
    base_path: PathBuf,
}

impl LocalReplayStore {
    /// Creates a new local replay store rooted at `base_path`.
    ///
    /// The directory is created lazily on the first `store` call, not at
    /// construction time.
    pub fn new(base_path: impl Into<PathBuf>) -> Self {
        let base_path = base_path.into();
        info!(path = %base_path.display(), "Creating local replay store");
        Self { base_path }
    }

    /// Returns the file path for a given key.
    ///
    /// Returns an error if the key contains path traversal characters.
    fn key_path(&self, key: &str) -> CloudResult<PathBuf> {
        sanitize_key(key)?;
        Ok(self.base_path.join(format!("{key}.bin")))
    }
}

impl traits::ReplayStore for LocalReplayStore {
    #[instrument(skip(self, replay), fields(key = %key))]
    fn store(&self, replay: &CompactReplay, key: &str) -> CloudResult<()> {
        let path = self.key_path(key)?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| StorageError::WriteFailed {
                path: parent.display().to_string(),
                reason: e.to_string(),
            })?;
        }
        let bytes = replay.to_bytes().map_err(|e| StorageError::WriteFailed {
            path: path.display().to_string(),
            reason: e,
        })?;
        fs::write(&path, &bytes).map_err(|e| StorageError::WriteFailed {
            path: path.display().to_string(),
            reason: e.to_string(),
        })?;
        debug!(path = %path.display(), size = bytes.len(), "Stored replay");
        Ok(())
    }

    #[instrument(skip(self), fields(key = %key))]
    fn load(&self, key: &str) -> CloudResult<CompactReplay> {
        let path = self.key_path(key)?;
        let bytes = fs::read(&path).map_err(|_| StorageError::NotFound {
            path: path.display().to_string(),
        })?;
        let replay =
            CompactReplay::from_bytes(&bytes).map_err(|e| StorageError::InvalidFormat {
                path: path.display().to_string(),
                reason: e,
            })?;
        debug!(path = %path.display(), "Loaded replay");
        Ok(replay)
    }

    #[instrument(skip(self), fields(prefix = %prefix))]
    fn list(&self, prefix: &str) -> CloudResult<Vec<String>> {
        if !self.base_path.exists() {
            return Ok(Vec::new());
        }
        let entries = fs::read_dir(&self.base_path).map_err(|e| StorageError::ReadFailed {
            path: self.base_path.display().to_string(),
            reason: e.to_string(),
        })?;
        let mut keys = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| StorageError::ReadFailed {
                path: self.base_path.display().to_string(),
                reason: e.to_string(),
            })?;
            let file_name = entry.file_name();
            let name = file_name.to_string_lossy();
            if let Some(key) = name.strip_suffix(".bin") {
                if key.starts_with(prefix) {
                    keys.push(key.to_string());
                }
            }
        }
        keys.sort();
        debug!(count = keys.len(), "Listed replay keys");
        Ok(keys)
    }

    #[instrument(skip(self), fields(key = %key))]
    fn delete(&self, key: &str) -> CloudResult<()> {
        let path = self.key_path(key)?;
        fs::remove_file(&path).map_err(|_| StorageError::NotFound {
            path: path.display().to_string(),
        })?;
        debug!(path = %path.display(), "Deleted replay");
        Ok(())
    }

    #[instrument(skip(self), fields(key = %key))]
    fn exists(&self, key: &str) -> CloudResult<bool> {
        let exists = self.key_path(key)?.exists();
        debug!(key = %key, exists, "Checked replay existence");
        Ok(exists)
    }
}

/// Local filesystem implementation of [`traits::ModelStore`] (u32 versions).
///
/// Also implements the [`forge_types::transport::ModelStore`] trait (string
/// versions). Model artifacts are stored at
/// `{base_path}/{name}/{version}/model.bin`, with a `latest.txt` file
/// tracking the most recent version.
pub struct LocalModelStore {
    /// Root directory for model storage.
    base_path: PathBuf,
}

impl LocalModelStore {
    /// Creates a new local model store rooted at `base_path`.
    pub fn new(base_path: impl Into<PathBuf>) -> Self {
        let base_path = base_path.into();
        info!(path = %base_path.display(), "Creating local model store");
        Self { base_path }
    }

    /// Returns the directory path for a model name.
    fn model_dir(&self, name: &str) -> PathBuf {
        self.base_path.join(name)
    }

    /// Returns the directory for a specific version of a model.
    fn version_dir(&self, name: &str, version: &str) -> PathBuf {
        self.model_dir(name).join(version)
    }

    /// Returns the path to the model binary within a version directory.
    fn model_file(&self, name: &str, version: &str) -> PathBuf {
        self.version_dir(name, version)
            .join(crate::constants::MODEL_BINARY_FILENAME)
    }

    /// Returns the path to the `latest.txt` tracking file.
    fn latest_file(&self, name: &str) -> PathBuf {
        self.model_dir(name)
            .join(crate::constants::LATEST_VERSION_FILENAME)
    }

    /// Validates model name and version to prevent path traversal.
    fn validate_model_params(name: &str, version: &str) -> Result<(), StorageError> {
        sanitize_key(name)?;
        sanitize_key(version)?;
        Ok(())
    }

    /// Validates model name to prevent path traversal.
    fn validate_model_name(name: &str) -> Result<(), StorageError> {
        sanitize_key(name)?;
        Ok(())
    }
}

impl traits::ModelStore for LocalModelStore {
    #[instrument(skip(self, data), fields(model_id = %model_id, version))]
    fn store_model(&self, model_id: &str, version: u32, data: &[u8]) -> CloudResult<()> {
        Self::validate_model_name(model_id)?;
        let version_str = version.to_string();
        let dir = self.version_dir(model_id, &version_str);
        fs::create_dir_all(&dir).map_err(|e| StorageError::WriteFailed {
            path: dir.display().to_string(),
            reason: e.to_string(),
        })?;
        let file = self.model_file(model_id, &version_str);
        fs::write(&file, data).map_err(|e| StorageError::WriteFailed {
            path: file.display().to_string(),
            reason: e.to_string(),
        })?;
        // Update latest.txt only if this version is newer than the current latest.
        let latest = self.latest_file(model_id);
        let should_update = match fs::read_to_string(&latest) {
            Ok(current) => current.trim().parse::<u32>().map_or(true, |v| version > v),
            Err(_) => true,
        };
        if should_update {
            fs::write(&latest, version_str).map_err(|e| StorageError::WriteFailed {
                path: latest.display().to_string(),
                reason: e.to_string(),
            })?;
        }
        debug!(
            model_id,
            version,
            size = data.len(),
            "Stored model (u32 version)"
        );
        Ok(())
    }

    #[instrument(skip(self), fields(model_id = %model_id, version))]
    fn load_model(&self, model_id: &str, version: u32) -> CloudResult<Vec<u8>> {
        Self::validate_model_name(model_id)?;
        let version_str = version.to_string();
        let file = self.model_file(model_id, &version_str);
        let data = fs::read(&file).map_err(|_| StorageError::NotFound {
            path: file.display().to_string(),
        })?;
        debug!(
            model_id,
            version,
            size = data.len(),
            "Loaded model (u32 version)"
        );
        Ok(data)
    }

    #[instrument(skip(self), fields(model_id = %model_id))]
    fn latest_version(&self, model_id: &str) -> CloudResult<Option<u32>> {
        Self::validate_model_name(model_id)?;
        let latest = self.latest_file(model_id);
        if !latest.exists() {
            return Ok(None);
        }
        let content = fs::read_to_string(&latest).map_err(|e| StorageError::ReadFailed {
            path: latest.display().to_string(),
            reason: e.to_string(),
        })?;
        let version: u32 = content
            .trim()
            .parse()
            .map_err(|e: std::num::ParseIntError| StorageError::InvalidFormat {
                path: latest.display().to_string(),
                reason: e.to_string(),
            })?;
        Ok(Some(version))
    }

    #[instrument(skip(self), fields(model_id = %model_id))]
    fn list_versions(&self, model_id: &str) -> CloudResult<Vec<u32>> {
        Self::validate_model_name(model_id)?;
        let dir = self.model_dir(model_id);
        if !dir.exists() {
            return Ok(Vec::new());
        }
        let entries = fs::read_dir(&dir).map_err(|e| StorageError::ReadFailed {
            path: dir.display().to_string(),
            reason: e.to_string(),
        })?;
        let mut versions = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| StorageError::ReadFailed {
                path: dir.display().to_string(),
                reason: e.to_string(),
            })?;
            if entry.path().is_dir() {
                if let Ok(v) = entry.file_name().to_string_lossy().parse::<u32>() {
                    versions.push(v);
                }
            }
        }
        versions.sort();
        debug!(
            model_id,
            count = versions.len(),
            "Listed model versions (u32)"
        );
        Ok(versions)
    }
}

impl forge_types::transport::ModelStore for LocalModelStore {
    #[instrument(skip(self, data), fields(name = %name, version = %version))]
    fn store_model(
        &self,
        name: &str,
        version: &str,
        data: &[u8],
    ) -> forge_types::error::ForgeResult<()> {
        Self::validate_model_params(name, version).map_err(|e| {
            forge_types::error::ForgeError::Cloud(forge_types::error::CloudError::Storage(
                e.to_string(),
            ))
        })?;
        let dir = self.version_dir(name, version);
        fs::create_dir_all(&dir).map_err(|e| {
            forge_types::error::ForgeError::Cloud(forge_types::error::CloudError::Storage(format!(
                "failed to create directory {}: {e}",
                dir.display()
            )))
        })?;
        let file = self.model_file(name, version);
        fs::write(&file, data).map_err(|e| {
            forge_types::error::ForgeError::Cloud(forge_types::error::CloudError::Storage(format!(
                "failed to write {}: {e}",
                file.display()
            )))
        })?;
        // Update latest.txt only if this version sorts after the current latest.
        let latest = self.latest_file(name);
        let should_update = match fs::read_to_string(&latest) {
            Ok(current) => version > current.trim(),
            Err(_) => true,
        };
        if should_update {
            fs::write(&latest, version).map_err(|e| {
                forge_types::error::ForgeError::Cloud(
                    forge_types::error::CloudError::Storage(format!(
                        "failed to write {}: {e}",
                        latest.display()
                    )),
                )
            })?;
        }
        debug!(
            name,
            version,
            size = data.len(),
            "Stored model (str version)"
        );
        Ok(())
    }

    #[instrument(skip(self), fields(name = %name, version = %version))]
    fn load_model(&self, name: &str, version: &str) -> forge_types::error::ForgeResult<Vec<u8>> {
        Self::validate_model_params(name, version).map_err(|e| {
            forge_types::error::ForgeError::Cloud(forge_types::error::CloudError::Storage(
                e.to_string(),
            ))
        })?;
        let file = self.model_file(name, version);
        let data = fs::read(&file).map_err(|_| {
            forge_types::error::ForgeError::Cloud(forge_types::error::CloudError::ModelRegistry(
                format!("model {name}@{version} not found at {}", file.display()),
            ))
        })?;
        debug!(
            name,
            version,
            size = data.len(),
            "Loaded model (str version)"
        );
        Ok(data)
    }

    #[instrument(skip(self), fields(name = %name))]
    fn latest_version(&self, name: &str) -> forge_types::error::ForgeResult<Option<String>> {
        Self::validate_model_name(name).map_err(|e| {
            forge_types::error::ForgeError::Cloud(forge_types::error::CloudError::Storage(
                e.to_string(),
            ))
        })?;
        let latest = self.latest_file(name);
        if !latest.exists() {
            return Ok(None);
        }
        let content = fs::read_to_string(&latest).map_err(|e| {
            forge_types::error::ForgeError::Cloud(forge_types::error::CloudError::Storage(format!(
                "failed to read {}: {e}",
                latest.display()
            )))
        })?;
        Ok(Some(content.trim().to_string()))
    }

    #[instrument(skip(self), fields(name = %name))]
    fn list_versions(&self, name: &str) -> forge_types::error::ForgeResult<Vec<String>> {
        Self::validate_model_name(name).map_err(|e| {
            forge_types::error::ForgeError::Cloud(forge_types::error::CloudError::Storage(
                e.to_string(),
            ))
        })?;
        let dir = self.model_dir(name);
        if !dir.exists() {
            return Ok(Vec::new());
        }
        let entries = fs::read_dir(&dir).map_err(|e| {
            forge_types::error::ForgeError::Cloud(forge_types::error::CloudError::Storage(format!(
                "failed to read directory {}: {e}",
                dir.display()
            )))
        })?;
        let mut versions = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| {
                forge_types::error::ForgeError::Cloud(forge_types::error::CloudError::Storage(
                    format!("failed to read entry in {}: {e}", dir.display()),
                ))
            })?;
            if entry.path().is_dir() {
                let name = entry.file_name().to_string_lossy().to_string();
                versions.push(name);
            }
        }
        versions.sort();
        debug!(name, count = versions.len(), "Listed model versions (str)");
        Ok(versions)
    }

    fn backend_name(&self) -> &str {
        "local"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{ModelStore as CloudModelStore, ReplayStore};
    use forge_types::config::ForgeConfig;
    use forge_types::transport::ModelStore as TransportModelStore;
    use tempfile::tempdir;

    fn test_config() -> ForgeConfig {
        let mut config = ForgeConfig::default();
        config.world.width = 16;
        config.world.height = 16;
        config.world.seed = 42;
        config.agents.num_agents = 1;
        config.agents.comm_vocab_size = 0;
        config.task.max_episode_length = 100;
        config
    }

    fn test_replay() -> CompactReplay {
        let config = test_config();
        let mut builder = CompactReplay::builder(config, 42);
        builder.record_tick(vec![0]);
        builder.record_tick(vec![1]);
        builder
            .agent_names(vec!["TestAgent".to_string()])
            .final_rewards(vec![1.0])
            .build()
    }

    // ---- LocalReplayStore tests ----

    #[test]
    fn test_replay_store_roundtrip() {
        let dir = tempdir().unwrap();
        let store = LocalReplayStore::new(dir.path());
        let replay = test_replay();

        store.store(&replay, "episode_001").unwrap();
        let loaded = store.load("episode_001").unwrap();

        assert_eq!(loaded.seed, replay.seed);
        assert_eq!(loaded.actions.len(), replay.actions.len());
        assert!(loaded.validate_config());
    }

    #[test]
    fn test_replay_load_nonexistent() {
        let dir = tempdir().unwrap();
        let store = LocalReplayStore::new(dir.path());

        let result = store.load("missing");
        assert!(result.is_err());
    }

    #[test]
    fn test_replay_list_with_prefix() {
        let dir = tempdir().unwrap();
        let store = LocalReplayStore::new(dir.path());
        let replay = test_replay();

        store.store(&replay, "train_001").unwrap();
        store.store(&replay, "train_002").unwrap();
        store.store(&replay, "eval_001").unwrap();

        let train_keys = store.list("train").unwrap();
        assert_eq!(train_keys.len(), 2);
        assert!(train_keys.contains(&"train_001".to_string()));
        assert!(train_keys.contains(&"train_002".to_string()));

        let eval_keys = store.list("eval").unwrap();
        assert_eq!(eval_keys.len(), 1);
        assert!(eval_keys.contains(&"eval_001".to_string()));

        let all_keys = store.list("").unwrap();
        assert_eq!(all_keys.len(), 3);
    }

    #[test]
    fn test_replay_delete() {
        let dir = tempdir().unwrap();
        let store = LocalReplayStore::new(dir.path());
        let replay = test_replay();

        store.store(&replay, "to_delete").unwrap();
        assert!(store.exists("to_delete").unwrap());

        store.delete("to_delete").unwrap();
        assert!(!store.exists("to_delete").unwrap());
    }

    #[test]
    fn test_replay_exists() {
        let dir = tempdir().unwrap();
        let store = LocalReplayStore::new(dir.path());
        let replay = test_replay();

        assert!(!store.exists("nope").unwrap());
        store.store(&replay, "yes").unwrap();
        assert!(store.exists("yes").unwrap());
    }

    #[test]
    fn test_replay_store_creates_directories() {
        let dir = tempdir().unwrap();
        let nested = dir.path().join("a").join("b").join("c");
        let store = LocalReplayStore::new(&nested);
        let replay = test_replay();

        store.store(&replay, "deep").unwrap();
        assert!(nested.join("deep.bin").exists());
    }

    // ---- LocalModelStore (u32 / cloud trait) tests ----

    #[test]
    fn test_model_store_u32_roundtrip() {
        let dir = tempdir().unwrap();
        let store = LocalModelStore::new(dir.path());
        let data = b"model_weights_v1";

        CloudModelStore::store_model(&store, "policy", 1, data).unwrap();
        let loaded = CloudModelStore::load_model(&store, "policy", 1).unwrap();
        assert_eq!(loaded, data);
    }

    #[test]
    fn test_model_store_u32_load_nonexistent() {
        let dir = tempdir().unwrap();
        let store = LocalModelStore::new(dir.path());

        let result = CloudModelStore::load_model(&store, "missing", 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_model_store_u32_latest_version_none() {
        let dir = tempdir().unwrap();
        let store = LocalModelStore::new(dir.path());

        let latest = CloudModelStore::latest_version(&store, "empty").unwrap();
        assert!(latest.is_none());
    }

    #[test]
    fn test_model_store_u32_latest_version() {
        let dir = tempdir().unwrap();
        let store = LocalModelStore::new(dir.path());

        CloudModelStore::store_model(&store, "policy", 1, b"v1").unwrap();
        CloudModelStore::store_model(&store, "policy", 2, b"v2").unwrap();
        CloudModelStore::store_model(&store, "policy", 3, b"v3").unwrap();

        let latest = CloudModelStore::latest_version(&store, "policy").unwrap();
        assert_eq!(latest, Some(3));
    }

    #[test]
    fn test_model_store_u32_list_versions() {
        let dir = tempdir().unwrap();
        let store = LocalModelStore::new(dir.path());

        CloudModelStore::store_model(&store, "muzero", 3, b"w3").unwrap();
        CloudModelStore::store_model(&store, "muzero", 1, b"w1").unwrap();
        CloudModelStore::store_model(&store, "muzero", 2, b"w2").unwrap();

        let versions = CloudModelStore::list_versions(&store, "muzero").unwrap();
        assert_eq!(versions, vec![1, 2, 3]);
    }

    // ---- LocalModelStore (str / transport trait) tests ----

    #[test]
    fn test_model_store_str_roundtrip() {
        let dir = tempdir().unwrap();
        let store = LocalModelStore::new(dir.path());
        let data = b"onnx_weights";

        TransportModelStore::store_model(&store, "muzero", "v1.0", data).unwrap();
        let loaded = TransportModelStore::load_model(&store, "muzero", "v1.0").unwrap();
        assert_eq!(loaded, data);
    }

    #[test]
    fn test_model_store_str_load_nonexistent() {
        let dir = tempdir().unwrap();
        let store = LocalModelStore::new(dir.path());

        let result = TransportModelStore::load_model(&store, "missing", "v1");
        assert!(result.is_err());
    }

    #[test]
    fn test_model_store_str_latest_version_none() {
        let dir = tempdir().unwrap();
        let store = LocalModelStore::new(dir.path());

        let latest = TransportModelStore::latest_version(&store, "empty").unwrap();
        assert!(latest.is_none());
    }

    #[test]
    fn test_model_store_str_latest_version() {
        let dir = tempdir().unwrap();
        let store = LocalModelStore::new(dir.path());

        TransportModelStore::store_model(&store, "policy", "v1", b"w1").unwrap();
        TransportModelStore::store_model(&store, "policy", "v2", b"w2").unwrap();

        let latest = TransportModelStore::latest_version(&store, "policy").unwrap();
        assert_eq!(latest, Some("v2".to_string()));
    }

    #[test]
    fn test_model_store_str_list_versions() {
        let dir = tempdir().unwrap();
        let store = LocalModelStore::new(dir.path());

        TransportModelStore::store_model(&store, "muzero", "v1", b"w1").unwrap();
        TransportModelStore::store_model(&store, "muzero", "v2", b"w2").unwrap();
        TransportModelStore::store_model(&store, "other", "v1", b"w3").unwrap();

        let versions = TransportModelStore::list_versions(&store, "muzero").unwrap();
        assert_eq!(versions, vec!["v1", "v2"]);
    }

    #[test]
    fn test_model_store_backend_name() {
        let dir = tempdir().unwrap();
        let store = LocalModelStore::new(dir.path());
        assert_eq!(TransportModelStore::backend_name(&store), "local");
    }

    // ---- Path traversal prevention tests (replay store) ----

    #[test]
    fn test_replay_path_traversal_rejected() {
        let dir = tempdir().unwrap();
        let store = LocalReplayStore::new(dir.path());
        let replay = test_replay();

        let result = store.store(&replay, "../etc/passwd");
        assert!(result.is_err());

        let result = store.load("../../secret");
        assert!(result.is_err());
    }

    #[test]
    fn test_replay_slash_in_key_rejected() {
        let dir = tempdir().unwrap();
        let store = LocalReplayStore::new(dir.path());
        let replay = test_replay();

        let result = store.store(&replay, "foo/bar");
        assert!(result.is_err());

        let result = store.load("a/b");
        assert!(result.is_err());
    }

    #[test]
    fn test_replay_backslash_in_key_rejected() {
        let dir = tempdir().unwrap();
        let store = LocalReplayStore::new(dir.path());
        let replay = test_replay();

        let result = store.store(&replay, "foo\\bar");
        assert!(result.is_err());
    }

    #[test]
    fn test_replay_empty_key_rejected() {
        let dir = tempdir().unwrap();
        let store = LocalReplayStore::new(dir.path());
        let replay = test_replay();

        let result = store.store(&replay, "");
        assert!(result.is_err());

        let result = store.load("");
        assert!(result.is_err());
    }

    #[test]
    fn test_replay_null_byte_in_key_rejected() {
        let dir = tempdir().unwrap();
        let store = LocalReplayStore::new(dir.path());
        let replay = test_replay();

        let result = store.store(&replay, "foo\0bar");
        assert!(result.is_err());
    }

    // ---- Path traversal prevention tests (model store, u32 trait) ----

    #[test]
    fn test_model_store_u32_path_traversal_rejected() {
        let dir = tempdir().unwrap();
        let store = LocalModelStore::new(dir.path());

        let result = CloudModelStore::store_model(&store, "../etc", 1, b"data");
        assert!(result.is_err());

        let result = CloudModelStore::load_model(&store, "../../secret", 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_model_store_u32_slash_in_name_rejected() {
        let dir = tempdir().unwrap();
        let store = LocalModelStore::new(dir.path());

        let result = CloudModelStore::store_model(&store, "foo/bar", 1, b"data");
        assert!(result.is_err());
    }

    #[test]
    fn test_model_store_u32_empty_name_rejected() {
        let dir = tempdir().unwrap();
        let store = LocalModelStore::new(dir.path());

        let result = CloudModelStore::store_model(&store, "", 1, b"data");
        assert!(result.is_err());
    }

    #[test]
    fn test_model_store_u32_null_byte_in_name_rejected() {
        let dir = tempdir().unwrap();
        let store = LocalModelStore::new(dir.path());

        let result = CloudModelStore::store_model(&store, "model\0bad", 1, b"data");
        assert!(result.is_err());
    }

    // ---- Path traversal prevention tests (model store, str trait) ----

    #[test]
    fn test_model_store_str_path_traversal_rejected() {
        let dir = tempdir().unwrap();
        let store = LocalModelStore::new(dir.path());

        let result = TransportModelStore::store_model(&store, "../etc", "v1", b"data");
        assert!(result.is_err());

        let result = TransportModelStore::store_model(&store, "policy", "../v1", b"data");
        assert!(result.is_err());
    }

    #[test]
    fn test_model_store_str_slash_in_version_rejected() {
        let dir = tempdir().unwrap();
        let store = LocalModelStore::new(dir.path());

        let result = TransportModelStore::store_model(&store, "policy", "v1/hack", b"data");
        assert!(result.is_err());
    }

    #[test]
    fn test_model_store_str_empty_name_rejected() {
        let dir = tempdir().unwrap();
        let store = LocalModelStore::new(dir.path());

        let result = TransportModelStore::store_model(&store, "", "v1", b"data");
        assert!(result.is_err());
    }

    #[test]
    fn test_model_store_str_empty_version_rejected() {
        let dir = tempdir().unwrap();
        let store = LocalModelStore::new(dir.path());

        let result = TransportModelStore::store_model(&store, "policy", "", b"data");
        assert!(result.is_err());
    }

    #[test]
    fn test_model_store_str_null_byte_rejected() {
        let dir = tempdir().unwrap();
        let store = LocalModelStore::new(dir.path());

        let result = TransportModelStore::store_model(&store, "model\0x", "v1", b"data");
        assert!(result.is_err());

        let result = TransportModelStore::store_model(&store, "model", "v1\0x", b"data");
        assert!(result.is_err());
    }
}
