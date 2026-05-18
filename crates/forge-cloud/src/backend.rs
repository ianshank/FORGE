//! Backend selection factory for runtime construction of storage backends.
//!
//! Reads the [`StorageConfig`](crate::config::StorageConfig) and constructs
//! the appropriate trait implementations. When
//! [`StorageConfig::backend`](crate::config::StorageConfig::backend) is
//! [`StorageBackend::Gcs`](crate::config::StorageBackend::Gcs), the `gcs`
//! feature flag must be enabled at compile time.

use crate::config::{StorageBackend, StorageConfig};
use crate::error::{CloudError, CloudResult, StorageError};
use crate::storage::{LocalModelStore, LocalReplayStore};
use crate::traits;

use tracing::info;

/// Creates a [`traits::ReplayStore`] from the given storage config.
///
/// # Errors
///
/// Returns an error if `Gcs` backend is requested without the `gcs` feature.
pub fn create_replay_store(config: &StorageConfig) -> CloudResult<Box<dyn traits::ReplayStore>> {
    match &config.backend {
        StorageBackend::Local => {
            info!(
                path = %config.replay_archive_path,
                "Using local replay store"
            );
            Ok(Box::new(LocalReplayStore::new(&config.replay_archive_path)))
        }
        StorageBackend::Gcs { bucket, prefix } => {
            #[cfg(feature = "gcs")]
            {
                info!(bucket, prefix, "Using GCS replay store");
                let store = crate::gcs_storage::GcsReplayStore::new(
                    bucket,
                    prefix,
                    &config.gcp_project,
                    &config.gcp_service_account,
                )?;
                Ok(Box::new(store))
            }
            #[cfg(not(feature = "gcs"))]
            {
                let _ = (bucket, prefix);
                Err(CloudError::Storage(StorageError::FeatureNotEnabled {
                    backend: "gcs".to_string(),
                    reason: "GCS backend requested but `gcs` feature is not enabled. \
                             Rebuild with `--features gcs`."
                        .to_string(),
                }))
            }
        }
    }
}

/// Creates a [`traits::ModelStore`] from the given storage config.
///
/// # Errors
///
/// Returns an error if `Gcs` backend is requested without the `gcs` feature.
pub fn create_model_store(config: &StorageConfig) -> CloudResult<Box<dyn traits::ModelStore>> {
    match &config.backend {
        StorageBackend::Local => {
            info!(
                path = %config.model_registry_path,
                "Using local model store"
            );
            Ok(Box::new(LocalModelStore::new(&config.model_registry_path)))
        }
        StorageBackend::Gcs { bucket, prefix } => {
            #[cfg(feature = "gcs")]
            {
                info!(bucket, prefix, "Using GCS model store");
                let store = crate::gcs_storage::GcsModelStore::new(
                    bucket,
                    prefix,
                    &config.gcp_project,
                    &config.gcp_service_account,
                )?;
                Ok(Box::new(store))
            }
            #[cfg(not(feature = "gcs"))]
            {
                let _ = (bucket, prefix);
                Err(CloudError::Storage(StorageError::FeatureNotEnabled {
                    backend: "gcs".to_string(),
                    reason: "GCS backend requested but `gcs` feature is not enabled. \
                             Rebuild with `--features gcs`."
                        .to_string(),
                }))
            }
        }
    }
}

/// Creates a [`forge_types::transport::ReplayTransport`] from the given storage config.
///
/// # Errors
///
/// Returns an error if `Gcs` backend is requested without the `gcs` feature.
pub fn create_replay_transport(
    config: &StorageConfig,
) -> CloudResult<Box<dyn forge_types::transport::ReplayTransport>> {
    match &config.backend {
        StorageBackend::Local => {
            // No local transport implementation exists yet; return a clear error.
            Err(CloudError::Storage(StorageError::ReadFailed {
                path: "local".to_string(),
                reason: "Local replay transport is not yet implemented. \
                         Use the cloud GCS transport or in-process queues."
                    .to_string(),
            }))
        }
        StorageBackend::Gcs { bucket, prefix } => {
            #[cfg(feature = "gcs")]
            {
                info!(bucket, prefix, "Using GCS replay transport");
                let transport = crate::gcs_storage::GcsReplayTransport::new(
                    bucket,
                    prefix,
                    &config.gcp_project,
                    &config.gcp_service_account,
                )?;
                Ok(Box::new(transport))
            }
            #[cfg(not(feature = "gcs"))]
            {
                let _ = (bucket, prefix);
                Err(CloudError::Storage(StorageError::FeatureNotEnabled {
                    backend: "gcs".to_string(),
                    reason: "GCS backend requested but `gcs` feature is not enabled. \
                             Rebuild with `--features gcs`."
                        .to_string(),
                }))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::StorageConfig;

    #[test]
    fn test_create_local_replay_store() {
        let config = StorageConfig::default();
        let store = create_replay_store(&config);
        assert!(store.is_ok());
    }

    #[test]
    fn test_create_local_model_store() {
        let config = StorageConfig::default();
        let store = create_model_store(&config);
        assert!(store.is_ok());
    }

    #[test]
    #[allow(unused_variables)]
    fn test_gcs_without_feature_returns_error() {
        let config = StorageConfig {
            backend: StorageBackend::Gcs {
                bucket: "test-bucket".to_string(),
                prefix: "test/".to_string(),
            },
            ..Default::default()
        };

        // Without the `gcs` feature, this should return an error.
        #[cfg(not(feature = "gcs"))]
        {
            assert!(create_replay_store(&config).is_err());
            assert!(create_model_store(&config).is_err());
            assert!(create_replay_transport(&config).is_err());
        }
    }

    /// `Local` replay transport is not implemented yet; calling the
    /// factory must return a clear `StorageError::ReadFailed`. This
    /// is the only branch of `create_replay_transport` reachable
    /// without the `gcs` feature and live GCS credentials.
    ///
    /// Uses `match` instead of `expect_err` because `Box<dyn ReplayTransport>`
    /// (the Ok payload) is not `Debug`, which `expect_err` requires.
    #[test]
    fn test_create_local_replay_transport_returns_not_implemented_error() {
        let config = StorageConfig::default();
        match create_replay_transport(&config) {
            Ok(_) => panic!("local transport should not succeed"),
            Err(e) => {
                let msg = format!("{e}");
                assert!(
                    msg.to_lowercase().contains("local")
                        || msg.to_lowercase().contains("not yet implemented"),
                    "expected diagnostic about local transport, got: {msg}"
                );
            }
        }
    }

    /// Sanity-check that `Local` replay store and model store accept a
    /// custom (non-default) directory path. Exercises the `info!` log
    /// path with a non-default config and the boxed-trait return.
    #[test]
    fn test_create_local_stores_accept_custom_paths() {
        let config = StorageConfig {
            replay_archive_path: "custom/replays".to_string(),
            model_registry_path: "custom/models".to_string(),
            ..Default::default()
        };
        assert!(create_replay_store(&config).is_ok());
        assert!(create_model_store(&config).is_ok());
    }
}
