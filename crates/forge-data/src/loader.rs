//! Core trait and container types for offline training datasets.
//!
//! All dataset sources implement [`DatasetLoader`] and return an
//! [`OfflineDataset`], which is a thin wrapper around a `Vec<Trajectory>`
//! from `forge-replay`. This makes all datasets directly compatible with the
//! existing `forge-replay` export pipeline (CSV / JSONL) and with Python-side
//! HuggingFace Datasets tooling.

use forge_replay::trajectory::Trajectory;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::instrument;

/// Errors that can occur while loading a dataset.
#[derive(Debug, Error)]
pub enum DatasetError {
    /// An I/O error reading a file.
    #[error("I/O error: {0}")]
    Io(String),
    /// JSON / serde deserialization failed.
    #[error("deserialization error: {0}")]
    Deserialize(String),
    /// The source data uses an unsupported format version.
    #[error("unsupported format version: {0}")]
    UnsupportedVersion(String),
    /// An action in the source data cannot be mapped to a FORGE action.
    #[error("unmappable action: {0}")]
    UnmappableAction(String),
}

impl From<std::io::Error> for DatasetError {
    fn from(e: std::io::Error) -> Self {
        DatasetError::Io(e.to_string())
    }
}

impl From<serde_json::Error> for DatasetError {
    fn from(e: serde_json::Error) -> Self {
        DatasetError::Deserialize(e.to_string())
    }
}

/// Trait implemented by all dataset loaders.
///
/// Each loader reads a specific external format and converts it to FORGE's
/// native [`Trajectory`] representation from `forge-replay`.
pub trait DatasetLoader {
    /// Loads all trajectories from the given source path.
    ///
    /// `path` semantics vary by implementation:
    /// - For file-based loaders it is a path to a `.jsonl` file or directory.
    /// - For the generator it may be ignored (pass `""` or `"."`) since data
    ///   is synthesized at runtime.
    fn load(&self, path: &str) -> Result<OfflineDataset, DatasetError>;

    /// Human-readable name identifying the dataset source.
    fn source_name(&self) -> &str;
}

/// A collection of trajectories from a single dataset source.
///
/// Wraps `Vec<Trajectory>` with metadata about where the data came from.
/// Can be merged with other `OfflineDataset`s via [`OfflineDataset::merge`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OfflineDataset {
    /// All trajectories in the dataset.
    pub trajectories: Vec<Trajectory>,
    /// Provenance metadata.
    pub metadata: DatasetMetadata,
}

/// Metadata describing a dataset.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DatasetMetadata {
    /// Name of the dataset source (e.g. "MineRL", "Minari-PointMaze").
    pub source: String,
    /// Total number of trajectories.
    pub num_trajectories: usize,
    /// Total number of steps across all trajectories.
    pub total_steps: u64,
    /// Optional URL pointing to the original dataset.
    pub source_url: Option<String>,
    /// License under which the original dataset is distributed.
    pub license: Option<String>,
}

impl OfflineDataset {
    /// Creates an empty dataset with the given source name.
    pub fn new(source: impl Into<String>) -> Self {
        Self {
            trajectories: Vec::new(),
            metadata: DatasetMetadata {
                source: source.into(),
                ..Default::default()
            },
        }
    }

    /// Returns the number of trajectories.
    pub fn len(&self) -> usize {
        self.trajectories.len()
    }

    /// Returns `true` if the dataset contains no trajectories.
    pub fn is_empty(&self) -> bool {
        self.trajectories.is_empty()
    }

    /// Returns the total number of steps across all trajectories.
    pub fn total_steps(&self) -> u64 {
        self.trajectories.iter().map(|t| t.len() as u64).sum()
    }

    /// Adds a trajectory and updates metadata step count.
    #[instrument(skip_all)]
    pub fn push(&mut self, traj: Trajectory) {
        self.metadata.total_steps += traj.len() as u64;
        self.metadata.num_trajectories += 1;
        self.trajectories.push(traj);
    }

    /// Merges `other` into `self`, appending all trajectories.
    ///
    /// The source name of `self` is preserved; the step counts are summed.
    #[instrument(skip_all)]
    pub fn merge(&mut self, other: OfflineDataset) {
        self.metadata.total_steps += other.metadata.total_steps;
        self.metadata.num_trajectories += other.metadata.num_trajectories;
        self.trajectories.extend(other.trajectories);
    }

    /// Serialises the dataset to a JSONL file, one trajectory step per line.
    ///
    /// Compatible with `forge_replay::export::export_to_jsonl` conventions and
    /// directly loadable by HuggingFace Datasets via `load_dataset("json")`.
    #[instrument(skip_all, fields(path = %path))]
    pub fn export_jsonl(&self, path: &str) -> Result<(), DatasetError> {
        use std::io::Write as _;
        let file = std::fs::File::create(path).map_err(DatasetError::from)?;
        let mut writer = std::io::BufWriter::new(file);
        for traj in &self.trajectories {
            for step in &traj.steps {
                let line = serde_json::to_string(step)?;
                writeln!(writer, "{line}").map_err(|e| DatasetError::Io(e.to_string()))?;
            }
        }
        Ok(())
    }

    /// Serialises the dataset metadata to a JSON file.
    #[instrument(skip_all, fields(path = %path))]
    pub fn export_metadata_json(&self, path: &str) -> Result<(), DatasetError> {
        let json = serde_json::to_string_pretty(&self.metadata)?;
        std::fs::write(path, json).map_err(DatasetError::from)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_replay::trajectory::Trajectory;

    #[test]
    fn test_empty_dataset() {
        let ds = OfflineDataset::new("test");
        assert!(ds.is_empty());
        assert_eq!(ds.len(), 0);
        assert_eq!(ds.total_steps(), 0);
    }

    #[test]
    fn test_push_updates_counts() {
        let mut ds = OfflineDataset::new("test");
        let traj = Trajectory::new();
        ds.push(traj);
        assert_eq!(ds.len(), 1);
        assert_eq!(ds.metadata.num_trajectories, 1);
    }

    #[test]
    fn test_merge() {
        let mut a = OfflineDataset::new("a");
        a.push(Trajectory::new());

        let mut b = OfflineDataset::new("b");
        b.push(Trajectory::new());
        b.push(Trajectory::new());

        a.merge(b);
        assert_eq!(a.len(), 3);
        assert_eq!(a.metadata.num_trajectories, 3);
    }

    #[test]
    fn test_export_jsonl_and_metadata() {
        let ds = OfflineDataset::new("unit-test");
        let dir = tempfile::tempdir().unwrap();
        let jsonl = dir.path().join("data.jsonl").to_str().unwrap().to_owned();
        let meta = dir.path().join("meta.json").to_str().unwrap().to_owned();
        assert!(ds.export_jsonl(&jsonl).is_ok());
        assert!(ds.export_metadata_json(&meta).is_ok());
        let content = std::fs::read_to_string(&meta).unwrap();
        assert!(content.contains("unit-test"));
    }

    #[test]
    fn test_dataset_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err = DatasetError::from(io_err);
        assert!(matches!(err, DatasetError::Io(_)));
        assert!(err.to_string().contains("I/O"));
    }

    #[test]
    fn test_dataset_error_from_serde() {
        let serde_err = serde_json::from_str::<serde_json::Value>("{bad json}").unwrap_err();
        let err = DatasetError::from(serde_err);
        assert!(matches!(err, DatasetError::Deserialize(_)));
    }

    #[test]
    fn test_dataset_error_display_variants() {
        assert!(DatasetError::Io("oops".to_string())
            .to_string()
            .contains("oops"));
        assert!(DatasetError::Deserialize("bad".to_string())
            .to_string()
            .contains("bad"));
        assert!(DatasetError::UnsupportedVersion("v99".to_string())
            .to_string()
            .contains("v99"));
        assert!(DatasetError::UnmappableAction("X".to_string())
            .to_string()
            .contains("X"));
    }

    #[test]
    fn test_total_steps_after_push() {
        let mut ds = OfflineDataset::new("test");
        // Push empty trajectory — step count stays 0.
        ds.push(Trajectory::new());
        assert_eq!(ds.total_steps(), 0);
    }

    #[test]
    fn test_is_empty_after_push() {
        let mut ds = OfflineDataset::new("test");
        assert!(ds.is_empty());
        ds.push(Trajectory::new());
        assert!(!ds.is_empty());
    }

    #[test]
    fn test_merge_with_empty() {
        let mut a = OfflineDataset::new("a");
        a.push(Trajectory::new());
        let empty = OfflineDataset::new("empty");
        a.merge(empty);
        assert_eq!(a.len(), 1);
    }

    #[test]
    fn test_merge_preserves_source() {
        let mut a = OfflineDataset::new("source-a");
        a.merge(OfflineDataset::new("source-b"));
        assert_eq!(a.metadata.source, "source-a");
    }

    #[test]
    fn test_export_jsonl_invalid_path() {
        let ds = OfflineDataset::new("x");
        let err = ds.export_jsonl("/nonexistent/dir/file.jsonl");
        assert!(err.is_err());
    }

    #[test]
    fn test_metadata_source_url_and_license() {
        let mut ds = OfflineDataset::new("x");
        ds.metadata.source_url = Some("https://example.com".to_string());
        ds.metadata.license = Some("Apache-2.0".to_string());
        assert_eq!(
            ds.metadata.source_url.as_deref(),
            Some("https://example.com")
        );
        assert_eq!(ds.metadata.license.as_deref(), Some("Apache-2.0"));
    }
}
